import os
import logging
import json
import re
from dotenv import load_dotenv
from groq import Groq
from src.retrieval.hybrid_retriever import HybridRetriever , print_results

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Groq free tier — llama3 is fast and good for legal reasoning
LLM_MODEL = "llama-3.1-8b-instant"

# confidence thresholds
# below these — self healing kicks in
FAITHFULNESS_THRESHOLD = 0.60
RELEVANCE_THRESHOLD = 0.55


def classify_query_intent(query: str) -> dict:
    """
    Classify what kind of legal query this is.
    Determines metadata filters to apply before retrieval.

    Why do this before retrieval?
    "What did the court hold on bail in criminal cases?"
    Without classification — searches all 34k chunks
    With classification — searches only 5,675 Criminal chunks
    Precision improves dramatically.
    """
    query_lower = query.lower()

    # case type detection
    case_type = None
    if any(word in query_lower for word in [
        "bail", "criminal", "accused", "arrested", "fir",
        "murder", "theft", "ipc", "crpc", "bnss"
    ]):
        case_type = "Criminal"
    elif any(word in query_lower for word in [
        "land", "property", "acquisition", "compensation",
        "eviction", "rent", "lease", "possession"
    ]):
        case_type = "Land&Property"
    elif any(word in query_lower for word in [
        "tax", "income", "gst", "revenue", "assessment",
        "penalty", "refund", "deduction"
    ]):
        case_type = "Tax"
    elif any(word in query_lower for word in [
        "labour", "worker", "employee", "termination",
        "industrial", "strike", "union", "wages"
    ]):
        case_type = "Industrial&Labour"
    elif any(word in query_lower for word in [
        "constitution", "fundamental", "article", "rights",
        "writ", "mandamus", "certiorari"
    ]):
        case_type = "Constitution"

    # court type preference
    court_type = None
    if any(word in query_lower for word in [
        "supreme court", "apex court", "sc judgment"
    ]):
        court_type = "Supreme_Court"

    return {
        "case_type": case_type,
        "court_type": court_type,
        "detected_intent": case_type or "General"
    }


def score_relevance(query: str, chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    scores = [c.get("rerank_score", 0) for c in chunks]
    avg_score = sum(scores) / len(scores)
    # cross-encoder range is roughly -10 to +10
    # normalize to 0-1
    normalized = (avg_score + 10) / 20.0
    return max(0.0, min(1.0, normalized))


def score_faithfulness(answer: str, chunks: list[dict]) -> float:
    """
    Check if the generated answer is grounded in retrieved chunks.

    Simple but effective approach:
    1. Extract key phrases from answer
    2. Check how many appear in source chunks
    3. Score = fraction of answer phrases found in sources

    Why not use RAGAS here?
    RAGAS needs OpenAI. We use Groq.
    This rule-based approach catches the most dangerous
    failure mode — numbers and names not in source.

    Production note: in a real system you'd use an NLI model
    for more accurate faithfulness scoring.
    """
    if not answer or not chunks:
        return 0.0

    combined_source = " ".join(c["text"] for c in chunks).lower()

    # extract numbers — most dangerous hallucination type in legal
    answer_numbers = re.findall(r'\b\d+\b', answer)
    answer_legal_refs = re.findall(
        r'section\s+\d+|article\s+\d+|act\s+\d{4}',
        answer.lower()
    )

    all_claims = answer_numbers + answer_legal_refs

    if not all_claims:
        # no verifiable claims — give moderate score
        return 0.70

    # check how many claims appear in source
    found = sum(
        1 for claim in all_claims
        if claim.lower() in combined_source
    )

    faithfulness = found / len(all_claims)
    return faithfulness


def rewrite_query(query: str, previous_results_summary: str) -> str:
    prompt = f"""You are an Indian legal expert helping improve a search query for Indian court judgments.

Original query: {query}

Rewrite this query using:
- Indian legal terminology (IPC, CrPC, BNSS, Indian Acts)
- Specific Indian Act names and Section numbers if relevant
- Alternative Indian legal terms for the same concept
- Reference to Indian courts (High Court, Supreme Court of India)

Return ONLY the rewritten query, nothing else."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.3
    )
    rewritten = response.choices[0].message.content.strip()
    logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
    return rewritten


def generate_answer(
    query: str,
    chunks: list[dict],
    query_intent: dict
) -> str:
    """
    Generate grounded answer using Groq LLM.
    Context is assembled from retrieved chunks with source citations.
    """
    # assemble context with source information
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        source = (f"{meta.get('judgment_title', 'Unknown')[:60]} "
                  f"({meta.get('court_type', '')}, "
                  f"{meta.get('year', '')})")
        context_parts.append(f"[Source {i+1}: {source}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an Indian legal research assistant. Answer the question using ONLY the provided legal sources.

STRICT RULES:
1. Only use information from the provided sources
2. Always cite which source supports each claim using [Source N]
3. If the sources don't contain enough information, say so explicitly
4. Never make up case names, section numbers, or legal facts
5. Keep the answer focused and precise

Question: {query}

Legal Sources:
{context}

Answer:"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.1  # low temperature — factual legal answers
    )

    return response.choices[0].message.content.strip()


def rag_query(
    query: str,
    retriever: HybridRetriever,
    top_k: int = 5
) -> dict:
    """
    Full RAG pipeline with self-healing.

    Flow:
    1. Classify query intent → metadata filters
    2. Retrieve with hybrid search
    3. Score relevance
    4. If relevance low → rewrite query → retry (self-healing)
    5. Generate grounded answer
    6. Score faithfulness
    7. If faithfulness low → flag response
    8. Return answer with full provenance

    This is the function your FastAPI endpoint calls.
    """
    logger.info(f"Processing query: {query[:80]}")

    # step 1 — classify intent
    intent = classify_query_intent(query)
    logger.info(f"Detected intent: {intent['detected_intent']}")

    # step 2 — retrieve
    chunks = retriever.retrieve(
        query=query,
        top_k=top_k,
        filter_case_type=intent["case_type"],
        filter_court_type=intent["court_type"]
    )

    # step 3 — score relevance
    relevance_score = score_relevance(query, chunks)
    logger.info(f"Relevance score: {relevance_score:.3f}")

    # step 4 — self-healing if relevance low
    healed = False
    if relevance_score < RELEVANCE_THRESHOLD:
        logger.warning(
            f"Low relevance ({relevance_score:.3f}) — triggering self-heal"
        )

        # rewrite query
        rewritten_query = rewrite_query(query, "")

        # retry with rewritten query and wider search (no filter)
        chunks = retriever.retrieve(
            query=rewritten_query,
            top_k=top_k,
            filter_case_type=None,  # widen search
            filter_court_type=None
        )

        new_relevance = score_relevance(rewritten_query, chunks)
        logger.info(f"Post-heal relevance: {new_relevance:.3f}")

        healed = True
        relevance_score = new_relevance

    # step 5 — generate answer
    answer = generate_answer(query, chunks, intent)

    # step 6 — score faithfulness
    faithfulness_score = score_faithfulness(answer, chunks)
    logger.info(f"Faithfulness score: {faithfulness_score:.3f}")

    # step 7 — build sources list
    sources = []
    seen_titles = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        title = meta.get("judgment_title", "")
        if title not in seen_titles:
            sources.append({
                "title": title,
                "court": meta.get("court_type", ""),
                "year": meta.get("year", ""),
                "case_type": meta.get("case_type", ""),
                "url": meta.get("doc_url", ""),
                "relevance_score": round(
                    chunk.get("rerank_score", 0), 3)
            })
            seen_titles.add(title)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "scores": {
            "relevance": round(relevance_score, 3),
            "faithfulness": round(faithfulness_score, 3),
        },
        "metadata": {
            "detected_intent": intent["detected_intent"],
            "self_healed": healed,
            "chunks_retrieved": len(chunks),
            "model": LLM_MODEL
        },
        "warning": (
            "Low faithfulness — verify answer against sources"
            if faithfulness_score < FAITHFULNESS_THRESHOLD
            else None
        )
    }


if __name__ == "__main__":
    import json

    retriever = HybridRetriever(
        chunks_path="data/processed/chunks.json"
    )

    queries = [
        "What are the conditions for granting anticipatory bail?",
        "How is compensation determined in land acquisition cases?",
        "What is the court's position on wrongful termination of employees?"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        result = rag_query(query, retriever)
        print(f"Query: {result['query']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nScores: {result['scores']}")
        print(f"Self-healed: {result['metadata']['self_healed']}")
        print(f"Warning: {result['warning']}")
        print(f"\nSources:")
        for s in result['sources']:
            print(f"  - {s['title'][:60]} ({s['year']})")