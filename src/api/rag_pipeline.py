import os
import logging
import json
import re
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
if not LLM_MODEL:
    raise RuntimeError(
        "LLM_MODEL is required. Set LLM_MODEL=<model-name> in your .env file."
    )

try:
    LLM_REQUEST_TIMEOUT_SECONDS = float(
        os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60")
    )
except ValueError as exc:
    raise RuntimeError("LLM_REQUEST_TIMEOUT_SECONDS must be a positive number.") from exc
if not LLM_REQUEST_TIMEOUT_SECONDS > 0 or LLM_REQUEST_TIMEOUT_SECONDS == float("inf"):
    raise RuntimeError("LLM_REQUEST_TIMEOUT_SECONDS must be a positive number.")

try:
    LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1536"))
except ValueError as exc:
    raise RuntimeError("LLM_MAX_OUTPUT_TOKENS must be a positive integer.") from exc
if LLM_MAX_OUTPUT_TOKENS <= 0:
    raise RuntimeError("LLM_MAX_OUTPUT_TOKENS must be a positive integer.")

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    timeout=LLM_REQUEST_TIMEOUT_SECONDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

from src.retrieval.hybrid_retriever import HybridRetriever, print_results


class LLMResponseError(RuntimeError):
    """Raised when the configured LLM returns no usable assistant text."""

    def __init__(self, message: str, finish_reason: str | None = None):
        self.finish_reason = finish_reason
        super().__init__(message)


# Retain the former exception name for callers that imported it directly.
OpenRouterResponseError = LLMResponseError


class RAGGenerationError(RuntimeError):
    """Generation failure carrying the retrieval score already computed."""

    def __init__(self, stage: str, relevance_score: float,
                 self_healed: bool, cause: Exception,
                 rewrite_attempted: bool = False,
                 retry_attempted: bool = False,
                 rewrite_succeeded: bool = False,
                 rewrite_failed: bool = False,
                 retry_succeeded: bool = False,
                 retry_failed: bool = False,
                 retrieval_diagnostics: dict | None = None):
        self.stage = stage
        self.relevance_score = relevance_score
        self.self_healed = self_healed
        self.rewrite_attempted = rewrite_attempted
        self.retry_attempted = retry_attempted
        self.rewrite_succeeded = rewrite_succeeded
        self.rewrite_failed = rewrite_failed
        self.retry_succeeded = retry_succeeded
        self.retry_failed = retry_failed
        self.retrieval_diagnostics = retrieval_diagnostics or {}
        self.cause = cause
        super().__init__(f"{stage} failed after retrieval: {cause}")


def extract_response_text(response, stage: str) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise LLMResponseError(
            f"LLM {stage} response contained no choices."
        )

    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        response_id = getattr(response, "id", None)
        raise LLMResponseError(
            f"LLM {stage} response was truncated at the token limit "
            f"(finish_reason='length', response_id={response_id!r}).",
            finish_reason=finish_reason,
        )

    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if content is None:
        # A refusal is still useful assistant text; otherwise the response is
        # unavailable and must not be scored as an empty answer.
        content = getattr(message, "refusal", None)

    if not isinstance(content, str) or not content.strip():
        response_id = getattr(response, "id", None)
        raise LLMResponseError(
            f"LLM {stage} response contained no assistant text "
            f"(finish_reason={finish_reason!r}, response_id={response_id!r}).",
            finish_reason=finish_reason,
        )
    return content.strip()

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


def extract_statutory_section_reference(query: str) -> str | None:
    """Return one explicit section reference in a stable searchable form."""
    match = re.search(
        r"\b(?:section|sec\.?)\s*(\d+)\s*[-‐‑‒–—−\s]*([A-Za-z]?)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return f"section {match.group(1)}{match.group(2).lower()}"


def score_relevance(query: str, chunks: list[dict]) -> float:
    """Return the mean reranker score linearly normalized to [0, 1].

    This is a retrieval ranking score, not a probability of correctness.
    """
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
    Apply a narrow rule-based check for numeric and legal-reference claims.
    This does not assess semantic entailment or legal correctness.

    Simple but effective approach:
    1. Extract key phrases from answer
    2. Check how many appear in source chunks
    3. Score = fraction of answer phrases found in sources

    Why not use RAGAS here?
    RAGAS would require a separate judge integration. This rule-based
    score is independent of the OpenRouter generation provider.
    This rule-based approach catches the most dangerous
    failure mode — numbers and names not in source.

    A zero means no extracted numeric/legal-reference claim matched the
    retrieved text (or that the answer/chunk input was empty).
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
    prompt = f"""Rewrite the original as one concise search query for Indian court judgments.
Preserve every legal/content term in the original and keep those terms in the
same order. You may remove only articles (a, an, the). Do not add facts, parties,
statutes, section numbers, case names, or assumptions absent from the original.
Return only one query of at most 20 words, with no explanation, alternatives,
markdown, or label.

Original query: {query}"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=512,
        temperature=0,
        reasoning_effort="low",
    )
    rewritten = extract_response_text(response, "query rewrite")
    words = rewritten.split()
    original_lower = query.casefold()
    articles = {"a", "an", "the"}

    def content_tokens(text: str) -> list[str]:
        return [
            token.casefold()
            for token in re.findall(r"\b[\w]+\b", text)
            if token.casefold() not in articles
        ]

    preserves_query_terms = content_tokens(rewritten) == content_tokens(query)
    reference_patterns = (
        # Common Indian statute abbreviations and named legal instruments.
        r"\b(?:IPC|CrPC|BNSS|BNS|IEA|BSA|CPC|Companies Act|"
        r"Indian Penal Code|Code of Criminal Procedure|"
        r"Constitution of India)\b",
        r"\b(?:[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,5}\s+"
        r"(?:Act|Code|Rules|Constitution))\b",
        r"\b(?:section|sec\.?|article)\s+\d+[A-Za-z]?\b",
        r"\b\d{4}\b",
        # Case-name form, including common v./vs./versus spellings.
        r"\b[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,5}\s+"
        r"v(?:s\.?|ersus)\s+[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,5}\b",
    )
    unsupported_reference = any(
        match.group(0).casefold() not in original_lower
        for pattern in reference_patterns
        for match in re.finditer(pattern, rewritten, flags=re.IGNORECASE)
    )
    unsupported_case_name = any(
        match.group(0).casefold() not in original_lower
        for match in re.finditer(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b", rewritten
        )
    )
    if (
        len(words) > 20
        or "\n" in rewritten
        or rewritten.startswith(("-", "*", "1.", "Query:"))
        or rewritten.startswith(('"', "'"))
        or rewritten.endswith(('"', "'"))
        or not preserves_query_terms
        or unsupported_reference
        or unsupported_case_name
    ):
        raise LLMResponseError(
            "LLM query rewrite response was not one safe, concise query."
        )
    logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
    return rewritten


def generate_answer(
    query: str,
    chunks: list[dict],
    query_intent: dict
) -> str:
    """
    Generate a bounded, grounded answer using the configured Groq model.
    Context is assembled from retrieved chunks with source citations.
    """
    # Number sources at judgment level, matching the API source records.
    sources = build_judgment_sources(chunks)
    context_parts = []
    for source in sources:
        label_title = source["title"][:60] or "Judgment metadata unavailable"
        court_year = ", ".join(value for value in (
            source["court"], source["year"],
        ) if value)
        suffix = f" ({court_year})" if court_year else ""
        label = f"[Source {source['source_id']}: {label_title}{suffix}]"
        passages = "\n\n".join(source["passages"])
        context_parts.append(f"{label}\n{passages}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an Indian legal research assistant. Answer the question using ONLY the provided legal sources.

STRICT RULES:
1. Only use information from the provided sources
2. Always cite which source supports each claim using [Source N]
3. If the sources don't contain enough information, say so explicitly
4. Never make up case names, section numbers, or legal facts
5. Keep the answer focused and precise; use no more than 400 words

Question: {query}

Legal Sources:
{context}

Answer:"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=LLM_MAX_OUTPUT_TOKENS,
        temperature=0.1,  # low temperature — factual legal answers
        reasoning_effort="low",
    )

    return extract_response_text(response, "answer generation")


def build_judgment_sources(chunks: list[dict]) -> list[dict]:
    """Group retrieved passages by document identity and assign query-local IDs.

    URLs are preferred. When unavailable, the judgment prefix in chunk_id is
    used. Chunks lacking both document identities are omitted: title alone is
    not enough to safely identify or cite a judgment.
    """
    grouped = {}
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        url = meta.get("doc_url")
        url_key = url.strip().rstrip("/") if isinstance(url, str) else ""
        chunk_id = meta.get("chunk_id")
        chunk_id = chunk_id if isinstance(chunk_id, str) else ""
        match = re.match(r"^(.*)_chunk_\d+$", chunk_id)
        if not url_key and not match:
            continue
        identity = ("url", url_key) if url_key else ("judgment", match.group(1))
        if identity not in grouped:
            grouped[identity] = {
                "source_id": str(len(grouped) + 1),
                "title": str(meta.get("judgment_title") or ""),
                "court": str(meta.get("court_type") or ""),
                "year": str(meta.get("year") or ""),
                "case_type": str(meta.get("case_type") or ""),
                "url": url.strip() if isinstance(url, str) else "",
                "relevance_score": round(chunk.get("rerank_score", 0), 3),
                "passages": [],
                "chunk_ids": [],
            }
        text = chunk.get("text")
        if isinstance(text, str) and text.strip():
            grouped[identity]["passages"].append(text)
        if chunk_id and chunk_id not in grouped[identity]["chunk_ids"]:
            grouped[identity]["chunk_ids"].append(chunk_id)
    sources = [source for source in grouped.values() if source["passages"]]
    # IDs are assigned after removing empty passages so prompt and API numbers
    # remain contiguous and only identify evidence actually supplied to the LLM.
    for source_id, source in enumerate(sources, start=1):
        source["source_id"] = str(source_id)
    return sources


def validate_source_references(answer: str, sources: list[dict]) -> dict:
    """Check citation IDs exist; this does not assess evidentiary support."""
    valid_ids = {str(source.get("source_id")) for source in sources}
    referenced = list(dict.fromkeys(
        re.findall(
            r"\[\s*Source\s+(\d+)(?:\s*:[^\]]*)?\s*\]",
            answer or "",
            re.IGNORECASE,
        )
    ))
    return {
        "referenced_source_ids": referenced,
        "unknown_source_ids": [source_id for source_id in referenced if source_id not in valid_ids],
    }


def rag_query(
    query: str,
    retriever: HybridRetriever,
    top_k: int = 5,
    filter_case_type: str = None
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
    requested_case_type = filter_case_type or intent["case_type"]
    required_reference = extract_statutory_section_reference(query)
    retrieval_diagnostics = {
        "initial_chunk_ids": [],
        "fallback_candidate_ids": [],
        "fallback_chunk_ids": [],
        "final_chunk_ids": [],
        "fallback_activated": False,
        "reference_recovered": False,
        "fallback_error": None,
    }
    reference_retrieval = getattr(
        type(retriever), "retrieve_with_reference_fallback", None
    )
    if required_reference and reference_retrieval is not None:
        chunks, retrieval_diagnostics = retriever.retrieve_with_reference_fallback(
            query=query,
            required_reference=required_reference,
            top_k=top_k,
            filter_case_type=requested_case_type,
            # Keep explicit caller filters; remove only the inferred filter.
            fallback_case_type=filter_case_type,
            filter_court_type=intent["court_type"],
        )
    else:
        chunks = retriever.retrieve(
            query=query,
            top_k=top_k,
            filter_case_type=requested_case_type,
            filter_court_type=intent["court_type"]
        )
        retrieval_diagnostics["initial_chunk_ids"] = [
            c.get("metadata", {}).get("chunk_id") for c in chunks
        ]
        retrieval_diagnostics["final_chunk_ids"] = list(
            retrieval_diagnostics["initial_chunk_ids"]
        )

    # step 3 — score relevance
    relevance_score = score_relevance(query, chunks)
    logger.info(f"Relevance score: {relevance_score:.3f}")

    # step 4 — self-healing if relevance low
    rewrite_attempted = False
    rewrite_succeeded = False
    rewrite_failed = False
    retry_attempted = False
    retry_succeeded = False
    retry_failed = False
    healed = False
    fallback_activated = retrieval_diagnostics.get("fallback_activated", False)
    if fallback_activated:
        # The reference fallback consumes the single recovery/retry budget.
        retry_attempted = True
        retry_failed = bool(retrieval_diagnostics.get("fallback_error"))
        retry_succeeded = not retry_failed
    if relevance_score < RELEVANCE_THRESHOLD and not fallback_activated:
        logger.warning(
            f"Low relevance ({relevance_score:.3f}) — triggering self-heal"
        )

        # rewrite query
        rewrite_attempted = True
        retry_query = query
        try:
            rewritten_query = rewrite_query(query, "")
            rewrite_succeeded = True
            retry_query = rewritten_query
        except Exception as exc:
            rewrite_failed = True
            logger.warning(
                "Query rewrite failed (%s); retrying the original query with "
                "the broader filter.", type(exc).__name__,
            )

        # Retry once. A rewrite failure falls back to the original query.
        retry_attempted = True
        try:
            chunks = retriever.retrieve(
                query=retry_query,
                top_k=top_k,
                filter_case_type=filter_case_type,  # preserve caller filter
                filter_court_type=None
            )
        except Exception as exc:
            retry_failed = True
            raise RAGGenerationError(
                "Retry retrieval", relevance_score, False, exc,
                rewrite_attempted=rewrite_attempted,
                retry_attempted=retry_attempted,
                rewrite_succeeded=rewrite_succeeded,
                rewrite_failed=rewrite_failed,
                retry_succeeded=retry_succeeded,
                retry_failed=retry_failed,
                retrieval_diagnostics=retrieval_diagnostics,
            ) from exc

        retry_succeeded = True
        retrieval_diagnostics["final_chunk_ids"] = [
            c.get("metadata", {}).get("chunk_id") for c in chunks
        ]
        new_relevance = score_relevance(retry_query, chunks)
        logger.info(f"Post-heal relevance: {new_relevance:.3f}")

        # A retry counts as a recovery only when it crosses the same threshold
        # that triggered self-healing.
        recovery_candidate = new_relevance >= RELEVANCE_THRESHOLD
        relevance_score = new_relevance
    else:
        recovery_candidate = bool(
            fallback_activated
            and retrieval_diagnostics.get("reference_recovered")
            and not retrieval_diagnostics.get("fallback_error")
        )

    # step 5 — generate answer
    try:
        answer = generate_answer(query, chunks, intent)
    except Exception as exc:
        raise RAGGenerationError(
            "Answer generation", relevance_score, False, exc,
            rewrite_attempted=rewrite_attempted,
            retry_attempted=retry_attempted,
            rewrite_succeeded=rewrite_succeeded,
            rewrite_failed=rewrite_failed,
            retry_succeeded=retry_succeeded,
            retry_failed=retry_failed,
            retrieval_diagnostics=retrieval_diagnostics,
        ) from exc

    # Recovery requires adequate retry relevance and a completed answer.
    healed = recovery_candidate

    # step 6 — score faithfulness
    faithfulness_score = score_faithfulness(answer, chunks)
    logger.info(f"Faithfulness score: {faithfulness_score:.3f}")

    # Use the same judgment-level IDs and identity rules as the prompt.
    grouped_sources = build_judgment_sources(chunks)
    sources = [
        {key: source[key] for key in (
            "source_id", "title", "court", "year", "case_type", "url",
            "relevance_score",
        )}
        for source in grouped_sources
    ]
    citation_validation = validate_source_references(answer, sources)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "citation_validation": citation_validation,
        "scores": {
            "relevance": round(relevance_score, 3),
            "faithfulness": round(faithfulness_score, 3),
        },
        "metadata": {
            "detected_intent": intent["detected_intent"],
            "self_healed": healed,
            "rewrite_attempted": rewrite_attempted,
            "rewrite_succeeded": rewrite_succeeded,
            "rewrite_failed": rewrite_failed,
            "retry_attempted": retry_attempted,
            "retry_succeeded": retry_succeeded,
            "retry_failed": retry_failed,
            "initial_chunk_ids": retrieval_diagnostics.get("initial_chunk_ids", []),
            "fallback_candidate_ids": retrieval_diagnostics.get("fallback_candidate_ids", []),
            "fallback_chunk_ids": retrieval_diagnostics.get("fallback_chunk_ids", []),
            "final_chunk_ids": retrieval_diagnostics.get("final_chunk_ids", []),
            "fallback_activated": fallback_activated,
            "reference_recovered": retrieval_diagnostics.get("reference_recovered", False),
            "fallback_error": retrieval_diagnostics.get("fallback_error"),
            "chunks_retrieved": len(chunks),
            "model": LLM_MODEL
        },
        "warning": (
            "; ".join(filter(None, [
                "Low faithfulness — verify answer against sources"
                if faithfulness_score < FAITHFULNESS_THRESHOLD else None,
                "Answer cites unknown source ID(s): " + ", ".join(
                    citation_validation["unknown_source_ids"]
                ) if citation_validation["unknown_source_ids"] else None,
            ])) or None
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
