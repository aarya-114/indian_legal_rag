import json
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "indian_legal_judgments"

# local embedding model — no API cost, no rate limits
# all-MiniLM-L6-v2: 384 dims, fast on CPU, good semantic quality
# for legal text specifically this performs well on English
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64  # larger batches fine for local model

# load model once at module level — expensive to reload
logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
logger.info("Embedding model loaded")


def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Collection '{COLLECTION_NAME}' — "
                f"existing docs: {collection.count()}")
    return collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using local sentence-transformers model.
    No API calls, no cost, no rate limits.
    """
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True  # normalize for cosine similarity
    )
    return embeddings.tolist()


def build_metadata(chunk: dict) -> dict:
    """
    ChromaDB metadata — must be str, int, float, or bool only.
    """
    return {
        "chunk_id": str(chunk.get("chunk_id", "")),
        "judgment_title": str(chunk.get("judgment_title", ""))[:500],
        "year": str(chunk.get("year", "Unknown")),
        "court_type": str(chunk.get("court_type", "")),
        "court_name_normalized": str(
            chunk.get("court_name_normalized", ""))[:200],
        "case_type": str(chunk.get("case_type", "")),
        "petitioner": str(chunk.get("petitioner", ""))[:200],
        "respondent": str(chunk.get("respondent", ""))[:200],
        "doc_url": str(chunk.get("doc_url", "")),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "total_chunks": int(chunk.get("total_chunks", 0)),
        "cites_count": int(chunk.get("cites_count", 0)),
        "cited_by_count": int(chunk.get("cited_by_count", 0)),
        "text_length": int(chunk.get("text_length", 0)),
        "predicted_section": str(
            chunk.get("predicted_section", "UNKNOWN")),
    }


def embed_and_store_chunks(
    chunks_path: str,
    max_chunks: int = None,
    skip_existing: bool = True
):
    """
    Embed all chunks and store in ChromaDB.
    Completely free — runs on local CPU.
    """
    with open(chunks_path) as f:
        chunks = json.load(f)

    if max_chunks:
        chunks = chunks[:max_chunks]
        logger.info(f"Using subset: {max_chunks} chunks")

    collection = get_chroma_collection()

    # skip already embedded chunks — idempotent pipeline
    existing_ids = set()
    if skip_existing and collection.count() > 0:
        existing = collection.get(include=[])
        existing_ids = set(existing['ids'])
        logger.info(f"Skipping {len(existing_ids)} existing chunks")

    new_chunks = [
        c for c in chunks
        if c['chunk_id'] not in existing_ids
    ]
    logger.info(f"Chunks to embed: {len(new_chunks)}")

    if not new_chunks:
        logger.info("Nothing to embed — already complete")
        return collection

    total_embedded = 0
    start_time = time.time()

    for batch_start in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[batch_start: batch_start + BATCH_SIZE]

        # truncate oversized chunks
        texts = []
        for chunk in batch:
            text = chunk['text']
            # MiniLM max: 512 tokens ≈ 2000 chars
            if len(text) > 2000:
                text = text[:2000]
            texts.append(text)

        embeddings = embed_texts(texts)

        collection.add(
            ids=[c['chunk_id'] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[build_metadata(c) for c in batch]
        )

        total_embedded += len(batch)

        if batch_start % (BATCH_SIZE * 10) == 0:
            elapsed = time.time() - start_time
            rate = total_embedded / elapsed if elapsed > 0 else 0
            remaining = len(new_chunks) - total_embedded
            eta = remaining / rate if rate > 0 else 0
            logger.info(
                f"Embedded {total_embedded}/{len(new_chunks)} "
                f"— {rate:.0f} chunks/sec "
                f"— ETA: {eta:.0f}s"
            )

    elapsed = time.time() - start_time
    logger.info(f"Done — {total_embedded} chunks in {elapsed:.1f}s")
    logger.info(f"Cost: $0.00 — local model")
    logger.info(f"Collection size: {collection.count()}")
    return collection


def test_retrieval(query: str, n_results: int = 5,
                   filter_case_type: str = None):
    """
    Test retrieval with optional metadata filtering.
    This is your dense retrieval — one half of hybrid search.
    """
    collection = get_chroma_collection()

    if collection.count() == 0:
        logger.error("Collection empty — run embedding first")
        return

    query_embedding = embed_texts([query])[0]

    # metadata filter — only return chunks matching case type
    where = None
    if filter_case_type:
        where = {"case_type": {"$eq": filter_case_type}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
        where=where
    )

    print(f"\n=== RETRIEVAL TEST ===")
    print(f"Query: {query}")
    if filter_case_type:
        print(f"Filter: case_type = {filter_case_type}")
    print()

    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        similarity = 1 - dist
        print(f"Result {i+1} — similarity: {similarity:.3f}")
        print(f"  Case: {meta['judgment_title'][:70]}")
        print(f"  Court: {meta['court_type']} | "
              f"Type: {meta['case_type']} | "
              f"Year: {meta['year']}")
        print(f"  Text: {doc[:200]}...")
        print()


if __name__ == "__main__":

    embed_and_store_chunks(
        chunks_path="data/processed/chunks.json",
        max_chunks=None  # remove the limit
        )

    # test with and without metadata filter
    test_retrieval("anticipatory bail criminal accused")
    test_retrieval(
        "property land acquisition compensation",
        filter_case_type="Land&Property"
    )