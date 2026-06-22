import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "indian_legal_judgments"

# same model as embedder — must match
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# cross encoder for reranking
# takes (query, document) pairs and scores relevance
# much more accurate than cosine similarity alone
# but slower — we run it only on top-20 candidates
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

logger.info("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

logger.info("Loading reranker model...")
reranker = CrossEncoder(RERANKER_MODEL)
logger.info("Models loaded")


def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(COLLECTION_NAME)


def preprocess_for_bm25(text: str) -> list[str]:
    """
    Tokenize text for BM25.
    BM25 works on token lists not raw strings.
    Legal text: lowercase, split on whitespace,
    remove punctuation but keep legal numbers like '138'
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    # remove very short tokens — noise in legal text
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


class HybridRetriever:
    """
    Combines dense retrieval (ChromaDB) with sparse retrieval (BM25).

    Why hybrid for legal text:
    - Dense retrieval is good at semantic similarity
      "anticipatory bail" finds "pre-arrest bail" — good
    - BM25 is good at exact term matching
      "Section 438 CrPC" finds chunks with that exact string — good
    - Legal queries need both:
      "What did court hold in Section 438 cases" needs
      semantic understanding AND exact section number matching
    - Neither alone is sufficient. Together they cover both failure modes.
    """

    def __init__(self, chunks_path: str = "data/processed/chunks.json"):
        self.collection = get_collection()

        # load all chunks for BM25 index
        # BM25 needs all documents in memory
        logger.info("Loading chunks for BM25 index...")
        with open(chunks_path) as f:
            self.all_chunks = json.load(f)

        # build BM25 index
        logger.info("Building BM25 index...")
        tokenized_corpus = [
            preprocess_for_bm25(c['text'])
            for c in self.all_chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunk_ids = [c['chunk_id'] for c in self.all_chunks]
        logger.info(f"BM25 index built — {len(self.all_chunks)} documents")

    def dense_search(
        self,
        query: str,
        n_results: int = 20,
        filter_case_type: str = None,
        filter_court_type: str = None,
        filter_year: str = None
    ) -> list[dict]:
        """
        Dense vector search using ChromaDB.
        Returns top-n semantically similar chunks.
        Supports metadata filtering before search.
        """
        query_embedding = embed_model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()[0]

        # build metadata filter
        where = None
        filters = {}
        if filter_case_type:
            filters["case_type"] = {"$eq": filter_case_type}
        if filter_court_type:
            filters["court_type"] = {"$eq": filter_court_type}
        if filter_year:
            filters["year"] = {"$eq": filter_year}

        if len(filters) == 1:
            where = filters
        elif len(filters) > 1:
            where = {"$and": [{k: v} for k, v in filters.items()]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            where=where
        )

        chunks = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            chunks.append({
                "text": doc,
                "metadata": meta,
                "dense_score": 1 - dist,
                "source": "dense"
            })

        return chunks

    def sparse_search(
        self,
        query: str,
        n_results: int = 20
    ) -> list[dict]:
        """
        BM25 sparse retrieval.
        Excellent at exact legal term matching:
        section numbers, act names, citation strings.
        """
        query_tokens = preprocess_for_bm25(query)
        scores = self.bm25.get_scores(query_tokens)

        # get top-n indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:n_results]

        chunks = []
        for idx in top_indices:
            if scores[idx] == 0:
                continue  # no term overlap at all — skip

            chunk = self.all_chunks[idx]
            chunks.append({
                "text": chunk['text'],
                "metadata": {
                    "chunk_id": chunk['chunk_id'],
                    "judgment_title": chunk['judgment_title'],
                    "year": chunk['year'],
                    "court_type": chunk['court_type'],
                    "court_name_normalized": chunk['court_name_normalized'],
                    "case_type": chunk['case_type'],
                    "petitioner": chunk['petitioner'],
                    "respondent": chunk['respondent'],
                    "doc_url": chunk['doc_url'],
                    "cites_count": chunk['cites_count'],
                    "cited_by_count": chunk['cited_by_count'],
                },
                "sparse_score": float(scores[idx]),
                "source": "sparse"
            })

        return chunks

    def reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        k: int = 60
    ) -> list[dict]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion.

        Why RRF instead of just averaging scores?
        Dense scores (cosine similarity) and BM25 scores are
        on completely different scales — you can't average them directly.
        RRF uses rank position instead of raw scores:
        RRF score = 1/(k + rank)
        k=60 is standard — dampens the impact of top ranks
        so a result ranked 1st in both lists scores highest.

        This is the standard fusion method used in production
        hybrid search systems.
        """
        scores = {}

        # score dense results by rank
        for rank, result in enumerate(dense_results):
            chunk_id = result['metadata'].get('chunk_id', '')
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "in_dense": False,
                    "in_sparse": False
                }
            scores[chunk_id]["rrf_score"] += 1 / (k + rank + 1)
            scores[chunk_id]["in_dense"] = True
            scores[chunk_id]["dense_score"] = result.get("dense_score", 0)

        # score sparse results by rank
        for rank, result in enumerate(sparse_results):
            chunk_id = result['metadata'].get('chunk_id', '')
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "result": result,
                    "rrf_score": 0.0,
                    "in_dense": False,
                    "in_sparse": False
                }
            scores[chunk_id]["rrf_score"] += 1 / (k + rank + 1)
            scores[chunk_id]["in_sparse"] = True
            scores[chunk_id]["sparse_score"] = result.get("sparse_score", 0)

        # sort by RRF score
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # return merged results
        fused = []
        for item in sorted_results:
            result = item["result"].copy()
            result["rrf_score"] = item["rrf_score"]
            result["in_dense"] = item["in_dense"]
            result["in_sparse"] = item["in_sparse"]
            fused.append(result)

        return fused

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5
    ) -> list[dict]:
        """
        Rerank candidates using cross-encoder.

        Why rerank?
        Embedding similarity asks: "are these vectors close?"
        Cross-encoder asks: "does this text actually answer this query?"
        Cross-encoder sees both query and document together —
        much more accurate but too slow to run on all 34k chunks.
        We run it only on top-20 candidates from hybrid search.

        This two-stage approach is standard in production:
        Stage 1 (fast): retrieve 20 candidates
        Stage 2 (accurate): rerank to get final top-5
        """
        if not candidates:
            return []

        # prepare pairs for cross-encoder
        pairs = [(query, c["text"][:512]) for c in candidates]

        # score all pairs
        scores = reranker.predict(pairs)

        # attach scores to candidates
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        # sort by rerank score
        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_case_type: str = None,
        filter_court_type: str = None,
        filter_year: str = None,
        use_reranker: bool = True
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline.
        Dense → Sparse → RRF Fusion → Rerank → Top-K

        This is the function your RAG pipeline calls.
        """
        # step 1: dense search — top 20
        dense = self.dense_search(
            query,
            n_results=20,
            filter_case_type=filter_case_type,
            filter_court_type=filter_court_type,
            filter_year=filter_year
        )

        # step 2: sparse BM25 search — top 20
        sparse = self.sparse_search(query, n_results=20)

        # step 3: fuse with RRF
        fused = self.reciprocal_rank_fusion(dense, sparse)

        # step 4: rerank top-20 fused results
        candidates = fused[:20]
        if use_reranker and candidates:
            final = self.rerank(query, candidates, top_k=top_k)
        else:
            final = candidates[:top_k]

        return final


def print_results(query: str, results: list[dict]):
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        meta = r['metadata']
        print(f"\nRank {i+1}")
        print(f"  Case: {meta.get('judgment_title','')[:65]}")
        print(f"  Court: {meta.get('court_type','')} | "
              f"Type: {meta.get('case_type','')} | "
              f"Year: {meta.get('year','')}")
        print(f"  Dense: {r.get('dense_score', 0):.3f} | "
              f"Rerank: {r.get('rerank_score', 0):.3f} | "
              f"In both: {r.get('in_dense') and r.get('in_sparse')}")
        print(f"  Text: {r['text'][:200]}...")


if __name__ == "__main__":
    retriever = HybridRetriever(
        chunks_path="data/processed/chunks.json"
    )

    # test 1 — criminal query
    results = retriever.retrieve(
        "anticipatory bail criminal accused arrested",
        filter_case_type="Criminal"
    )
    print_results("anticipatory bail criminal accused arrested", results)

    # test 2 — land query
    results = retriever.retrieve(
        "land acquisition compensation market value",
        filter_case_type="Land&Property"
    )
    print_results("land acquisition compensation market value", results)

    # test 3 — exact legal term — BM25 should shine here
    results = retriever.retrieve(
        "Section 438 CrPC anticipatory bail conditions"
    )
    print_results("Section 438 CrPC anticipatory bail conditions", results)