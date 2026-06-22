import time
import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# global retriever — loaded once at startup
retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy models once at startup.
    Why lifespan instead of loading on each request?
    BM25 index + two neural models = ~4 seconds to load.
    Loading per request would make every query take 4+ seconds.
    Load once, reuse forever.
    """
    global retriever
    logger.info("Loading retriever — this takes ~10 seconds...")

    from src.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(
        chunks_path="data/processed/chunks.json"
    )
    logger.info("Retriever ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Indian Legal RAG API",
    description="Explainable Legal Intelligence System for Indian Supreme Court judgments",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request/response models ──

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Legal question to answer",
        example="What are the conditions for granting anticipatory bail?"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of chunks to retrieve"
    )
    filter_case_type: Optional[str] = Field(
        default=None,
        description="Filter by case type: Criminal, Land&Property, Tax, etc."
    )


class SourceDocument(BaseModel):
    title: str
    court: str
    year: str
    case_type: str
    url: str
    relevance_score: float


class QueryScores(BaseModel):
    relevance: float
    faithfulness: float


class QueryMetadata(BaseModel):
    detected_intent: str
    self_healed: bool
    chunks_retrieved: int
    model: str
    latency_ms: float
    cost_usd: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceDocument]
    scores: QueryScores
    metadata: QueryMetadata
    warning: Optional[str]
    disclaimer: str = (
        "This system provides legal research assistance only. "
        "It is not legal advice. Always consult a qualified "
        "Indian legal professional for legal matters."
    )


# ── cost tracking ──

def estimate_cost(query: str, answer: str, model: str) -> float:
    """
    Estimate cost per query.
    Groq free tier = $0.00 for now.
    Track it anyway — shows cost awareness in interviews.
    Formula: (input_tokens + output_tokens) * price_per_token
    Groq llama-3.1-8b: $0.05 per million input, $0.08 per million output
    """
    input_tokens = len(query) / 4  # rough: 1 token ≈ 4 chars
    output_tokens = len(answer) / 4
    input_cost = (input_tokens / 1_000_000) * 0.05
    output_cost = (output_tokens / 1_000_000) * 0.08
    return round(input_cost + output_cost, 6)


def log_to_postgres(
    query: str,
    answer: str,
    relevance: float,
    faithfulness: float,
    latency_ms: float,
    cost_usd: float,
    self_healed: bool,
    detected_intent: str
):
    """
    Log every query to PostgreSQL.
    Why log to postgres?
    - Track costs over time
    - Monitor quality degradation
    - Build eval dataset from real queries
    - Show cost/ROI analysis — maps to Capgemini JD
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return  # skip if postgres not configured

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO query_logs (
                query, answer_preview, relevance_score,
                faithfulness_score, latency_ms, cost_usd,
                self_healed, detected_intent, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            query,
            answer[:200],
            relevance,
            faithfulness,
            latency_ms,
            cost_usd,
            self_healed,
            detected_intent
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to log to postgres: {e}")
        # never crash the API because of logging failure


# ── endpoints ──

@app.get("/health")
async def health():
    """Health check — used by Cloud Run and load balancers."""
    return {
        "status": "healthy",
        "retriever_loaded": retriever is not None,
        "version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main RAG query endpoint.
    Receives legal question, returns grounded answer with citations.
    """
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not loaded — server starting up"
        )

    start_time = time.time()

    try:
        from src.api.rag_pipeline import rag_query
        result = rag_query(
            query=request.query,
            retriever=retriever,
            top_k=request.top_k
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

    latency_ms = round((time.time() - start_time) * 1000, 2)
    cost_usd = estimate_cost(
        request.query,
        result["answer"],
        result["metadata"]["model"]
    )

    # log to postgres asynchronously
    log_to_postgres(
        query=request.query,
        answer=result["answer"],
        relevance=result["scores"]["relevance"],
        faithfulness=result["scores"]["faithfulness"],
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        self_healed=result["metadata"]["self_healed"],
        detected_intent=result["metadata"]["detected_intent"]
    )

    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result["sources"]],
        scores=QueryScores(**result["scores"]),
        metadata=QueryMetadata(
            **result["metadata"],
            latency_ms=latency_ms,
            cost_usd=cost_usd
        ),
        warning=result.get("warning")
    )


@app.get("/stats")
async def stats():
    """
    Query statistics — cost tracking dashboard.
    Maps directly to Capgemini JD 'cost/ROI analysis'.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return {"message": "PostgreSQL not configured"}

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT
                COUNT(*) as total_queries,
                ROUND(AVG(relevance_score)::numeric, 3) as avg_relevance,
                ROUND(AVG(faithfulness_score)::numeric, 3) as avg_faithfulness,
                ROUND(AVG(latency_ms)::numeric, 0) as avg_latency_ms,
                ROUND(SUM(cost_usd)::numeric, 6) as total_cost_usd,
                ROUND(AVG(cost_usd)::numeric, 6) as avg_cost_per_query,
                SUM(CASE WHEN self_healed THEN 1 ELSE 0 END) as self_heal_count,
                detected_intent,
                COUNT(*) as intent_count
            FROM query_logs
            GROUP BY detected_intent
            ORDER BY intent_count DESC
        """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        return {"stats": [dict(r) for r in rows]}

    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))