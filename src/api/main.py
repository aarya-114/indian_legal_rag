import time
import logging
import os
import threading
from collections import defaultdict, deque
from math import ceil
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except (ImportError, OSError):
    # PostgreSQL is optional; a missing or blocked native driver must not
    # prevent the API from serving retrieval and generation requests.
    psycopg2 = None
    RealDictCursor = None

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

QUERY_BODY_MAX_BYTES = 64 * 1024


def _positive_int_setting(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


QUERY_RATE_LIMIT_REQUESTS = _positive_int_setting("QUERY_RATE_LIMIT_REQUESTS", 30)
QUERY_RATE_LIMIT_WINDOW_SECONDS = _positive_int_setting(
    "QUERY_RATE_LIMIT_WINDOW_SECONDS", 60
)


class RequestBodyLimitMiddleware:
    """Bound POST /query bodies, including requests without Content-Length."""

    def __init__(self, app, max_body_bytes: int = QUERY_BODY_MAX_BYTES):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if not (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/query"
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = None  # Let the normal request parser reject it.
            if declared_size is not None and declared_size > self.max_body_bytes:
                await self._send_too_large(scope, receive, send)
                return

        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                await self._send_too_large(scope, receive, send)
                return
            more_body = message.get("more_body", False)

        body_bytes = bytes(body)
        body_sent = False

        async def receive_buffered():
            nonlocal body_sent
            if body_sent:
                return {"type": "http.disconnect"}
            body_sent = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        await self.app(scope, receive_buffered, send)

    async def _send_too_large(self, scope, receive, send):
        response = JSONResponse(
            status_code=413,
            content={
                "detail": (
                    "Request body exceeds the "
                    f"{self.max_body_bytes}-byte limit."
                )
            },
        )
        await response(scope, receive, send)


class QueryRateLimitMiddleware:
    """Apply a per-client in-memory sliding-window limit to POST /query."""

    def __init__(
        self,
        app,
        max_requests: int = QUERY_RATE_LIMIT_REQUESTS,
        window_seconds: int = QUERY_RATE_LIMIT_WINDOW_SECONDS,
    ):
        if max_requests <= 0 or window_seconds <= 0:
            raise ValueError("Rate-limit values must be positive integers.")
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests_by_client = defaultdict(deque)
        self.lock = threading.Lock()

    async def __call__(self, scope, receive, send):
        if not (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/query"
        ):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_id = client[0] if client else "unknown"
        now = time.monotonic()
        with self.lock:
            cutoff = now - self.window_seconds
            for key, timestamps in list(self.requests_by_client.items()):
                while timestamps and timestamps[0] <= cutoff:
                    timestamps.popleft()
                if not timestamps:
                    del self.requests_by_client[key]

            timestamps = self.requests_by_client[client_id]
            if len(timestamps) >= self.max_requests:
                retry_after = max(1, ceil(timestamps[0] + self.window_seconds - now))
            else:
                timestamps.append(now)
                retry_after = None

        if retry_after is not None:
            response = JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry later."},
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

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
app.add_middleware(
    QueryRateLimitMiddleware,
    max_requests=QUERY_RATE_LIMIT_REQUESTS,
    window_seconds=QUERY_RATE_LIMIT_WINDOW_SECONDS,
)
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=QUERY_BODY_MAX_BYTES,
)


@app.exception_handler(Exception)
async def safe_unexpected_error_handler(request: Request, exc: Exception):
    """Return a stable error body without leaking exception details."""
    logger.error("Unhandled API error (%s)", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── request/response models ──

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=10,
        max_length=4000,
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

    @validator("query")
    def query_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return value


class SourceDocument(BaseModel):
    source_id: str
    title: str
    court: str
    year: str
    case_type: str
    url: str
    relevance_score: float


class QueryScores(BaseModel):
    relevance: float
    faithfulness: float


class CitationValidation(BaseModel):
    referenced_source_ids: list[str] = Field(default_factory=list)
    unknown_source_ids: list[str] = Field(default_factory=list)


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
    citation_validation: CitationValidation = Field(default_factory=CitationValidation)
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
    This rough estimate uses fixed rates and does not query OpenRouter pricing.
    Formula: (input_tokens + output_tokens) * price_per_token
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
    if psycopg2 is None:
        logger.warning("PostgreSQL logging skipped: driver unavailable")
        return

    conn = None
    cur = None
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
    except Exception as e:
        logger.warning("Failed to log to PostgreSQL (%s)", type(e).__name__)
        # never crash the API because of logging failure
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception as e:
                logger.warning("Failed to close PostgreSQL cursor (%s)", type(e).__name__)
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.warning("Failed to close PostgreSQL connection (%s)", type(e).__name__)


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
            top_k=request.top_k,
            filter_case_type=request.filter_case_type
        )
        latency_ms = round((time.time() - start_time) * 1000, 2)
        cost_usd = estimate_cost(
            request.query,
            result["answer"],
            result["metadata"]["model"]
        )

        # Database logging is optional and must not discard a successful answer.
        try:
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
        except Exception as e:
            logger.warning("Optional PostgreSQL logging failed (%s)", type(e).__name__)

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=[SourceDocument(**s) for s in result["sources"]],
            citation_validation=CitationValidation(
                **result.get("citation_validation", {})
            ),
            scores=QueryScores(**result["scores"]),
            metadata=QueryMetadata(
                **result["metadata"],
                latency_ms=latency_ms,
                cost_usd=cost_usd
            ),
            warning=result.get("warning")
        )
    except Exception as e:
        stage = getattr(e, "stage", None)
        cause = getattr(e, "cause", e)
        if stage == "Answer generation":
            logger.warning("RAG answer generation failed (%s)", type(cause).__name__)
            raise HTTPException(
                status_code=502,
                detail="The language model could not complete the response. Please retry.",
            ) from e
        if stage == "Retry retrieval":
            logger.warning("RAG retrieval recovery failed (%s)", type(cause).__name__)
            raise HTTPException(
                status_code=503,
                detail="Retrieval is temporarily unavailable. Please retry.",
            ) from e
        logger.error("RAG query failed (%s)", type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail="Query processing failed. Please retry.",
        ) from e


@app.get("/stats")
async def stats():
    """
    Query statistics — cost tracking dashboard.
    Maps directly to Capgemini JD 'cost/ROI analysis'.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return {"message": "PostgreSQL not configured"}
    if psycopg2 is None:
        logger.warning("Stats unavailable: PostgreSQL driver unavailable")
        return {"message": "PostgreSQL unavailable"}

    conn = None
    cur = None
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
        return {"stats": [dict(r) for r in rows]}

    except Exception as e:
        logger.warning("Stats unavailable because PostgreSQL failed (%s)", type(e).__name__)
        return {"message": "PostgreSQL unavailable"}
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception as e:
                logger.warning("Failed to close PostgreSQL cursor (%s)", type(e).__name__)
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.warning("Failed to close PostgreSQL connection (%s)", type(e).__name__)
