# Indian Legal RAG

A self-healing RAG system over Indian court judgments, with hybrid retrieval, automated faithfulness checking, and LLM evaluation CI/CD.

## What this is

Generic RAG fails on Indian legal text — dense citation networks, archaic statutory language, and exact section-number references that semantic search alone misses. This project builds a retrieval-augmented generation pipeline specifically for Indian court judgments that:

- Combines dense vector search with BM25 keyword search, fused with Reciprocal Rank Fusion
- Reranks candidates with a cross-encoder for accuracy
- Automatically detects low-confidence retrieval and retries with a legally-rephrased query (self-healing)
- Verifies every generated answer's numbers and citations against the retrieved source text (faithfulness check)
- Runs an automated evaluation suite on every pull request, blocking merges if quality regresses

## Architecture
HuggingFace (OpenNyAI: judgments + rhetorical role labels)

↓

Ingestion → Cleaning → Chunking (sentence-boundary, 1000 chars, 200 overlap)

↓

Local embeddings (sentence-transformers) → ChromaDB

↓

FastAPI

↓

Query Intent Classifier

↓              ↓

Dense (ChromaDB)   Sparse (BM25)

↓              ↓

Reciprocal Rank Fusion

↓

Cross-Encoder Reranker

↓

Relevance Check

(low? → rewrite query → retry)

↓

Groq Generation (grounded, cited)

↓

Faithfulness Check

↓

PostgreSQL logging 

↓

Response with sources + scores

## Known limitations
- Cost/quality logging to PostgreSQL is implemented but optional — the app runs fully without a database configured; query logging silently skips if `DATABASE_URL` isn't set

## Why PostgreSQL
PostgreSQL is not part of the RAG pipeline itself — retrieval, generation, and faithfulness checking all work independently of it. It exists solely as a query log: every call to `/query` writes a row recording the question, scores, latency, and estimated cost. This powers the `/stats` endpoint, which aggregates cost-per-query and quality trends over time. The system runs fully without it — logging is skipped silently if `DATABASE_URL` isn't set — but with it, the project demonstrates basic cost/ROI tracking and SQL-based analytics on top of the AI pipeline.
# create database and table
createdb legal_rag
psql legal_rag -c "
CREATE TABLE query_logs (
    id SERIAL PRIMARY KEY,
    query TEXT,
    answer_preview TEXT,
    relevance_score FLOAT,
    faithfulness_score FLOAT,
    latency_ms FLOAT,
    cost_usd FLOAT,
    self_healed BOOLEAN,
    detected_intent TEXT,
    created_at TIMESTAMP
);
"



## Results

| Metric | Score |
|---|---|
| Avg relevance | 0.718 |
| Avg faithfulness | 0.842 |
| Self-heal rate (eval set) | 0% |
| Chunks indexed | 34,709 |
| Source judgments | 11,970 |
| Embedding cost | $0.00 (local model) |

## Tech stack

Python · FastAPI · sentence-transformers · ChromaDB · rank-bm25 · cross-encoder reranking · Groq (llama-3.1-8b-instant) · PostgreSQL · GitHub Actions · Docker

## CI/CD

Every push and pull request triggers `.github/workflows/rag-eval.yml`, which runs 10 curated legal Q&A pairs through the full pipeline and checks scores against thresholds (relevance ≥ 0.55, faithfulness ≥ 0.60). If either drops below threshold, the pipeline fails and blocks the merge.

## Data

[OpenNyAI](https://huggingface.co/opennyaiorg) datasets via HuggingFace — `InJudgements_dataset` (11,970 Indian High Court / Supreme Court judgments) and `InRhetoricalRoles` (26,133 human-labeled rhetorical role spans). Predominantly High Court civil matters — Land & Property, Constitutional, Tax, and Financial cases are most represented; Criminal and Industrial & Labour cases are comparatively fewer.

## Running locally

```bash
git clone https://github.com/<your-username>/indian-legal-rag.git
cd indian-legal-rag
pip install -r requirements.txt

# set environment variables
cp .env.example .env   # add your GROQ_API_KEY and HF_TOKEN

# run data pipeline (downloads, cleans, chunks, embeds — takes a few minutes)
python src/ingestion/data_loader.py
python src/ingestion/text_cleaner.py
python src/chunking/legal_chunker.py
python src/embeddings/embedder.py

# start the API
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive API explorer.

## Known limitations

- No authentication or rate limiting on the API
- BM25 index rebuilds from disk on every cold start rather than persisting
- Faithfulness check is rule-based (number/citation matching), not a full NLI model
- Self-healing is a deterministic retry, not an agentic decision loop
- Not legal advice — research and engineering demonstration only

## License

MIT