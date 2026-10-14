# Indian Legal RAG

A portfolio research prototype for developers, recruiters and technical interviews: it finds passages in Indian court judgments and generates source-labeled answers using local hybrid retrieval, FastAPI and a static browser UI. It is not a legal-advice product and does not establish that an answer is legally correct.

## What it does

- Prepares judgment text from OpenNyAI Hugging Face datasets using scripts under **src/ingestion/**.
- Chunks cleaned judgments and stores local embeddings in persistent ChromaDB.
- Combines dense and BM25 search using Reciprocal Rank Fusion, then cross-encoder reranking.
- Applies rule-based query classification and metadata filters, with bounded retry and statutory-section fallback paths.
- Sends retrieved passages to a Groq-compatible chat-completions endpoint and returns source records, retry/timing metadata, and heuristic scores; richer bounded chunk provenance is opt-in server logging.
- Provides a static HTML/CSS/JavaScript frontend and optional PostgreSQL query telemetry.

See [PROJECT_DEEP_DIVE.md](PROJECT_DEEP_DIVE.md) for function-level details and interview preparation. AWS instructions are proposals, not deployed or validated.

## Architecture

PostgreSQL is optional telemetry only. Offline data/index preparation is separate from the live API.

~~~mermaid
flowchart LR
  subgraph Build["Offline preparation"]
    HF["Hugging Face datasets"] --> Loader["data_loader.py"]
    Loader --> Raw["data/raw"]
    Raw --> Cleaner["text_cleaner.py"]
    Cleaner --> Processed["clean judgments"]
    Processed --> Chunker["legal_chunker.py"]
    Chunker --> Chunks["chunks.json"]
    Chunks --> Embedder["embedder.py"]
    Embedder --> Chroma[("Persistent ChromaDB")]
  end
  subgraph Runtime["Query service"]
    Browser["Browser"] --> UI["Static frontend"]
    UI --> API["FastAPI"]
    API --> Guards["Validation · body cap · rate limit"]
    Guards --> RAG["rag_query"]
    RAG --> Classify["Keyword classification and filters"]
    Classify --> Dense["Dense search"]
    Classify --> Sparse["BM25 search"]
    Chroma --> Dense
    Chunks --> Sparse
    Dense --> RRF["RRF"]
    Sparse --> RRF
    RRF --> Rank["Cross-encoder reranker"]
    Rank --> Recovery["Bounded section fallback or<br/>low-relevance rewrite/retry"]
    Recovery --> LLM["Groq-compatible model"]
    LLM --> Check["Citation-ID check and<br/>rule-based faithfulness score"]
    Check --> API
    API -. optional .-> PG[("PostgreSQL query logs")]
  end
  subgraph Verify["Evaluation and CI"]
    Dataset["Golden questions"] --> Eval["Local evaluation runner"]
    Eval --> Gate["Score gate"]
    Eval -. in-process query .-> RAG
    CI["GitHub Actions"] --> Tests["Mocked tests"]
    CI --> Docker["Linux Docker build"]
  end
~~~

## Technology

| Technology | Role here |
| --- | --- |
| Python 3.11 | Data preparation, embeddings, retrieval, API, evaluation |
| FastAPI, Pydantic, Uvicorn | HTTP service and schema validation |
| Sentence Transformers | Local embeddings and cross-encoder reranking |
| ChromaDB | Persistent cosine-similarity vector index |
| rank-bm25 | Sparse ranking over chunk JSON loaded at startup |
| OpenAI Python client | Chat completions via configured Groq-compatible endpoint |
| Hugging Face datasets | Offline source dataset downloads |
| HTML/CSS/JavaScript | Static frontend; no build framework |
| PostgreSQL/psycopg2 | Optional query logs and stats only |
| Docker/GitHub Actions | Image build check and lightweight tests |

## Repository map

- **src/ingestion/** — dataset loading, metadata extraction and cleaning.
- **src/chunking/legal_chunker.py** — chunks and optional debug limit.
- **src/embeddings/embedder.py** — local vector generation and Chroma writes.
- **src/retrieval/hybrid_retriever.py** — dense/BM25 retrieval, RRF, reranking and reference fallback.
- **src/api/** — FastAPI service and RAG orchestration.
- **src/evaluation/** — benchmark runner and score gate.
- **frontend/** — static interface, config and Node tests.
- **tests/** — Python unittest modules, fixtures and mocks.
- **data/eval/eval_dataset.json** — tracked 10-question evaluation set.
- **Dockerfile**, **.dockerignore**, **.github/workflows/rag-eval.yml** — image, context exclusions and CI.

Generated corpus, processed data and Chroma index are local artifacts, not in the Docker image.

## Local setup

Use Python 3.11. In Windows PowerShell:

~~~powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

The runtime requirements mark **uvloop** for non-Windows platforms, so Windows skips that package while installing the remaining dependencies. **requirements-ci.txt** is intentionally a lightweight test set; it is not a substitute for runtime requirements when serving the real retriever.

Set **GROQ_API_KEY** and **LLM_MODEL** before model requests. Do not commit credentials or put them in browser configuration. **HF_TOKEN** is only needed if Hugging Face requires authentication.

| Setting | Use |
| --- | --- |
| **GROQ_API_KEY** | Backend-only credential for answer and rewrite calls. |
| **LLM_MODEL** | Required provider model ID; no hardcoded Python fallback. |
| **LLM_REQUEST_TIMEOUT_SECONDS** | Optional provider timeout; default 60. |
| **LLM_MAX_OUTPUT_TOKENS** / **LLM_MAX_REWRITE_TOKENS** | Optional caps; defaults 4096 / 2048. |
| **DATABASE_URL** | Optional PostgreSQL query telemetry; leave unset without a DB. |
| **HF_TOKEN** | Optional Hugging Face authentication for data downloads. |
| **QUERY_RATE_LIMIT_REQUESTS** / **QUERY_RATE_LIMIT_WINDOW_SECONDS** | Optional per-process limit; defaults 30 / 60 seconds. |
| **RAG_DIAGNOSTICS** | Optional bounded server-side retrieval/timing logs; default false. |

See the deep-dive configuration table for exact readers/default behavior. Never put credentials in **frontend/config.js**.

### Build data and index

These commands process the full available train splits by default and may consume substantial time, disk, and memory:

~~~powershell
python src/ingestion/data_loader.py
python src/ingestion/text_cleaner.py
python src/chunking/legal_chunker.py
python src/embeddings/embedder.py
~~~

Use **python src/chunking/legal_chunker.py --limit N** for a limited debug chunking run. Runtime requires matching **data/processed/chunks.json** and **data/chroma_db/**. These generated files are not committed or copied into Docker.

### Start API and frontend

From the repo root:

~~~powershell
$env:PYTHONPATH = "."
python -m uvicorn src.api.main:app --reload --port 8000
~~~

FastAPI startup loads models, the local Chroma collection and chunk JSON. Visit **http://127.0.0.1:8000/docs**.

In a second terminal:

~~~powershell
python -m http.server 5173 --bind 127.0.0.1 --directory frontend
~~~

Visit **http://127.0.0.1:5173**. **frontend/config.js** contains the public API base URL, defaulting to **http://127.0.0.1:8000**. Current backend allows all CORS origins; browser code contains no provider key.

## API

- **GET /health** reports status, retriever-loaded state and version; it does not ping provider/database.
- **POST /query** accepts query, optional top-k and optional case-type filter; returns answer, judgment-level sources, citation-ID validation, scores, metadata and disclaimer.
- **GET /stats** returns an availability message or PostgreSQL aggregates from a manually provisioned **query_logs** table.

Query strings must be nonblank and 10–4,000 characters. Body cap is 65,536 bytes. The configurable in-memory rate limit defaults to 30 requests per client per 60 seconds. See the deep dive for models and errors.

## Retrieval and answer interpretation

Dense and sparse retrieval each return up to 20 candidates before fusion; the reranker normally selects five chunks. Relevance is a normalized reranker score, not a probability of correctness. Faithfulness is a narrow numeric/legal-reference substring heuristic, not semantic entailment or legal verification. Citation validation checks that answer source IDs exist in this response; it does not prove the cited judgment supports a claim.

A missing explicit section reference can trigger one broader-filter fallback when absent from selected evidence (top five by default). Otherwise relevance below 0.55 can trigger one guarded query rewrite/retrieval retry. These branches share one recovery opportunity.

## Evaluation

The live benchmark uses the tracked 10 questions, local index and configured provider; it may incur usage:

~~~powershell
python src/evaluation/eval_runner.py
python src/evaluation/score_gate.py data/eval/eval_results.json
~~~

The current worktree artifact **data/eval/eval_results.json** records relevance **0.805**, faithfulness **0.749**, answer overlap **0.149**, latency **12,830.08 ms**, **2/2** successful recoveries/eligible cases, and overall pass **true**. This file is modified in the worktree. These are values read from the saved file, not a rerun for this documentation task, and differ from other values previously quoted. A small benchmark pass is not legal correctness or production readiness.

## Tests and CI

~~~powershell
python -m pip install -r requirements-ci.txt
python -m unittest tests.test_phase1 -v
python -m unittest tests.test_self_healing_telemetry -v
python -m unittest tests.test_evaluation_audit tests.test_text_cleaner_encoding -v
node --test frontend/app.test.js
node --check frontend/app.js
~~~

GitHub Actions runs on push, pull request and manual dispatch. It installs **requirements-ci.txt**, runs four Python unittest modules and separately builds Docker on Linux. Tests use mocks/lightweight paths; CI does not ingest data, build the Chroma index, run live evaluation or push/deploy. The workflow was inspected but not run for this documentation change.

## Docker

The Dockerfile uses **python:3.11.9-slim**, installs runtime requirements, copies **src/** and starts Uvicorn on container port 8000. It excludes **data/**, **frontend/** and **.env**. Image build does not supply runtime corpus/index.

~~~powershell
docker build -t indian-legal-rag .
docker run --rm -p 8000:8000 -v "$(Resolve-Path data):/app/data" --env-file .env indian-legal-rag
~~~

The mount must include matching processed chunks and Chroma files. This documentation task did not build or run the container.

## AWS plan — proposed, not deployed or validated

The existing proposal uses one Linux EC2 instance, Docker, host Nginx and persistent EBS for chunk JSON and Chroma. Nginx would serve the static frontend and proxy the API. Provider credentials stay backend-side, optionally through a managed secret store. There is no AWS infrastructure code, Nginx/TLS config, registry integration or deploy workflow.

EC2/EBS fits a single local-file index with fewer components than ECS/Fargate, but requires host maintenance and has a single failure domain. AWS summarizes the operational trade-off in its [EC2/Fargate comparison](https://aws.amazon.com/compare/ec2-and-fargate/). Fargate's [task storage](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-task-storage.html) is not automatically a persistent Chroma volume. S3 can stage/backup artifacts but is not a drop-in Chroma SQLite store. A separate frontend origin needs production API URL and CORS review. Account-specific steps remain unvalidated. See the [AWS Pricing Calculator](https://calculator.aws/) before provisioning.

For the frontend, serving **frontend/** from Nginx on the same EC2 host is the fewest-service option and can keep browser/API traffic on one origin. A managed static host such as Amplify Hosting reduces static-server work; S3 website hosting with CloudFront is another option but adds services and needs API-origin/CORS configuration. An S3 website endpoint by itself is HTTP-only; see AWS guidance for [S3 website hosting](https://docs.aws.amazon.com/AmazonS3/latest/userguide/WebsiteHosting.html) and [secure CloudFront/S3 hosting](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/getting-started-secure-static-website-cloudformation-template.html). None is configured in this repository.

Indicative sequence to validate in your own AWS account:

1. Measure the existing chunk JSON, Chroma directory, model cache and retriever peak memory before selecting EC2/EBS size. Do not build the corpus on the instance as part of deployment.
2. Provision one Linux EC2 host and persistent EBS volume. Restrict ingress and administrative access; do not expose Uvicorn directly.
3. Transfer the prepared **data/processed/chunks.json** and **data/chroma_db/** to the mounted volume. Keep backups separate.
4. Check out the application and build the existing Dockerfile. GitHub Actions builds but does not publish an image.
5. Provide **GROQ_API_KEY** and **LLM_MODEL** through a protected host environment file or separately configured secret store. Mount data at **/app/data** and bind Uvicorn to loopback behind Nginx.
6. Configure Nginx/TLS to serve **frontend/** and proxy **/query**, **/health**, and **/stats**. Set the deployed API origin in **frontend/config.js**. Restrict wildcard CORS before public exposure.
7. Check logs and **/health**, then send one controlled query to verify mounted data, model availability, outbound provider access and browser routing. This repository has not executed or validated these AWS steps.

For this proposal EC2, EBS and network/security configuration are needed. Systems Manager Parameter Store SecureString or Secrets Manager can hold backend credentials, but integration is deployment work and is not in this codebase. S3 artifact staging/backup, centralized logging, a separate static host, ECR, load balancer, database, DNS and CDN are optional or unnecessary for a first single-host deployment. PostgreSQL is not required by query generation. This is not a high-availability design.

## Optional PostgreSQL query statistics

The application does not create its table. If PostgreSQL logging is desired, provision a compatible table before setting **DATABASE_URL**:

~~~sql
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
~~~

## Security and limitations

- No authentication. In-memory rate limiting is per process, resets at restart and is not distributed abuse protection.
- CORS is wildcard; production needs an explicit origin/network policy.
- Corpus/index are local, ignored by Git and excluded from Docker.
- Runtime needs local embedding/reranker models and remote Groq-compatible generation.
- Faithfulness and citation-ID checks do not establish legal correctness or evidence support.
- Rhetorical spans are prepared offline but unused in runtime retrieval. Citation-graph retrieval, NLI verification, agents and deployment automation are not implemented.
- PostgreSQL is optional telemetry only. Logging errors do not discard successful answers; **/stats** needs an accessible database and manually created table.

See [PROJECT_DEEP_DIVE.md](PROJECT_DEEP_DIVE.md) for schemas, exact flow, metrics, configuration and interview preparation.

## Troubleshooting

- **API startup cannot load retriever:** confirm **data/processed/chunks.json**, **data/chroma_db/** and collection **indian_legal_judgments** are present and from the same build; local model files may also need to be downloaded.
- **Model request fails:** verify backend-only **GROQ_API_KEY**, provider-supported **LLM_MODEL**, outbound network access and configured timeout. A missing model setting is rejected at import.
- **Docker starts but cannot retrieve:** mount the matching data directory at **/app/data**; the Docker build intentionally contains no corpus/index.
- **Browser cannot reach API:** verify **frontend/config.js**, API origin, backend availability and CORS. Current CORS is wildcard, but deployed proxy/origin settings still need checking.
- **PostgreSQL stats unavailable:** leave DATABASE_URL unset if unused; otherwise confirm the database is reachable and the query_logs table was created. Database failure is optional for query generation.
