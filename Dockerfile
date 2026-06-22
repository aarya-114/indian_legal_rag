FROM python:3.12-slim

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first — Docker caches this layer
# if requirements don't change, this layer is reused
# makes rebuilds much faster
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY data/chroma_db/ ./data/chroma_db/
COPY data/eval/ ./data/eval/

# create necessary init files
RUN touch src/__init__.py \
    src/api/__init__.py \
    src/retrieval/__init__.py \
    src/evaluation/__init__.py \
    src/ingestion/__init__.py \
    src/chunking/__init__.py \
    src/embeddings/__init__.py

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]