# One container: FastAPI serves both the JSON API and the web page. Sized for 512 MB free tiers
# (measured: ~360 MB after start-up). All data lives in Neon Postgres and Qdrant Cloud.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    BM25_BUILD_SUBPROCESS=1 \
    INDEX_STRATEGIES=sentence CHUNK_STRATEGY=sentence
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd -m -u 1000 app && mkdir -p /app/models && chown -R app /app

USER app
# Download the ONNX embedding model at build time so start-up does not depend on the network.
RUN python -c "from investigator.embeddings import FastEmbedEmbedder; FastEmbedEmbedder()"

EXPOSE 7860
# Render sets $PORT; elsewhere 7860 is used.
CMD ["sh", "-c", "uvicorn investigator.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
