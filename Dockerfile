FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data
RUN useradd --create-home app && chown -R app /app
USER app

ENV RAG_CHROMA_PATH=/app/.chroma RAG_SEED_DIR=/app/data/docs
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').getenv('PORT', '8000') + '/health')"
CMD ["sh", "-c", "uvicorn agentic_rag.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
