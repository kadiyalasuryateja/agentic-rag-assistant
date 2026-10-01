"""Document ingestion and hybrid retrieval (dense vectors + BM25, fused with RRF)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from .config import Settings
from .llm import Embedder, tokenize


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    score: float = 0.0
    meta: dict = field(default_factory=dict)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware chunking: pack paragraphs up to ``size`` chars, overlap on split."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
            continue
        if buf:
            chunks.append(buf)
        while len(p) > size:  # hard-split very long paragraphs
            chunks.append(p[:size])
            p = p[size - overlap :]
        buf = (chunks[-1][-overlap:] + "\n\n" + p) if chunks and overlap else p
    if buf:
        chunks.append(buf)
    return chunks


log = logging.getLogger(__name__)


class LocalVectorStore:
    """Dependency-free cosine-similarity store with the subset of the Chroma collection API we use.

    Used when ``chromadb`` is not installed (or ``RAG_VECTOR_STORE=local``); persists to a JSON file.
    Fine for thousands of chunks; use Chroma for anything larger.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.rows: dict[str, dict[str, Any]] = {}
        if self.path and self.path.exists():
            self.rows = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.rows), encoding="utf-8")

    def count(self) -> int:
        return len(self.rows)

    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        for i, d, e, m in zip(ids, documents, embeddings, metadatas, strict=True):
            self.rows[i] = {"document": d, "embedding": list(e), "metadata": m}
        self._save()

    def get(self, include=None) -> dict:
        return {
            "ids": list(self.rows),
            "documents": [r["document"] for r in self.rows.values()],
            "metadatas": [r["metadata"] for r in self.rows.values()],
        }

    def query(self, query_embeddings, n_results: int) -> dict:
        q = query_embeddings[0]
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        scored = []
        for i, r in self.rows.items():
            e = r["embedding"]
            en = math.sqrt(sum(x * x for x in e)) or 1.0
            sim = sum(a * b for a, b in zip(q, e, strict=False)) / (qn * en)
            scored.append((1 - sim, i, r))
        scored.sort(key=lambda t: t[0])
        top = scored[:n_results]
        return {
            "ids": [[i for _, i, _ in top]],
            "documents": [[r["document"] for _, _, r in top]],
            "metadatas": [[r["metadata"] for _, _, r in top]],
            "distances": [[d for d, _, _ in top]],
        }


def open_collection(settings: Settings, client: Any = None):
    """Return a Chroma collection when available, else the local JSON-backed store."""
    if settings.vector_store != "local":
        try:
            import chromadb

            client = client or chromadb.PersistentClient(path=settings.chroma_path)
            return client.get_or_create_collection(settings.collection, metadata={"hnsw:space": "cosine"})
        except ImportError:
            if settings.vector_store == "chroma":
                raise
            log.info("chromadb not installed; using the built-in local vector store")
    return LocalVectorStore(Path(settings.chroma_path) / f"{settings.collection}.json")


class KnowledgeBase:
    def __init__(self, settings: Settings, embedder: Embedder, client: Any = None):
        self.settings = settings
        self.embedder = embedder
        self.col = open_collection(settings, client)
        self._bm25: BM25Okapi | None = None
        self._corpus: list[Chunk] = []
        self._refresh_bm25()

    # ------------------------------------------------------------------ ingest
    def add_text(self, text: str, source: str) -> int:
        pieces = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not pieces:
            return 0
        ids = [hashlib.sha1(f"{source}:{i}:{p}".encode()).hexdigest()[:16] for i, p in enumerate(pieces)]
        self.col.upsert(
            ids=ids,
            documents=pieces,
            embeddings=self.embedder.embed(pieces),
            metadatas=[{"source": source, "chunk": i} for i in range(len(pieces))],
        )
        self._refresh_bm25()
        return len(pieces)

    def add_directory(self, path: str | Path) -> int:
        total = 0
        for f in sorted(Path(path).rglob("*")):
            if f.suffix.lower() in {".md", ".txt"}:
                total += self.add_text(f.read_text(encoding="utf-8"), source=f.name)
        return total

    def count(self) -> int:
        return self.col.count()

    def _refresh_bm25(self) -> None:
        data = self.col.get(include=["documents", "metadatas"])
        self._corpus = [
            Chunk(id=i, text=d, source=m.get("source", "?"))
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"], strict=True)
        ]
        self._bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in self._corpus]) if self._corpus else None

    # ------------------------------------------------------------------ search
    def dense_search(self, query: str, k: int) -> list[Chunk]:
        if not self.count():
            return []
        res = self.col.query(query_embeddings=self.embedder.embed([query]), n_results=min(k, self.count()))
        return [
            Chunk(id=i, text=d, source=m.get("source", "?"), score=1 - dist)
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
        ]

    def keyword_search(self, query: str, k: int) -> list[Chunk]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._corpus, scores, strict=True), key=lambda x: -x[1])[:k]
        return [Chunk(id=c.id, text=c.text, source=c.source, score=float(s)) for c, s in ranked if s > 0]

    def hybrid_search(self, query: str, k: int | None = None, rrf_k: int = 60) -> list[Chunk]:
        """Reciprocal Rank Fusion over dense and BM25 results, then lexical re-rank."""
        k = k or self.settings.top_k
        n = self.settings.candidate_k
        fused: dict[str, Chunk] = {}
        for results in (self.dense_search(query, n), self.keyword_search(query, n)):
            for rank, c in enumerate(results):
                entry = fused.setdefault(c.id, Chunk(id=c.id, text=c.text, source=c.source))
                entry.score += 1.0 / (rrf_k + rank + 1)
        return rerank(query, list(fused.values()))[:k]


def rerank(query: str, chunks: list[Chunk]) -> list[Chunk]:
    """Lightweight re-ranker: RRF score boosted by query-term coverage.

    Swap for a cross-encoder (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``) in production.
    """
    q = set(tokenize(query))
    for c in chunks:
        coverage = len(q & set(tokenize(c.text))) / (len(q) or 1)
        c.meta["coverage"] = round(coverage, 3)
        c.score = c.score * (1 + coverage)
    return sorted(chunks, key=lambda c: -c.score)
