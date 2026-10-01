"""FastAPI service exposing the multi-agent assistant."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .assistant import Assistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agentic_rag.api")
state: dict[str, Assistant] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    assistant = Assistant()
    seed = os.getenv("RAG_SEED_DIR", "data/docs")
    if assistant.kb.count() == 0 and os.path.isdir(seed):
        n = assistant.kb.add_directory(seed)
        log.info("seeded knowledge base with %d chunks from %s", n, seed)
    state["assistant"] = assistant
    yield


app = FastAPI(title="Agentic RAG Assistant", version="1.0.0", lifespan=lifespan)


class IngestRequest(BaseModel):
    source: str = Field(..., examples=["handbook.md"])
    text: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    action: Literal["approve", "edit", "reject"]
    answer: str | None = None


def _assistant() -> Assistant:
    return state["assistant"]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": _assistant().kb.count()}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    n = await run_in_threadpool(_assistant().kb.add_text, req.text, req.source)
    return {"source": req.source, "chunks_added": n}


@app.post("/chat")
async def chat(req: ChatRequest):
    res = await run_in_threadpool(_assistant().ask, req.question, req.thread_id)
    out = asdict(res)
    out.pop("contexts")
    return out


@app.post("/threads/{thread_id}/resume")
async def resume(thread_id: str, req: ResumeRequest):
    try:
        res = await run_in_threadpool(_assistant().resume, thread_id, req.action, req.answer)
    except LookupError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    out = asdict(res)
    out.pop("contexts")
    return out
