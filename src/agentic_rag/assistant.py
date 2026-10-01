"""High-level facade used by the API, CLI and evaluation scripts."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from langgraph.types import Command

from .config import Settings, get_settings
from .graph import build_graph
from .llm import LLM, Embedder, build_embedder, build_llm
from .retrieval import KnowledgeBase


@dataclass
class ChatResult:
    thread_id: str
    status: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    grounding: float | None = None
    review: dict | None = None
    trace: list[str] = field(default_factory=list)
    latency_ms: int = 0
    contexts: list[str] = field(default_factory=list)


class Assistant:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None,
                 embedder: Embedder | None = None, kb: KnowledgeBase | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or build_llm(self.settings)
        self.kb = kb or KnowledgeBase(self.settings, embedder or build_embedder(self.settings))
        self.graph = build_graph(self.settings, self.llm, self.kb)

    def _result(self, thread_id: str, out: dict, started: float, trace_from: int) -> ChatResult:
        interrupts = out.get("__interrupt__") or []
        return ChatResult(
            thread_id=thread_id,
            status="needs_review" if interrupts else out.get("status", "answered"),
            answer=out.get("answer", ""),
            citations=out.get("citations", []),
            grounding=out.get("grounding"),
            review=interrupts[0].value if interrupts else None,
            trace=out.get("trace", [])[trace_from:],
            latency_ms=int((time.perf_counter() - started) * 1000),
            contexts=[d["text"] for d in out.get("docs", [])],
        )

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}

    def ask(self, question: str, thread_id: str | None = None) -> ChatResult:
        thread_id = thread_id or uuid.uuid4().hex
        cfg = self._config(thread_id)
        prev = self.graph.get_state(cfg).values.get("trace", [])
        started = time.perf_counter()
        out = self.graph.invoke({"question": question}, cfg)
        return self._result(thread_id, out, started, len(prev))

    def resume(self, thread_id: str, action: str, answer: str | None = None) -> ChatResult:
        cfg = self._config(thread_id)
        state = self.graph.get_state(cfg)
        if not state.next:
            raise LookupError(f"thread {thread_id} is not waiting for review")
        started = time.perf_counter()
        out = self.graph.invoke(Command(resume={"action": action, "answer": answer}), cfg)
        return self._result(thread_id, out, started, len(state.values.get("trace", [])))
