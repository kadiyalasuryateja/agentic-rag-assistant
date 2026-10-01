"""LangGraph multi-agent workflow.

    guard_input ─▶ planner ─▶ supervisor ─┬─▶ retriever ─┐
                                          ├─▶ executor  ─┤ (loop back to supervisor)
                                          └─▶ writer ─▶ reviewer ─┬─▶ END
                                                                  ├─▶ supervisor (retry, wider search)
                                                                  └─▶ human_review (interrupt) ─▶ END

* The **supervisor** walks the plan and routes each step to the right agent.
* The **reviewer** scores grounding; low scores trigger a retry, then a
  human-in-the-loop checkpoint that pauses the thread until someone resumes it.
* State is checkpointed per ``thread_id`` so paused runs resume without losing work.
"""

from __future__ import annotations

import json
import logging
import operator
import re
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .guardrails import check_input, grounding_score, is_refusal
from .llm import LLM
from .retrieval import KnowledgeBase
from .tools import TOOLS, ToolError, run_tool

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- schemas
class Step(BaseModel):
    action: Literal["retrieve", "tool"]
    input: str = Field(..., min_length=1)
    tool: str | None = None


class Plan(BaseModel):
    steps: list[Step] = Field(..., min_length=1, max_length=5)


class AgentState(TypedDict, total=False):
    question: str
    blocked_reason: str
    plan: list[dict]
    cursor: int
    attempt: int
    docs: list[dict]
    tool_results: list[str]
    answer: str
    citations: list[dict]
    grounding: float
    status: Literal["running", "answered", "blocked", "needs_review", "approved", "edited", "rejected"]
    trace: Annotated[list[str], operator.add]


# ----------------------------------------------------------------------------- prompts
PLANNER_SYS = (
    "TASK: plan\nYou are the planner agent. Break the user question into at most 5 steps. "
    "Use action 'retrieve' to search the knowledge base and action 'tool' for exact computation. "
    f"Available tools: {', '.join(f'{t.name} ({t.description})' for t in TOOLS.values())}. "
    'Reply with JSON only: {"steps": [{"action": "retrieve", "input": "..."}, '
    '{"action": "tool", "tool": "calculator", "input": "2*3"}]}'
)
REWRITE_SYS = (
    "TASK: rewrite\nRewrite the question into a concise search query with the key terms. "
    "Reply with the query only."
)
WRITER_SYS = (
    "TASK: answer\nYou are a careful assistant. Answer ONLY from the numbered context and tool "
    "results. Cite sources inline like [1]. If the context does not contain the answer, reply "
    "exactly: I don't know based on the provided documents."
)
FIX_SYS = "TASK: fix_tool_input\nThe tool call failed. Return a corrected input only."


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in planner output")
    return json.loads(m.group(0))


# ----------------------------------------------------------------------------- graph
def build_graph(settings: Settings, llm: LLM, kb: KnowledgeBase, checkpointer=None):
    def guard_input(state: AgentState) -> AgentState:
        res = check_input(state["question"])
        if not res.allowed:
            return {
                "status": "blocked",
                "blocked_reason": res.reason,
                "answer": f"Request blocked by guardrails: {res.reason}.",
                "trace": [f"guard_input: blocked ({res.reason})"],
            }
        return {"question": res.sanitized, "attempt": 1, "status": "running", "trace": ["guard_input: ok"]}

    def planner(state: AgentState) -> AgentState:
        q = state["question"]
        try:
            plan = Plan.model_validate(_extract_json(llm.complete(PLANNER_SYS, f"QUESTION: {q}")))
            steps = [s.model_dump() for s in plan.steps if s.action == "retrieve" or s.tool in TOOLS]
            note = f"planner: {len(steps)} step(s)"
        except (ValueError, ValidationError) as e:  # self-correct: fall back to a safe plan
            steps, note = [], f"planner: invalid plan ({type(e).__name__}), using fallback"
        if not any(s["action"] == "retrieve" for s in steps):
            steps.insert(0, {"action": "retrieve", "input": q, "tool": None})
        return {"plan": steps, "cursor": 0, "docs": [], "tool_results": [], "trace": [note]}

    def supervisor(state: AgentState) -> AgentState:
        return {"trace": [f"supervisor: step {state['cursor'] + 1}/{len(state['plan'])}"]
                if state["cursor"] < len(state["plan"]) else ["supervisor: plan complete -> writer"]}

    def route_supervisor(state: AgentState) -> str:
        if state["cursor"] >= len(state["plan"]):
            return "writer"
        return "retriever" if state["plan"][state["cursor"]]["action"] == "retrieve" else "executor"

    def retriever(state: AgentState) -> AgentState:
        step = state["plan"][state["cursor"]]
        query = llm.complete(REWRITE_SYS, f"QUESTION: {step['input']}").strip() or step["input"]
        k = settings.top_k * state.get("attempt", 1)  # widen the search on retries
        hits = kb.hybrid_search(query, k=k)
        seen = {d["id"] for d in state.get("docs", [])}
        new = [{"id": h.id, "text": h.text, "source": h.source, "score": round(h.score, 4)}
               for h in hits if h.id not in seen]
        return {"docs": state.get("docs", []) + new, "cursor": state["cursor"] + 1,
                "trace": [f"retriever: '{query}' -> {len(new)} chunk(s)"]}

    def executor(state: AgentState) -> AgentState:
        step = state["plan"][state["cursor"]]
        raw, result = step["input"], None
        for attempt in range(settings.max_tool_retries + 1):
            try:
                result = run_tool(step["tool"], raw)
                break
            except ToolError as e:
                log.warning("tool error: %s", e)
                if attempt == settings.max_tool_retries:
                    result = f"{step['tool']} unavailable ({e})"
                else:
                    raw = llm.complete(FIX_SYS, f"TOOL: {step['tool']}\nINPUT: {raw}\nERROR: {e}").strip()
        return {"tool_results": state.get("tool_results", []) + [result],
                "cursor": state["cursor"] + 1, "trace": [f"executor: {result}"]}

    def writer(state: AgentState) -> AgentState:
        docs = state.get("docs", [])
        context = "\n".join(f"[{i}] {d['text']}" for i, d in enumerate(docs, 1)) or "(no documents)"
        tools = "; ".join(state.get("tool_results", [])) or "none"
        prompt = f"TOOL RESULTS: {tools}\n\nCONTEXT:\n{context}\n\nQUESTION: {state['question']}"
        answer = llm.complete(WRITER_SYS, prompt)
        used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        citations = [{"ref": i, "source": d["source"], "chunk_id": d["id"]}
                     for i, d in enumerate(docs, 1) if i in used]
        return {"answer": answer.strip(), "citations": citations, "trace": ["writer: drafted answer"]}

    def reviewer(state: AgentState) -> AgentState:
        contexts = [d["text"] for d in state.get("docs", [])] + state.get("tool_results", [])
        score = 0.0 if is_refusal(state["answer"]) else grounding_score(state["answer"], contexts)
        return {"grounding": score, "trace": [f"reviewer: grounding={score}"]}

    def route_reviewer(state: AgentState) -> str:
        if state["grounding"] >= settings.min_grounding_score:
            return "accept"
        if state.get("attempt", 1) < settings.max_retrieval_attempts:
            return "retry"
        return "human"

    def accept(state: AgentState) -> AgentState:
        return {"status": "answered", "trace": ["done"]}

    def retry(state: AgentState) -> AgentState:
        return {"attempt": state.get("attempt", 1) + 1, "cursor": 0, "docs": [], "tool_results": [],
                "trace": ["reviewer: low grounding, retrying with wider search"]}

    def human_review(state: AgentState) -> AgentState:
        decision = interrupt({
            "reason": "Answer could not be verified against the knowledge base.",
            "draft_answer": state["answer"],
            "grounding": state.get("grounding"),
            "options": ["approve", "edit", "reject"],
        })
        action = (decision or {}).get("action", "reject")
        if action == "approve":
            return {"status": "approved", "trace": ["human_review: approved"]}
        if action == "edit" and decision.get("answer"):
            return {"status": "edited", "answer": decision["answer"], "trace": ["human_review: edited"]}
        return {"status": "rejected", "answer": "A reviewer declined to answer this question.",
                "citations": [], "trace": ["human_review: rejected"]}

    g = StateGraph(AgentState)
    for name, fn in [("guard_input", guard_input), ("planner", planner), ("supervisor", supervisor),
                     ("retriever", retriever), ("executor", executor), ("writer", writer),
                     ("reviewer", reviewer), ("accept", accept), ("retry", retry),
                     ("human_review", human_review)]:
        g.add_node(name, fn)

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", lambda s: END if s.get("status") == "blocked" else "planner")
    g.add_edge("planner", "supervisor")
    g.add_conditional_edges("supervisor", route_supervisor, ["retriever", "executor", "writer"])
    g.add_edge("retriever", "supervisor")
    g.add_edge("executor", "supervisor")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges("reviewer", route_reviewer, {"accept": "accept", "retry": "retry", "human": "human_review"})
    g.add_edge("retry", "supervisor")
    g.add_edge("accept", END)
    g.add_edge("human_review", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
