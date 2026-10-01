"""Input and output guardrails."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import tokenize

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts)",
    r"disregard (the )?(system|previous) prompt",
    r"you are now (?:dan|in developer mode)",
    r"reveal (your|the) (system prompt|instructions)",
]
PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
}


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    sanitized: str = ""


def check_input(text: str, max_len: int = 4000) -> GuardResult:
    if not text.strip():
        return GuardResult(False, "empty question")
    if len(text) > max_len:
        return GuardResult(False, f"question longer than {max_len} characters")
    lowered = text.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            return GuardResult(False, "possible prompt injection")
    sanitized = text
    for name, pat in PII_PATTERNS.items():
        sanitized = re.sub(pat, f"[REDACTED_{name.upper()}]", sanitized)
    return GuardResult(True, sanitized=sanitized)


def grounding_score(answer: str, contexts: list[str]) -> float:
    """Fraction of answer content-words that appear in the retrieved context.

    A cheap, deterministic faithfulness proxy used at runtime; the offline eval
    suite uses Ragas faithfulness for a model-graded score.
    """
    ans = set(tokenize(re.sub(r"\[\d+\]", "", answer)))
    if not ans:
        return 0.0
    ctx = set(tokenize(" ".join(contexts)))
    return round(len(ans & ctx) / len(ans), 3)


def is_refusal(answer: str) -> bool:
    return "i don't know" in answer.lower()
