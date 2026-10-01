"""LLM and embedding providers.

Uses OpenAI through LangChain when an API key is configured. Without a key the
project falls back to deterministic offline components so the whole pipeline
(and the test suite) runs on any machine with no network access.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from typing import Protocol

from .config import Settings

_WORD = re.compile(r"[a-zA-Z0-9]+")
STOPWORDS = frozenset(
    (
        "a an the is are was were be been of to in on for and or with what which who how why when "
        "does do did can i you we it this that these those my our your at by from as"
    ).split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in (t.lower() for t in _WORD.findall(text)) if w not in STOPWORDS and len(w) > 1]


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------------- online


class OpenAILLM:
    def __init__(self, settings: Settings):
        from langchain_openai import ChatOpenAI

        self._chat = ChatOpenAI(
            model=settings.chat_model, api_key=settings.openai_api_key, temperature=0, max_retries=3
        )

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        return str(self._chat.invoke([SystemMessage(system), HumanMessage(user)]).content)


class OpenAIEmbedder:
    def __init__(self, settings: Settings):
        from langchain_openai import OpenAIEmbeddings

        self._emb = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._emb.embed_documents(texts)


# --------------------------------------------------------------------------- offline


class HashingEmbedder:
    """Feature-hashing bag-of-words embedder (unigrams + bigrams), L2-normalised."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
        return h % self.dim, (1.0 if (h >> 64) & 1 else -1.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            toks = tokenize(text)
            for feat in toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]:
                i, sign = self._bucket(feat)
                vec[i] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class OfflineLLM:
    """Deterministic stand-in for a chat model.

    It understands the handful of prompt "tasks" this project issues (tagged with
    ``TASK:``) and answers them with simple heuristics, which keeps demos and CI
    reproducible. Swap in a real model by setting ``OPENAI_API_KEY``.
    """

    def complete(self, system: str, user: str) -> str:
        task = re.search(r"TASK:\s*(\w+)", system)
        task = task.group(1) if task else "answer"
        return getattr(self, f"_{task}", self._answer)(user)

    def _rewrite(self, user: str) -> str:
        q = user.split("QUESTION:", 1)[-1].strip()
        return " ".join(tokenize(q)) or q

    def _plan(self, user: str) -> str:
        q = user.split("QUESTION:", 1)[-1].strip()
        steps = [{"action": "retrieve", "input": q}]
        expr = re.search(r"(\d[\d\s.+\-*/()%]*[+\-*/%][\d\s.+\-*/()%]*\d)", q)
        if expr:
            steps.append({"action": "tool", "tool": "calculator", "input": expr.group(1).strip()})
        return json.dumps({"steps": steps})

    def _fix_tool_input(self, user: str) -> str:
        bad = user.split("INPUT:", 1)[-1].split("ERROR:", 1)[0]
        return re.sub(r"[^0-9.+\-*/()% ]", "", bad).strip()

    def _answer(self, user: str) -> str:
        q_tokens = set(tokenize(user.split("QUESTION:", 1)[-1]))
        context = user.split("CONTEXT:", 1)[-1].split("QUESTION:", 1)[0]
        scored = []
        for block in re.split(r"\n(?=\[\d+\])", context.strip()):
            m = re.match(r"\[(\d+)\]\s*(.*)", block, re.DOTALL)
            if not m:
                continue
            cid, body = m.groups()
            body = " ".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
            for sent in re.split(r"(?<=[.!?])\s+", body.strip()):
                overlap = len(q_tokens & set(tokenize(sent)))
                if overlap:
                    scored.append((overlap, sent.strip(), cid))
        tools = ""
        if "TOOL RESULTS:" in user:
            tools = user.split("TOOL RESULTS:", 1)[1].split("CONTEXT:", 1)[0].strip()
        tools = "" if tools == "none" else tools
        if not scored and not tools:
            return "I don't know based on the provided documents."
        best = sorted(scored, key=lambda s: -s[0])[:2]
        parts = [f"{s} [{c}]" for _, s, c in best]
        if tools:
            parts.append(f"Computed result: {tools}")
        return " ".join(parts)


def build_llm(settings: Settings) -> LLM:
    return OpenAILLM(settings) if settings.openai_api_key else OfflineLLM()


def build_embedder(settings: Settings) -> Embedder:
    return OpenAIEmbedder(settings) if settings.openai_api_key else HashingEmbedder()
