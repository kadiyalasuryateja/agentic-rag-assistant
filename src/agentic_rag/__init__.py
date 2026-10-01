"""Agentic RAG Assistant: a LangGraph multi-agent RAG system with guardrails and human review."""

from .assistant import Assistant, ChatResult

__all__ = ["Assistant", "ChatResult"]
__version__ = "1.0.0"
