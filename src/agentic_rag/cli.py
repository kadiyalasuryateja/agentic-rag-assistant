"""Command-line interface: ingest documents and chat with the assistant."""

from __future__ import annotations

import argparse
import json

from .assistant import Assistant


def main() -> None:
    p = argparse.ArgumentParser(prog="agentic-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="index .md/.txt files from a directory")
    ing.add_argument("path")
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--trace", action="store_true", help="print the agent trace")
    args = p.parse_args()

    assistant = Assistant()
    if args.cmd == "ingest":
        print(f"indexed {assistant.kb.add_directory(args.path)} chunks "
              f"({assistant.kb.count()} total)")
        return

    res = assistant.ask(args.question)
    print(res.answer)
    for c in res.citations:
        print(f"  [{c['ref']}] {c['source']}")
    print(f"status={res.status} grounding={res.grounding} latency={res.latency_ms}ms")
    if res.status == "needs_review":
        print("review payload:", json.dumps(res.review, indent=2))
    if args.trace:
        print("\n".join(f"  - {t}" for t in res.trace))


if __name__ == "__main__":
    main()
