"""Evaluate the assistant on a golden set.

Always reports retrieval hit rate, citation accuracy, grounding (faithfulness proxy),
answer rate, and latency. With ``--ragas`` (requires ``pip install .[eval]`` and an
OpenAI key) it also computes Ragas faithfulness, answer relevancy and context precision.

    python eval/run_eval.py --docs data/docs [--ragas] [--out eval/report.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from agentic_rag import Assistant
from agentic_rag.config import Settings, get_settings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default=str(Path(__file__).with_name("golden_set.jsonl")))
    p.add_argument("--docs", default="data/docs")
    p.add_argument("--ragas", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    base = get_settings()
    settings = Settings(**{**base.model_dump(), "chroma_path": tempfile.mkdtemp(), "collection": "eval"})
    assistant = Assistant(settings=settings)
    assistant.kb.add_directory(args.docs)

    rows = [json.loads(line) for line in Path(args.golden).read_text().splitlines() if line.strip()]
    records = []
    for row in rows:
        res = assistant.ask(row["question"])
        sources = [c["source"] for c in res.citations]
        records.append({
            **row,
            "answer": res.answer,
            "status": res.status,
            "contexts": res.contexts,
            "hit": row["source"] in {h.source for h in assistant.kb.hybrid_search(row["question"])},
            "cited_correct": bool(sources) and sources[0] == row["source"],
            "grounding": res.grounding or 0.0,
            "latency_ms": res.latency_ms,
        })

    n = len(records)
    report = {
        "n": n,
        "answer_rate": sum(r["status"] == "answered" for r in records) / n,
        "retrieval_hit_rate": sum(r["hit"] for r in records) / n,
        "top_citation_accuracy": sum(r["cited_correct"] for r in records) / n,
        "mean_grounding": round(statistics.mean(r["grounding"] for r in records), 3),
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in records),
    }

    if args.ragas:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        ds = Dataset.from_list([{"question": r["question"], "answer": r["answer"], "contexts": r["contexts"],
                                 "ground_truth": r["ground_truth"]} for r in records])
        scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
        report["ragas"] = {k: round(float(v), 3) for k, v in scores._repr_dict.items()}

    print(json.dumps(report, indent=2))
    for r in records:
        mark = "OK " if r["cited_correct"] else "MISS"
        print(f"[{mark}] {r['question']}\n       -> {r['answer'][:140]}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": report, "records": records}, indent=2))


if __name__ == "__main__":
    main()
