from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..musique_chain_handoff import (
    build_chain_cases,
    collect_candidate_records,
    collect_distractor_documents,
    is_linear_chain,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen 3/4-agent MuSiQue chains")
    parser.add_argument("--dev", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hops", type=int, choices=[3, 4], required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--min-a-document-chars", type=int, default=24_000)
    args = parser.parse_args()

    train_rows = list(_jsonl(Path(args.train)))
    dev_rows = list(_jsonl(Path(args.dev)))
    candidates = collect_candidate_records(train_rows)
    distractors = collect_distractor_documents(train_rows)
    eligible = sum(is_linear_chain(row, hops=args.hops) for row in dev_rows)
    cases = build_chain_cases(
        dev_rows,
        candidates,
        distractors,
        hops=args.hops,
        samples=args.samples,
        seed=args.seed,
        candidate_count=args.candidate_count,
        min_agent_a_document_chars=args.min_a_document_chars,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for case in cases:
            sink.write(json.dumps(case, ensure_ascii=False) + "\n")
    summary = {
        "benchmark": "musique_linear_chain_handoff_v1",
        "agent_count": args.hops,
        "num_cases": len(cases),
        "eligible_dev_cases": eligible,
        "candidate_records": len(candidates),
        "candidate_count_per_stage": args.candidate_count,
        "chance_accuracy_per_stage": 1.0 / args.candidate_count,
        "min_agent_a_document_chars": min(case["agent_a"]["document_chars"] for case in cases),
        "mean_agent_a_document_chars": sum(case["agent_a"]["document_chars"] for case in cases) / len(cases),
        "uses_train_only_for_decoy_records": True,
        "learned_parameters": 0,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
