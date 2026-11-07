from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..musique_handoff import build_cases, build_training_pools, eligible_dev_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a frozen MuSiQue split-hop A-to-B handoff benchmark"
    )
    parser.add_argument("--dev", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--samples",
        type=int,
        default=197,
        help="197 dev rows satisfy the strict eight-unique-answer v1 filter",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--min-a-document-chars", type=int, default=24_000)
    parser.add_argument("--case-split", default="dev")
    parser.add_argument("--decoy-source-split", default="train")
    parser.add_argument(
        "--exclude-case-ids-from",
        action="append",
        default=[],
        help="JSONL benchmark whose case IDs must not appear in the new split; repeatable.",
    )
    parser.add_argument(
        "--include-cases-from",
        action="append",
        default=[],
        help=(
            "Frozen JSONL cases to include verbatim before sampling the remaining "
            "requested cases; repeatable. Included IDs are excluded from resampling."
        ),
    )
    args = parser.parse_args()

    train_rows = _jsonl(Path(args.train))
    pools = build_training_pools(train_rows)
    dev_rows = list(_jsonl(Path(args.dev)))
    eligible = eligible_dev_rows(
        dev_rows, pools, candidate_count=args.candidate_count
    )
    explicit_excluded_ids = {
        row["id"]
        for excluded_path in args.exclude_case_ids_from
        for row in _jsonl(Path(excluded_path))
    }
    included_cases = [
        row
        for included_path in args.include_cases_from
        for row in _jsonl(Path(included_path))
    ]
    included_ids = [case["id"] for case in included_cases]
    if len(included_ids) != len(set(included_ids)):
        parser.error("--include-cases-from contains duplicate case IDs")
    if set(included_ids) & explicit_excluded_ids:
        parser.error("included and explicitly excluded case IDs overlap")
    if len(included_cases) > args.samples:
        parser.error("included cases exceed --samples")
    excluded_ids = explicit_excluded_ids | set(included_ids)
    eligible_after_exclusion = [
        row for row in eligible if row["id"] not in excluded_ids
    ]
    additional_count = args.samples - len(included_cases)
    additional_cases = (
        build_cases(
            dev_rows,
            pools,
            samples=additional_count,
            seed=args.seed,
            candidate_count=args.candidate_count,
            min_a_document_chars=args.min_a_document_chars,
            case_split=args.case_split,
            decoy_source_split=args.decoy_source_split,
            excluded_case_ids=excluded_ids,
        )
        if additional_count
        else []
    )
    cases = [*included_cases, *additional_cases]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for case in cases:
            sink.write(json.dumps(case, ensure_ascii=False) + "\n")

    relation_counts = Counter(case["relation_key"] for case in cases)
    summary = {
        "benchmark": "musique_split_handoff_v1",
        "num_cases": len(cases),
        "eligible_dev_cases": len(eligible),
        "eligible_after_exclusion": len(eligible_after_exclusion),
        "excluded_case_id_count": len(explicit_excluded_ids),
        "sampling_excluded_case_id_count": len(excluded_ids),
        "exclude_case_ids_from": args.exclude_case_ids_from,
        "included_case_count": len(included_cases),
        "include_cases_from": args.include_cases_from,
        "candidate_count": args.candidate_count,
        "chance_accuracy": 1.0 / args.candidate_count,
        "min_a_document_chars": min(
            case["agent_a"]["document_chars"] for case in cases
        ),
        "mean_a_document_chars": sum(
            case["agent_a"]["document_chars"] for case in cases
        )
        / len(cases),
        "relation_counts": dict(relation_counts.most_common()),
        "case_split": args.case_split,
        "decoy_source_split": args.decoy_source_split,
        "uses_train_only_for_decoy_documents": args.decoy_source_split == "train",
        "learned_parameters": 0,
        "case_ids_sha256": hashlib.sha256(
            "\n".join(case["id"] for case in cases).encode("utf-8")
        ).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
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
