from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def summarize(rows: list[dict], *, expected_documents: int | None = None) -> dict[str, float | int]:
    by_index: dict[int, dict] = {}
    for row in rows:
        index = int(row["index"])
        if index in by_index:
            raise ValueError(f"duplicate document index: {index}")
        by_index[index] = row

    if expected_documents is not None:
        expected = set(range(expected_documents))
        actual = set(by_index)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise ValueError(
                f"index coverage mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

    count = len(by_index)
    if count == 0:
        raise ValueError("no evaluation rows found")
    standalone_correct = sum(
        row["standalone_prediction"] == row["label"] for row in by_index.values()
    )
    transfer_correct = sum(row["transfer_prediction"] == row["label"] for row in by_index.values())
    standalone = standalone_correct / count
    transfer = transfer_correct / count
    chance = 0.25
    return {
        "documents": count,
        "standalone_correct": standalone_correct,
        "transfer_correct": transfer_correct,
        "standalone_acc_norm": standalone,
        "transfer_acc_norm": transfer,
        "accuracy_delta_pp": 100.0 * (transfer - standalone),
        "retention_percent": 100.0 * transfer / standalone,
        "floor_normalized_retention_percent": 100.0
        * (transfer - chance)
        / (standalone - chance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and validate sharded HellaSwag JSONL results")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--expected-documents", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = []
    for input_name in args.inputs:
        with Path(input_name).open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    result = summarize(rows, expected_documents=args.expected_documents)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(rendered)
        os.replace(temporary, destination)
    print(rendered, end="")


if __name__ == "__main__":
    main()
