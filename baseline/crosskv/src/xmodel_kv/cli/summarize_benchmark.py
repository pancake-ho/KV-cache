from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


CHANCE = {"arc_challenge": 0.25, "winogrande": 0.5, "mmlu": 0.25, "gsm8k": 0.0}


def summarize(task: str, rows: list[dict], expected_documents: int | None = None) -> dict:
    by_index = {}
    for row in rows:
        index = int(row["index"])
        if index in by_index:
            raise ValueError(f"duplicate document index: {index}")
        by_index[index] = row
    if not by_index:
        raise ValueError("no evaluation rows found")
    if expected_documents is not None and set(by_index) != set(range(expected_documents)):
        missing = sorted(set(range(expected_documents)) - set(by_index))
        unexpected = sorted(set(by_index) - set(range(expected_documents)))
        raise ValueError(f"index coverage mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")

    rows = list(by_index.values())
    if task == "wikitext2":
        tokens = sum(int(row["tokens"]) for row in rows)
        standalone_ll = sum(float(row["standalone_log_likelihood"]) for row in rows)
        transfer_ll = sum(float(row["transfer_log_likelihood"]) for row in rows)
        return {
            "task": task,
            "chunks": len(rows),
            "scored_tokens": tokens,
            "standalone_nll": -standalone_ll / tokens,
            "transfer_nll": -transfer_ll / tokens,
            "standalone_ppl": math.exp(-standalone_ll / tokens),
            "transfer_ppl": math.exp(-transfer_ll / tokens),
            "ppl_ratio_percent": 100.0 * math.exp(-transfer_ll / tokens) / math.exp(-standalone_ll / tokens),
        }

    if task == "gsm8k":
        result = {"task": task, "documents": len(rows)}
        for filter_name in ("strict", "flexible"):
            standalone = sum(bool(row[f"standalone_{filter_name}_correct"]) for row in rows) / len(rows)
            transfer = sum(bool(row[f"transfer_{filter_name}_correct"]) for row in rows) / len(rows)
            result[f"standalone_{filter_name}_acc"] = standalone
            result[f"transfer_{filter_name}_acc"] = transfer
            result[f"{filter_name}_retention_percent"] = 100.0 * transfer / standalone if standalone else float("nan")
        return result

    standalone = sum(row["standalone_prediction"] == row["label"] for row in rows) / len(rows)
    transfer = sum(row["transfer_prediction"] == row["label"] for row in rows) / len(rows)
    chance = CHANCE[task]
    metric = "acc_norm" if task == "arc_challenge" else "acc"
    return {
        "task": task,
        "documents": len(rows),
        f"standalone_{metric}": standalone,
        f"transfer_{metric}": transfer,
        "accuracy_delta_pp": 100.0 * (transfer - standalone),
        "retention_percent": 100.0 * transfer / standalone,
        "floor_normalized_retention_percent": 100.0 * (transfer - chance) / (standalone - chance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and validate generic benchmark shards")
    parser.add_argument("--task", choices=tuple(CHANCE) + ("wikitext2",), required=True)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument(
        "--replacement-inputs",
        nargs="+",
        help="JSONL rows that replace matching indices from the positional inputs",
    )
    parser.add_argument("--expected-documents", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = []
    for name in args.inputs:
        with Path(name).open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if args.replacement_inputs:
        by_index = {int(row["index"]): row for row in rows}
        for name in args.replacement_inputs:
            with Path(name).open() as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        by_index[int(row["index"])] = row
        rows = list(by_index.values())
    result = summarize(args.task, rows, args.expected_documents)
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
