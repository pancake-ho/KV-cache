from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .summarize_policy_triad import _exact_mcnemar, _rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict summary for tail-policy JSONL")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [row for path in args.inputs for row in _jsonl(Path(path))]
    identifiers = [row["tail_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit("duplicate tail ids detected")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        triad = row["triad"]
        groups[(triad["family"], triad["direction"], triad["layout"])].append(row)
    summary = {
        "rows": len(rows),
        "unique_rows": len(set(identifiers)),
        "cells": [_summarize(key, group) for key, group in sorted(groups.items())],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(summary["cells"], ensure_ascii=False, indent=2))
    print({"rows": len(rows), "output": str(output_path)})


def _summarize(key, rows):
    family, direction, layout = key
    eligible = [
        row
        for row in rows
        if row["validity"]["front_target_correct"]
        and row["validity"]["front_source_correct"]
        and row["validity"]["tail_native_correct"]
    ]
    result = {
        "family": family,
        "direction": direction,
        "layout": layout,
        "rows": len(rows),
        "eligible": len(eligible),
        "eligible_rate": _rate(len(eligible), len(rows)),
        "prefix_shift_min": min(
            row["token_lengths"]["source_receiver_prefix_shift"] for row in rows
        ),
        "prefix_shift_max": max(
            row["token_lengths"]["source_receiver_prefix_shift"] for row in rows
        ),
    }
    if not eligible:
        return result
    outcomes = {
        "front_hybrid_matches_target": lambda row: row["front"]["hybrid"][
            "matches_target_native"
        ],
        "front_hybrid_matches_source": lambda row: row["front"]["hybrid"][
            "matches_source_native"
        ],
        "tail_identity_matches_target": lambda row: row["tail"]["identity"][
            "matches_target_native"
        ],
        "tail_stitched_matches_target": lambda row: row["tail"]["stitched"][
            "matches_target_native"
        ],
        "tail_stitched_matches_source": lambda row: row["tail"]["stitched"][
            "matches_source_native"
        ],
        "tail_keep_source_matches_target": lambda row: row["tail"][
            "keep_source_prefix"
        ]["matches_target_native"],
        "tail_keep_source_matches_source": lambda row: row["tail"][
            "keep_source_prefix"
        ]["matches_source_native"],
    }
    for name, predicate in outcomes.items():
        count = sum(predicate(row) for row in eligible)
        result[name] = count
        result[f"{name}_rate"] = _rate(count, len(eligible))

    front_target = [
        row["front"]["hybrid"]["matches_target_native"] for row in eligible
    ]
    tail_target = [
        row["tail"]["stitched"]["matches_target_native"] for row in eligible
    ]
    tail_only = sum(tail and not front for front, tail in zip(front_target, tail_target))
    front_only = sum(front and not tail for front, tail in zip(front_target, tail_target))
    result["paired_tail_vs_front"] = {
        "tail_only": tail_only,
        "front_only": front_only,
        "paired_target_recovery": (
            sum(tail_target) - sum(front_target)
        ) / len(eligible),
        "exact_mcnemar_two_sided_p": _exact_mcnemar(tail_only, front_only),
    }
    result["tail_stitched_median_mean_kl"] = statistics.median(
        row["tail"]["stitched"]["distribution"]["mean_kl"] for row in eligible
    )
    result["tail_stitched_median_source_leakage"] = statistics.median(
        row["tail"]["stitched"]["source_leakage_shift"] for row in eligible
    )
    result["tail_keep_source_median_mean_kl"] = statistics.median(
        row["tail"]["keep_source_prefix"]["distribution"]["mean_kl"]
        for row in eligible
    )
    return result


def _jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
