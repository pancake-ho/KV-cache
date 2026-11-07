from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from ..hotpotqa import normalize_answer
from .summarize_policy_triad import _exact_mcnemar, _rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize two-agent HotpotQA accuracy")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument(
        "--deduplicate-question",
        action="store_true",
        help="keep the first row for repeated normalized question text across splits",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [row for path in args.inputs for row in _jsonl(Path(path))]
    input_rows = len(rows)
    if args.deduplicate_question:
        rows = _deduplicate_questions(rows)
    duplicate_questions_dropped = input_rows - len(rows)
    # Local indices restart for each LongBench split (hotpotqa and hotpotqa_e),
    # while the dataset's stable question IDs remain globally unique.
    row_keys = [row.get("id") or (row.get("dataset"), row["index"]) for row in rows]
    if len(set(row_keys)) != len(row_keys):
        raise SystemExit("duplicate HotpotQA question IDs detected")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(rows)}")
    summary = _summarize(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["input_rows"] = input_rows
    summary["duplicate_questions_dropped"] = duplicate_questions_dropped
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _summarize(rows, *, bootstrap_samples, bootstrap_seed):
    if not rows:
        raise ValueError("cannot summarize an empty HotpotQA run")
    has_source_readout = all("source_ab" in row for row in rows)
    systems = ("direct_b", "native_ab", "identity_ab", "reuse_ab")
    if has_source_readout:
        systems = ("direct_b", "source_ab", "native_ab", "identity_ab", "reuse_ab")
    accuracy = {
        system: {
            "em": statistics.mean(row[system]["em"] for row in rows),
            "f1": statistics.mean(row[system]["f1"] for row in rows),
        }
        for system in systems
    }
    native_reuse_flips = _paired_em_flips(rows, "native_ab", "reuse_ab")
    identity_reuse_flips = _paired_em_flips(rows, "identity_ab", "reuse_ab")
    native_identity_flips = _paired_em_flips(rows, "native_ab", "identity_ab")
    lengths = [row["token_lengths"]["shared_task_and_report"] for row in rows]
    saved_fractions = [
        row["token_lengths"]["reuse_b_transferred_tokens"]
        / row["token_lengths"]["native_b_prefill"]
        for row in rows
    ]
    summary = {
        "rows": len(rows),
        "protocols": sorted({row.get("protocol", "legacy") for row in rows}),
        "accuracy": accuracy,
        "reuse_minus_native": {
            "em": _bootstrap_delta(
                rows,
                "em",
                "reuse_ab",
                "native_ab",
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "f1": _bootstrap_delta(
                rows,
                "f1",
                "reuse_ab",
                "native_ab",
                samples=bootstrap_samples,
                seed=bootstrap_seed + 1,
            ),
        },
        # Primary causal contrast: both arms use the same chunked/stitching path;
        # only the role under which the shared history KV was constructed changes.
        "reuse_minus_identity": {
            "em": _bootstrap_delta(
                rows,
                "em",
                "reuse_ab",
                "identity_ab",
                samples=bootstrap_samples,
                seed=bootstrap_seed + 4,
            ),
            "f1": _bootstrap_delta(
                rows,
                "f1",
                "reuse_ab",
                "identity_ab",
                samples=bootstrap_samples,
                seed=bootstrap_seed + 5,
            ),
        },
        # Quantifies numerical/execution drift from splitting a long prefill and
        # stitching an otherwise identical target cache.
        "identity_minus_native": {
            "em": _bootstrap_delta(
                rows,
                "em",
                "identity_ab",
                "native_ab",
                samples=bootstrap_samples,
                seed=bootstrap_seed + 6,
            ),
            "f1": _bootstrap_delta(
                rows,
                "f1",
                "identity_ab",
                "native_ab",
                samples=bootstrap_samples,
                seed=bootstrap_seed + 7,
            ),
        },
        "native_minus_direct": {
            "em": _bootstrap_delta(
                rows,
                "em",
                "native_ab",
                "direct_b",
                samples=bootstrap_samples,
                seed=bootstrap_seed + 2,
            ),
            "f1": _bootstrap_delta(
                rows,
                "f1",
                "native_ab",
                "direct_b",
                samples=bootstrap_samples,
                seed=bootstrap_seed + 3,
            ),
        },
        "paired_em": {
            "reuse_vs_native": native_reuse_flips,
            "reuse_vs_identity": identity_reuse_flips,
            "identity_vs_native": native_identity_flips,
        },
        "answer_agreement": {
            "reuse_vs_native": _rate(
                sum(row["reuse_ab"]["native_answer_agreement"] for row in rows),
                len(rows),
            ),
            "identity_vs_native": _rate(
                sum(row["identity_ab"]["native_answer_agreement"] for row in rows),
                len(rows),
            ),
            "reuse_vs_identity": _rate(
                sum(
                    normalize_answer(row["reuse_ab"]["answer"])
                    == normalize_answer(row["identity_ab"]["answer"])
                    for row in rows
                ),
                len(rows),
            ),
        },
        "agent_a_report_contains_gold": _rate(
            sum(row["agent_a"]["report_contains_gold"] for row in rows), len(rows)
        ),
        "agent_a_gold_cells": _agent_a_gold_cells(rows),
        "tokens": {
            "shared_task_and_report_min": min(lengths),
            "shared_task_and_report_median": statistics.median(lengths),
            "shared_task_and_report_max": max(lengths),
            "mean_b_prefill_fraction_reused": statistics.mean(saved_fractions),
            "median_b_prefill_fraction_reused": statistics.median(saved_fractions),
        },
        "length_cells": _length_cells(
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 100,
        ),
    }
    if has_source_readout:
        source_identity_disagree = [
            row
            for row in rows
            if normalize_answer(row["source_ab"]["answer"])
            != normalize_answer(row["identity_ab"]["answer"])
        ]
        summary["source_policy_readout"] = {
            "reuse_source_answer_agreement": _rate(
                sum(
                    normalize_answer(row["reuse_ab"]["answer"])
                    == normalize_answer(row["source_ab"]["answer"])
                    for row in rows
                ),
                len(rows),
            ),
            "identity_source_answer_agreement": _rate(
                sum(
                    normalize_answer(row["identity_ab"]["answer"])
                    == normalize_answer(row["source_ab"]["answer"])
                    for row in rows
                ),
                len(rows),
            ),
            "when_source_and_identity_disagree": {
                "rows": len(source_identity_disagree),
                "reuse_matches_source": sum(
                    normalize_answer(row["reuse_ab"]["answer"])
                    == normalize_answer(row["source_ab"]["answer"])
                    for row in source_identity_disagree
                ),
                "reuse_matches_identity": sum(
                    normalize_answer(row["reuse_ab"]["answer"])
                    == normalize_answer(row["identity_ab"]["answer"])
                    for row in source_identity_disagree
                ),
                "reuse_matches_neither": sum(
                    normalize_answer(row["reuse_ab"]["answer"])
                    not in {
                        normalize_answer(row["source_ab"]["answer"]),
                        normalize_answer(row["identity_ab"]["answer"]),
                    }
                    for row in source_identity_disagree
                ),
            },
            "memo_schema_rate": {
                system: _rate(
                    sum(_has_memo_schema(row[system]["generation"]) for row in rows),
                    len(rows),
                )
                for system in ("source_ab", "identity_ab", "reuse_ab")
            },
            "agent_a_memo_schema_valid": _rate(
                sum(_has_memo_schema(row["agent_a"]["report"]) for row in rows),
                len(rows),
            ),
        }
    return summary


def _has_memo_schema(text):
    lowered = text.lower()
    return "bridge:" in lowered and "evidence:" in lowered


def _deduplicate_questions(rows):
    deduplicated = []
    seen = set()
    for row in rows:
        key = " ".join(row["question"].casefold().split())
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(row)
    return deduplicated


def _agent_a_gold_cells(rows):
    cells = []
    for contains_gold in (True, False):
        cell = [
            row
            for row in rows
            if bool(row["agent_a"]["report_contains_gold"]) is contains_gold
        ]
        if not cell:
            continue
        cells.append(
            {
                "report_contains_gold": contains_gold,
                "rows": len(cell),
                "direct_em": statistics.mean(row["direct_b"]["em"] for row in cell),
                "identity_em": statistics.mean(
                    row["identity_ab"]["em"] for row in cell
                ),
                "reuse_em": statistics.mean(row["reuse_ab"]["em"] for row in cell),
                "direct_f1": statistics.mean(row["direct_b"]["f1"] for row in cell),
                "identity_f1": statistics.mean(
                    row["identity_ab"]["f1"] for row in cell
                ),
                "reuse_f1": statistics.mean(row["reuse_ab"]["f1"] for row in cell),
                "reuse_minus_identity_em": statistics.mean(
                    row["reuse_ab"]["em"] - row["identity_ab"]["em"]
                    for row in cell
                ),
                "reuse_minus_identity_f1": statistics.mean(
                    row["reuse_ab"]["f1"] - row["identity_ab"]["f1"]
                    for row in cell
                ),
                "reuse_identity_answer_agreement": statistics.mean(
                    normalize_answer(row["reuse_ab"]["answer"])
                    == normalize_answer(row["identity_ab"]["answer"])
                    for row in cell
                ),
            }
        )
    return cells


def _paired_em_flips(rows, reference, candidate):
    reference_correct_candidate_wrong = sum(
        row[reference]["em"] == 1 and row[candidate]["em"] == 0 for row in rows
    )
    reference_wrong_candidate_correct = sum(
        row[reference]["em"] == 0 and row[candidate]["em"] == 1 for row in rows
    )
    return {
        "reference_correct_candidate_wrong": reference_correct_candidate_wrong,
        "reference_wrong_candidate_correct": reference_wrong_candidate_correct,
        "exact_mcnemar_two_sided_p": _exact_mcnemar(
            reference_correct_candidate_wrong, reference_wrong_candidate_correct
        ),
    }


def _bootstrap_delta(rows, metric, first, second, *, samples, seed):
    observed = statistics.mean(
        row[first][metric] - row[second][metric] for row in rows
    )
    if samples < 1:
        return {"value": observed, "bootstrap_95_low": None, "bootstrap_95_high": None}
    generator = random.Random(seed)
    deltas = []
    for _ in range(samples):
        draw = [rows[generator.randrange(len(rows))] for _ in rows]
        deltas.append(
            statistics.mean(row[first][metric] - row[second][metric] for row in draw)
        )
    deltas.sort()
    return {
        "value": observed,
        "bootstrap_95_low": _quantile(deltas, 0.025),
        "bootstrap_95_high": _quantile(deltas, 0.975),
    }


def _length_cells(rows, *, bootstrap_samples, bootstrap_seed):
    boundaries = (
        ("<8K", 0, 8000),
        ("8K-12K", 8000, 12000),
        ("12K-16K", 12000, 16000),
        (">=16K", 16000, float("inf")),
    )
    cells = []
    for cell_index, (label, lower, upper) in enumerate(boundaries):
        cell = [
            row
            for row in rows
            if lower <= row["token_lengths"]["shared_task_and_report"] < upper
        ]
        if not cell:
            continue
        cells.append(
            {
                "range": label,
                "rows": len(cell),
                "native_f1": statistics.mean(row["native_ab"]["f1"] for row in cell),
                "native_em": statistics.mean(row["native_ab"]["em"] for row in cell),
                "identity_f1": statistics.mean(
                    row["identity_ab"]["f1"] for row in cell
                ),
                "identity_em": statistics.mean(
                    row["identity_ab"]["em"] for row in cell
                ),
                "reuse_f1": statistics.mean(row["reuse_ab"]["f1"] for row in cell),
                "reuse_em": statistics.mean(row["reuse_ab"]["em"] for row in cell),
                "reuse_minus_native_f1": statistics.mean(
                    row["reuse_ab"]["f1"] - row["native_ab"]["f1"]
                    for row in cell
                ),
                "reuse_minus_identity_f1": statistics.mean(
                    row["reuse_ab"]["f1"] - row["identity_ab"]["f1"]
                    for row in cell
                ),
                "reuse_minus_identity": {
                    "em": _bootstrap_delta(
                        cell,
                        "em",
                        "reuse_ab",
                        "identity_ab",
                        samples=bootstrap_samples,
                        seed=bootstrap_seed + 2 * cell_index,
                    ),
                    "f1": _bootstrap_delta(
                        cell,
                        "f1",
                        "reuse_ab",
                        "identity_ab",
                        samples=bootstrap_samples,
                        seed=bootstrap_seed + 2 * cell_index + 1,
                    ),
                },
                "paired_em_reuse_vs_identity": _paired_em_flips(
                    cell, "identity_ab", "reuse_ab"
                ),
                "reuse_native_answer_agreement": statistics.mean(
                    row["reuse_ab"]["native_answer_agreement"] for row in cell
                ),
                "reuse_identity_answer_agreement": statistics.mean(
                    normalize_answer(row["reuse_ab"]["answer"])
                    == normalize_answer(row["identity_ab"]["answer"])
                    for row in cell
                ),
            }
        )
    return cells


def _quantile(values, probability):
    if not values:
        raise ValueError("quantile needs values")
    position = probability * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
