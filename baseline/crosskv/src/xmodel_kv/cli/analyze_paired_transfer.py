from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired exact-accuracy and bootstrap-F1 analysis for transfer results."
    )
    parser.add_argument("--candidate-results", required=True)
    parser.add_argument("--candidate-quant-bits", type=int, default=16)
    parser.add_argument("--candidate-layer-pattern", default="all")
    parser.add_argument("--reference-results")
    parser.add_argument("--reference-quant-bits", type=int, default=16)
    parser.add_argument("--reference-layer-pattern", default="all")
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument(
        "--min-index",
        type=int,
        default=None,
        help="Optional inclusive dataset-index floor for a preregistered holdout.",
    )
    parser.add_argument(
        "--cluster-fields",
        default="",
        help="Comma-separated candidate-row fields defining paired bootstrap clusters.",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    if args.min_index is not None and args.min_index < 0:
        parser.error("--min-index must be non-negative")

    candidate = _load_rows(
        args.candidate_results,
        args.candidate_quant_bits,
        layer_pattern=args.candidate_layer_pattern,
        min_index=args.min_index,
    )
    if args.reference_results:
        reference = _load_rows(
            args.reference_results,
            args.reference_quant_bits,
            layer_pattern=args.reference_layer_pattern,
            min_index=args.min_index,
        )
        reference_exact_key = "student_exact"
        reference_f1_key = "student_f1"
        reference_name = str(args.reference_results)
    else:
        reference = candidate
        reference_exact_key = "no_summary_exact"
        reference_f1_key = "no_summary_f1"
        reference_name = "candidate.no_summary"
    if candidate.keys() != reference.keys():
        missing_reference = sorted(candidate.keys() - reference.keys())
        missing_candidate = sorted(reference.keys() - candidate.keys())
        raise ValueError(
            "paired result IDs differ: "
            f"missing_reference={missing_reference[:5]}, "
            f"missing_candidate={missing_candidate[:5]}"
        )

    case_ids = sorted(candidate)
    cluster_fields = tuple(
        field.strip() for field in args.cluster_fields.split(",") if field.strip()
    )
    summary = paired_summary(
        [candidate[case_id]["student_exact"] for case_id in case_ids],
        [reference[case_id][reference_exact_key] for case_id in case_ids],
        [candidate[case_id]["student_f1"] for case_id in case_ids],
        [reference[case_id][reference_f1_key] for case_id in case_ids],
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    summary.update(
        {
            "candidate_results": str(args.candidate_results),
            "candidate_quant_bits": args.candidate_quant_bits,
            "candidate_layer_pattern": args.candidate_layer_pattern,
            "min_index": args.min_index,
            "reference_results": reference_name,
            "reference_quant_bits": (
                args.reference_quant_bits if args.reference_results else None
            ),
            "reference_layer_pattern": (
                args.reference_layer_pattern if args.reference_results else None
            ),
        }
    )
    if cluster_fields:
        missing = [
            field
            for field in cluster_fields
            if any(field not in candidate[case_id] for case_id in case_ids)
        ]
        if missing:
            raise ValueError(f"cluster fields missing from candidate rows: {missing}")
        labels = [
            tuple(candidate[case_id][field] for field in cluster_fields)
            for case_id in case_ids
        ]
        deltas = [
            candidate[case_id]["student_f1"]
            - reference[case_id][reference_f1_key]
            for case_id in case_ids
        ]
        cluster_ci, cluster_count = cluster_bootstrap_mean_ci(
            deltas,
            labels,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        summary["cluster_fields"] = list(cluster_fields)
        summary["cluster_count"] = cluster_count
        summary["mean_f1_delta_cluster_bootstrap_95ci"] = cluster_ci
    bridge_values = [candidate[case_id].get("bridge_exact") for case_id in case_ids]
    if all(value is not None for value in bridge_values):
        summary["candidate_bridge_accuracy"] = float(np.mean(bridge_values))
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")


def _load_rows(
    path: str | Path,
    quant_bits: int,
    *,
    layer_pattern: str = "all",
    min_index: int | None = None,
) -> dict[str, dict]:
    selected = {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if min_index is not None and int(row.get("index", -1)) < min_index:
                continue
            if int(row.get("quant_bits", -1)) != quant_bits:
                continue
            if row.get("layer_pattern", "all") != layer_pattern:
                continue
            if row.get("head_pattern", "all") != "all":
                continue
            if row.get("head_rank", "full") != "full":
                continue
            case_id = row["id"]
            if case_id in selected:
                raise ValueError(f"duplicate selected result ID: {case_id}")
            selected[case_id] = row
    if not selected:
        raise ValueError(
            f"no quant_bits={quant_bits}, layer_pattern={layer_pattern} rows in {path}"
        )
    return selected


def paired_summary(
    candidate_exact,
    reference_exact,
    candidate_f1,
    reference_f1,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    candidate_exact = np.asarray(candidate_exact, dtype=np.float64)
    reference_exact = np.asarray(reference_exact, dtype=np.float64)
    candidate_f1 = np.asarray(candidate_f1, dtype=np.float64)
    reference_f1 = np.asarray(reference_f1, dtype=np.float64)
    sizes = {
        len(candidate_exact),
        len(reference_exact),
        len(candidate_f1),
        len(reference_f1),
    }
    if len(sizes) != 1 or not candidate_exact.size:
        raise ValueError("paired metric vectors must be non-empty and equally sized")
    candidate_only = int(np.sum((candidate_exact == 1) & (reference_exact == 0)))
    reference_only = int(np.sum((candidate_exact == 0) & (reference_exact == 1)))
    both_correct = int(np.sum((candidate_exact == 1) & (reference_exact == 1)))
    neither_correct = int(np.sum((candidate_exact == 0) & (reference_exact == 0)))
    discordant = candidate_only + reference_only
    mcnemar_p = _two_sided_binomial_p(candidate_only, discordant)
    delta_f1 = candidate_f1 - reference_f1
    delta_exact = candidate_exact - reference_exact
    generator = np.random.default_rng(seed)
    sample_indices = generator.integers(
        0, len(delta_f1), size=(bootstrap_replicates, len(delta_f1))
    )
    bootstrap_means = delta_f1[sample_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    bootstrap_accuracy = delta_exact[sample_indices].mean(axis=1)
    accuracy_lower, accuracy_upper = np.quantile(
        bootstrap_accuracy, [0.025, 0.975]
    )
    return {
        "cases": int(candidate_exact.size),
        "candidate_accuracy": float(candidate_exact.mean()),
        "reference_accuracy": float(reference_exact.mean()),
        "accuracy_delta": float((candidate_exact - reference_exact).mean()),
        "accuracy_delta_bootstrap_95ci": [
            float(accuracy_lower),
            float(accuracy_upper),
        ],
        "candidate_only_correct": candidate_only,
        "reference_only_correct": reference_only,
        "both_correct": both_correct,
        "neither_correct": neither_correct,
        "oracle_union_accuracy": float(
            np.maximum(candidate_exact, reference_exact).mean()
        ),
        "exact_mcnemar_two_sided_p": mcnemar_p,
        "candidate_mean_f1": float(candidate_f1.mean()),
        "reference_mean_f1": float(reference_f1.mean()),
        "mean_f1_delta": float(delta_f1.mean()),
        "mean_f1_delta_bootstrap_95ci": [float(lower), float(upper)],
        "f1_wins": int(np.sum(delta_f1 > 0)),
        "f1_ties": int(np.sum(delta_f1 == 0)),
        "f1_losses": int(np.sum(delta_f1 < 0)),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
    }


def _two_sided_binomial_p(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, value) for value in range(min(successes, trials - successes) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def cluster_bootstrap_mean_ci(
    values,
    labels,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[float], int]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(labels) or not len(values):
        raise ValueError("cluster values and labels must be non-empty and equally sized")
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    grouped_indices = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        sampled_groups = generator.integers(0, len(grouped_indices), len(grouped_indices))
        sampled_indices = np.concatenate(
            [grouped_indices[group_index] for group_index in sampled_groups]
        )
        bootstrap_means[replicate] = values[sampled_indices].mean()
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return [float(lower), float(upper)], len(grouped_indices)


if __name__ == "__main__":
    main()
