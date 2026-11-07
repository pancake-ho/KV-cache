from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("student_f1", "bridge_f1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply frozen gates to the multi-domain CE8 continuation."
    )
    parser.add_argument("--parent-dev", required=True)
    parser.add_argument("--parent-dev-shift", required=True)
    parser.add_argument("--candidate-dev", required=True)
    parser.add_argument("--candidate-dev-shift", required=True)
    parser.add_argument("--parent-musique", required=True)
    parser.add_argument("--candidate-musique", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2102)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("bootstrap count must be positive")

    groups = {
        "parent_dev": load_rows(args.parent_dev),
        "parent_dev_shift": load_rows(args.parent_dev_shift),
        "candidate_dev": load_rows(args.candidate_dev),
        "candidate_dev_shift": load_rows(args.candidate_dev_shift),
        "parent_musique": load_rows(args.parent_musique),
        "candidate_musique": load_rows(args.candidate_musique),
    }
    result = analyze_groups(
        groups,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    result.update(
        {
            "analysis": "multidomain_decision_writer_frozen_gates",
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def analyze_groups(groups: dict, *, bootstrap_replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    comparisons = {
        "dev_candidate_minus_parent": compare(
            groups["candidate_dev"],
            groups["parent_dev"],
            rng=rng,
            bootstrap_replicates=bootstrap_replicates,
        ),
        "dev_candidate_correct_minus_shift": compare(
            groups["candidate_dev"],
            groups["candidate_dev_shift"],
            rng=rng,
            bootstrap_replicates=bootstrap_replicates,
        ),
        "dev_parent_correct_minus_shift": compare(
            groups["parent_dev"],
            groups["parent_dev_shift"],
            rng=rng,
            bootstrap_replicates=bootstrap_replicates,
        ),
        "musique_candidate_minus_parent": compare(
            groups["candidate_musique"],
            groups["parent_musique"],
            rng=rng,
            bootstrap_replicates=bootstrap_replicates,
        ),
    }
    dev_delta = comparisons["dev_candidate_minus_parent"]
    candidate_causal = comparisons["dev_candidate_correct_minus_shift"]
    musique_delta = comparisons["musique_candidate_minus_parent"]
    gates = {
        "dev_bridge_improvement": (
            dev_delta["bridge_f1"]["mean_delta"] >= 0.05
            and dev_delta["bridge_f1"]["bootstrap_95ci"][0] > 0
        ),
        "dev_final_noninferiority": (
            dev_delta["student_f1"]["mean_delta"] >= -0.02
            and dev_delta["student_f1"]["bootstrap_95ci"][0] > -0.08
        ),
        "dev_source_causality": (
            candidate_causal["student_f1"]["bootstrap_95ci"][0] > 0
            and candidate_causal["bridge_f1"]["bootstrap_95ci"][0] > 0
        ),
        "musique_safety": (
            musique_delta["student_f1"]["mean_delta"] >= -0.03
            and musique_delta["student_f1"]["bootstrap_95ci"][0] > -0.10
            and musique_delta["bridge_f1"]["mean_delta"] >= -0.05
        ),
    }
    return {
        "cases": {
            name: len(rows) for name, rows in groups.items()
        },
        "means": {
            name: {
                metric: float(np.mean([row[metric] for row in rows.values()]))
                for metric in METRICS
            }
            for name, rows in groups.items()
        },
        "comparisons": comparisons,
        "gates": gates,
        "external_evaluation_authorized": all(gates.values()),
    }


def compare(left: dict, right: dict, *, rng, bootstrap_replicates: int) -> dict:
    if set(left) != set(right):
        raise ValueError("paired result IDs differ")
    ids = sorted(left)
    result = {}
    for metric in METRICS:
        values = np.asarray(
            [left[row_id][metric] - right[row_id][metric] for row_id in ids],
            dtype=np.float64,
        )
        samples = values[
            rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
        ].mean(axis=1)
        result[metric] = {
            "mean_delta": float(values.mean()),
            "bootstrap_95ci": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
            "wins": int(np.sum(values > 0)),
            "ties": int(np.sum(values == 0)),
            "losses": int(np.sum(values < 0)),
        }
    return result


def load_rows(path: str) -> dict[str, dict]:
    rows = {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("quant_bits") == 4 and row.get("layer_pattern") == "all":
                rows[row["id"]] = row
    if not rows:
        raise ValueError(f"no all-layer K4 rows in {path}")
    return rows


if __name__ == "__main__":
    main()
