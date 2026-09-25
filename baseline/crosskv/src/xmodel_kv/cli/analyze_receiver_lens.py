from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("student_f1", "bridge_f1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply frozen Receiver-Lens gates.")
    parser.add_argument("--dev-correct", required=True)
    parser.add_argument("--dev-shift", required=True)
    parser.add_argument("--musique-correct", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2112)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("bootstrap count must be positive")
    groups = {
        "dev_correct": load_rows(args.dev_correct),
        "dev_shift": load_rows(args.dev_shift),
        "musique_correct": load_rows(args.musique_correct),
    }
    result = analyze_groups(
        groups, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed
    )
    result.update(
        {
            "analysis": "receiver_lens4_frozen_gates",
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
    dev = groups["dev_correct"]
    shift = groups["dev_shift"]
    musique = groups["musique_correct"]
    comparisons = {
        "dev_lens_minus_base": compare_arms(
            dev, "lens4", dev, "base", rng=rng, replicates=bootstrap_replicates
        ),
        "dev_lens_minus_hard4": compare_arms(
            dev, "lens4", dev, "hard4", rng=rng, replicates=bootstrap_replicates
        ),
        "dev_lens_correct_minus_shift": compare_arms(
            dev, "lens4", shift, "lens4", rng=rng, replicates=bootstrap_replicates
        ),
        "musique_lens_minus_base": compare_arms(
            musique,
            "lens4",
            musique,
            "base",
            rng=rng,
            replicates=bootstrap_replicates,
        ),
    }
    dev_base = comparisons["dev_lens_minus_base"]
    dev_hard = comparisons["dev_lens_minus_hard4"]
    causal = comparisons["dev_lens_correct_minus_shift"]
    safety = comparisons["musique_lens_minus_base"]
    gates = {
        "dev_final_improvement": (
            dev_base["student_f1"]["mean_delta"] >= 0.04
            and dev_base["student_f1"]["bootstrap_95ci"][0] > 0
        ),
        "dev_bridge_safety": (
            dev_base["bridge_f1"]["mean_delta"] >= -0.03
            and dev_base["bridge_f1"]["bootstrap_95ci"][0] > -0.10
        ),
        "dev_source_causality": all(
            causal[metric]["bootstrap_95ci"][0] > 0 for metric in METRICS
        ),
        "hard4_control": all(
            dev_hard[metric]["mean_delta"] >= 0
            and dev_hard[metric]["bootstrap_95ci"][0] > -0.08
            for metric in METRICS
        ),
        "musique_safety": (
            safety["student_f1"]["mean_delta"] >= -0.02
            and safety["student_f1"]["bootstrap_95ci"][0] > -0.08
            and safety["bridge_f1"]["mean_delta"] >= -0.05
            and safety["bridge_f1"]["bootstrap_95ci"][0] > -0.10
        ),
    }
    return {
        "cases": {name: len(rows) for name, rows in groups.items()},
        "means": {
            name: arm_means(rows)
            for name, rows in groups.items()
        },
        "comparisons": comparisons,
        "gates": gates,
        "eviction_test_authorized": all(gates.values()),
        "external_evaluation_authorized": False,
    }


def arm_means(rows: dict) -> dict:
    arms = sorted(
        {
            field[: -len("_student_f1")]
            for row in rows.values()
            for field in row
            if field.endswith("_student_f1")
        }
    )
    return {
        arm: {
            metric: float(
                np.mean([row[f"{arm}_{metric}"] for row in rows.values()])
            )
            for metric in METRICS
            if all(f"{arm}_{metric}" in row for row in rows.values())
        }
        for arm in arms
    }


def compare_arms(
    left: dict,
    left_arm: str,
    right: dict,
    right_arm: str,
    *,
    rng,
    replicates: int,
) -> dict:
    if set(left) != set(right):
        raise ValueError("paired result IDs differ")
    ids = sorted(left)
    result = {}
    for metric in METRICS:
        left_field = f"{left_arm}_{metric}"
        right_field = f"{right_arm}_{metric}"
        if any(left_field not in left[row_id] or right_field not in right[row_id] for row_id in ids):
            raise ValueError(f"missing paired fields {left_field}/{right_field}")
        values = np.asarray(
            [left[row_id][left_field] - right[row_id][right_field] for row_id in ids],
            dtype=np.float64,
        )
        samples = values[
            rng.integers(0, len(values), size=(replicates, len(values)))
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
            rows[row["id"]] = row
    if not rows:
        raise ValueError(f"no result rows in {path}")
    return rows


if __name__ == "__main__":
    main()
