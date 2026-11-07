from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .analyze_paired_transfer import paired_summary


ARMS = ("core", "full", "shifted_full", "no_state")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen feasibility analysis for a 4+4 progressive capsule."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=16)
    parser.add_argument("--expected-steps", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_progressive_refinement(
        args.run_dir,
        expected_cases=args.expected_cases,
        expected_steps=args.expected_steps,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def analyze_progressive_refinement(
    run_dir: str | Path,
    *,
    expected_cases: int,
    expected_steps: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    logs = [
        json.loads(line)
        for line in (run_dir / "training_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != expected_cases or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("evaluation case count or IDs are invalid")
    if len(logs) != expected_steps or [row["step"] for row in logs] != list(
        range(1, expected_steps + 1)
    ):
        raise ValueError("training log is incomplete or unordered")
    ids = sorted(range(len(rows)), key=lambda index: rows[index]["id"])
    comparisons = {
        "full_minus_core": _compare(
            rows,
            ids,
            candidate="full",
            reference="core",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
        "full_minus_shifted": _compare(
            rows,
            ids,
            candidate="full",
            reference="shifted_full",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
        "full_minus_no_state": _compare(
            rows,
            ids,
            candidate="full",
            reference="no_state",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
    }
    finite = all(
        isinstance(row.get(field), (int, float)) and math.isfinite(row[field])
        for row in logs
        for field in ("loss", "grad_norm")
    )
    window = min(20, expected_steps)
    first_loss = float(np.mean([row["loss"] for row in logs[:window]]))
    last_loss = float(np.mean([row["loss"] for row in logs[-window:]]))
    reduction = (first_loss - last_loss) / first_loss if first_loss > 0 else -math.inf
    training_checks = {
        "steps_exact": config.get("steps") == expected_steps,
        "finite": finite,
        "loss_reduction_at_least_20pct": reduction >= 0.20,
        "trainable_count_exact": config.get("trainable_parameter_count") == 147_456,
        "core_unchanged": (
            config.get("core_parameters_unchanged") is True
            and config.get("core_parameter_sha256_before")
            == config.get("core_parameter_sha256_after")
        ),
        "trainables_changed": (
            config.get("trainable_parameters_changed") is True
            and config.get("trainable_parameter_sha256_before")
            != config.get("trainable_parameter_sha256_after")
        ),
    }
    wire_checks = {
        "core_bytes": all(row.get("core_packet_bytes") == 67_642 for row in rows),
        "refinement_bytes": all(
            row.get("refinement_packet_bytes") == 67_642 for row in rows
        ),
        "shift_changes_source": all(
            row.get("source_id") != row.get("shifted_source_id") for row in rows
        ),
    }
    utility = comparisons["full_minus_core"]
    causality = comparisons["full_minus_shifted"]
    gates = {
        "optimization_integrity": all(training_checks.values()),
        "refinement_utility": (
            utility["mean_f1_delta"] >= 0.02
            and utility["mean_f1_delta_bootstrap_95ci"][0] > -0.05
        ),
        "source_causality": (
            causality["mean_f1_delta"] > 0
            and causality["mean_f1_delta_bootstrap_95ci"][0] > 0
        ),
        "wire_invariants": all(wire_checks.values()),
    }
    return {
        "analysis": "progressive_refinement_feasibility",
        "run_dir": str(run_dir),
        "cases": expected_cases,
        "steps": expected_steps,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "absolute": {
            arm: {
                "accuracy": float(np.mean([row[f"{arm}_em"] for row in rows])),
                "mean_f1": float(np.mean([row[f"{arm}_f1"] for row in rows])),
            }
            for arm in ARMS
        },
        "comparisons": comparisons,
        "training_audit": {
            "first20_loss": first_loss,
            "last20_loss": last_loss,
            "fractional_reduction": reduction,
            "checks": training_checks,
        },
        "wire_audit": {"checks": wire_checks},
        "gates": {**gates, "authorize_larger_development_run": all(gates.values())},
    }


def _compare(
    rows,
    indices,
    *,
    candidate: str,
    reference: str,
    bootstrap_replicates: int,
    seed: int,
):
    return paired_summary(
        [rows[index][f"{candidate}_em"] for index in indices],
        [rows[index][f"{reference}_em"] for index in indices],
        [rows[index][f"{candidate}_f1"] for index in indices],
        [rows[index][f"{reference}_f1"] for index in indices],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )


if __name__ == "__main__":
    main()
