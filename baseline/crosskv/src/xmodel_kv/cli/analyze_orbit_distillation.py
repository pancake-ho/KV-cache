from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .analyze_paired_transfer import paired_summary
from .analyze_summary_transfer_variants import load_protocol_rows


PROTOCOLS = ("state_readout", "question_conditioned")
RUN_NAMES = ("parent_target", "parent_canonical", "orbit_target", "orbit_canonical")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preregistered four-arm analysis for positional-orbit distillation."
    )
    for name in RUN_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default="question_conditioned",
    )
    parser.add_argument("--expected-cases", type=int, default=150)
    parser.add_argument("--expected-steps", type=int, default=1_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(
        args.expected_cases,
        args.expected_steps,
        args.bootstrap_replicates,
    ) < 1:
        parser.error("case, step, and bootstrap counts must be positive")

    paths = {name: getattr(args, name) for name in RUN_NAMES}
    result = analyze_orbit_distillation(
        paths,
        training_dir=args.training_dir,
        protocol=args.protocol,
        expected_cases=args.expected_cases,
        expected_steps=args.expected_steps,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def analyze_orbit_distillation(
    paths: dict[str, str | Path],
    *,
    training_dir: str | Path,
    protocol: str,
    expected_cases: int,
    expected_steps: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if set(paths) != set(RUN_NAMES):
        raise ValueError(f"expected run paths {RUN_NAMES}")
    rows = {
        name: load_protocol_rows(
            [path], protocol=protocol, expected_cases=expected_cases
        )
        for name, path in paths.items()
    }
    _require_paired_cases(rows)

    comparisons = {
        "target_retention_Ot_minus_Pt": _compare(
            rows["orbit_target"],
            rows["parent_target"],
            candidate_arm="capsule",
            reference_arm="capsule",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
        "canonical_safety_Oc_minus_Ot": _compare(
            rows["orbit_canonical"],
            rows["orbit_target"],
            candidate_arm="capsule",
            reference_arm="capsule",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
        "parent_canonical_Pc_minus_Pt": _compare(
            rows["parent_canonical"],
            rows["parent_target"],
            candidate_arm="capsule",
            reference_arm="capsule",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
        "canonical_candidate_Oc_minus_Pc": _compare(
            rows["orbit_canonical"],
            rows["parent_canonical"],
            candidate_arm="capsule",
            reference_arm="capsule",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
        "source_causality_Oc_correct_minus_shifted": _compare(
            rows["orbit_canonical"],
            rows["orbit_canonical"],
            candidate_arm="capsule",
            reference_arm="shifted_capsule",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
    }
    difference_in_differences = _difference_in_differences(
        rows,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    training = audit_training(
        training_dir,
        expected_steps=expected_steps,
        expected_trainable_parameters=131_072,
    )
    wire = audit_wire(paths, expected_cases=expected_cases)

    target = comparisons["target_retention_Ot_minus_Pt"]
    canonical = comparisons["canonical_safety_Oc_minus_Ot"]
    causality = comparisons["source_causality_Oc_correct_minus_shifted"]
    gates = {
        "optimization_integrity": training["pass"],
        "target_quality_retention": (
            target["mean_f1_delta"] >= -0.01
            and target["mean_f1_delta_bootstrap_95ci"][0] > -0.03
        ),
        "canonical_safety": (
            canonical["mean_f1_delta"] >= -0.01
            and canonical["mean_f1_delta_bootstrap_95ci"][0] > -0.03
        ),
        "source_causality": causality["mean_f1_delta_bootstrap_95ci"][0] > 0,
        "wire_invariants": wire["pass"],
    }
    strong_repair = (
        difference_in_differences["f1"]["mean"] > 0
        and difference_in_differences["f1"]["bootstrap_95ci"][0] > 0
    )
    return {
        "analysis": "positional_orbit_distillation_four_arm",
        "protocol": protocol,
        "expected_cases": expected_cases,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "paths": {name: str(path) for name, path in paths.items()},
        "training_dir": str(training_dir),
        "training_audit": training,
        "wire_audit": wire,
        "comparisons": comparisons,
        "difference_in_differences": difference_in_differences,
        "gates": {
            **gates,
            "external_safety_pass": all(gates.values()),
            "strong_mechanistic_repair_pass": strong_repair,
        },
    }


def _compare(
    candidate: dict[str, dict],
    reference: dict[str, dict],
    *,
    candidate_arm: str,
    reference_arm: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    ids = sorted(candidate)
    return paired_summary(
        [candidate[case_id][f"{candidate_arm}_em"] for case_id in ids],
        [reference[case_id][f"{reference_arm}_em"] for case_id in ids],
        [candidate[case_id][f"{candidate_arm}_f1"] for case_id in ids],
        [reference[case_id][f"{reference_arm}_f1"] for case_id in ids],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )


def _require_paired_cases(rows: dict[str, dict[str, dict]]) -> None:
    first_ids = set(rows[RUN_NAMES[0]])
    for name in RUN_NAMES[1:]:
        if set(rows[name]) != first_ids:
            raise ValueError(f"result IDs differ for {name}")
    for case_id in first_ids:
        first = rows[RUN_NAMES[0]][case_id]
        for name in RUN_NAMES[1:]:
            current = rows[name][case_id]
            for field in ("question", "gold_answers"):
                if current.get(field) != first.get(field):
                    raise ValueError(f"{field} differs for {name} case {case_id}")


def _difference_in_differences(
    rows: dict[str, dict[str, dict]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    ids = sorted(rows["parent_target"])
    output = {}
    for offset, metric in enumerate(("f1", "em")):
        field = f"capsule_{metric}"
        values = np.asarray(
            [
                (
                    rows["orbit_canonical"][case_id][field]
                    - rows["orbit_target"][case_id][field]
                )
                - (
                    rows["parent_canonical"][case_id][field]
                    - rows["parent_target"][case_id][field]
                )
                for case_id in ids
            ],
            dtype=np.float64,
        )
        generator = np.random.default_rng(seed + offset)
        indices = generator.integers(
            0, len(values), size=(bootstrap_replicates, len(values))
        )
        lower, upper = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
        output[metric] = {
            "mean": float(values.mean()),
            "bootstrap_95ci": [float(lower), float(upper)],
            "positive_cases": int(np.sum(values > 0)),
            "zero_cases": int(np.sum(values == 0)),
            "negative_cases": int(np.sum(values < 0)),
        }
    output["cases"] = len(ids)
    output["bootstrap_replicates"] = bootstrap_replicates
    output["seed"] = seed
    return output


def audit_training(
    training_dir: str | Path,
    *,
    expected_steps: int,
    expected_trainable_parameters: int,
    window: int = 20,
) -> dict:
    training_dir = Path(training_dir)
    config = json.loads((training_dir / "config.json").read_text())
    logs = [
        json.loads(line)
        for line in (training_dir / "training_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(logs) != expected_steps or [row.get("step") for row in logs] != list(
        range(1, expected_steps + 1)
    ):
        raise ValueError("training log is incomplete or step order is invalid")
    if expected_steps < window:
        raise ValueError("expected steps must cover the audit window")
    numeric_fields = (
        "loss",
        "stage_ce",
        "bridge_ce",
        "teacher_kl",
        "preanswer_teacher_kl",
        "orbit_kl",
        "grad_norm",
    )
    finite = all(
        isinstance(row.get(field), (int, float)) and math.isfinite(row[field])
        for row in logs
        for field in numeric_fields
    )
    first = float(np.mean([row["orbit_kl"] for row in logs[:window]]))
    last = float(np.mean([row["orbit_kl"] for row in logs[-window:]]))
    reduction = float((first - last) / first) if first > 0 else float("-inf")
    checks = {
        "step_count_exact": config.get("steps") == expected_steps,
        "all_metrics_finite": finite,
        "orbit_kl_reduction_at_least_20pct": reduction >= 0.20,
        "trainable_parameter_count_exact": (
            config.get("trainable_parameter_count") == expected_trainable_parameters
        ),
        "frozen_capsule_unchanged": (
            config.get("frozen_capsule_parameters_unchanged") is True
            and config.get("frozen_capsule_parameter_sha256_before")
            == config.get("frozen_capsule_parameter_sha256_after")
        ),
        "trainable_capsule_changed": (
            config.get("trainable_capsule_parameters_changed") is True
            and config.get("trainable_capsule_parameter_sha256_before")
            != config.get("trainable_capsule_parameter_sha256_after")
        ),
        "orbit_base_unchanged": (
            config.get("orbit_base_parameters_unchanged") is True
            and config.get("orbit_base_parameter_sha256_before")
            == config.get("orbit_base_parameter_sha256_after")
        ),
    }
    return {
        "expected_steps": expected_steps,
        "observed_log_steps": len(logs),
        "expected_trainable_parameters": expected_trainable_parameters,
        "observed_trainable_parameters": config.get("trainable_parameter_count"),
        "orbit_kl_first20": first,
        "orbit_kl_last20": last,
        "orbit_kl_fractional_reduction": reduction,
        "checks": checks,
        "pass": all(checks.values()),
    }


def audit_wire(
    paths: dict[str, str | Path],
    *,
    expected_cases: int,
    expected_base_bytes: int = 67_642,
    expected_delta_bytes: int = 4_252,
) -> dict:
    run_audits = {}
    for name, raw_path in paths.items():
        rows = _load_all_rows(raw_path)
        expected_mode = "canonical" if name.endswith("canonical") else "target"
        checks = {
            "row_count": len(rows) == expected_cases * len(PROTOCOLS),
            "protocols_complete": _protocols_complete(rows, expected_cases),
            "position_mode": all(
                row.get("packet_position_mode") == expected_mode for row in rows
            ),
            "key_codec_cartesian": all(
                row.get("key_codec") == "cartesian" for row in rows
            ),
            "capsule_bytes": _expected_packet_bytes(
                rows,
                "capsule",
                expected_base_bytes=expected_base_bytes,
                expected_delta_bytes=expected_delta_bytes,
            ),
            "shifted_capsule_bytes": _expected_packet_bytes(
                rows,
                "shifted_capsule",
                expected_base_bytes=expected_base_bytes,
                expected_delta_bytes=expected_delta_bytes,
            ),
        }
        if expected_mode == "canonical":
            checks["receiver_independent_hashes"] = _canonical_hashes_invariant(rows)
        run_audits[name] = {"checks": checks, "pass": all(checks.values())}
    return {
        "expected_base_bytes": expected_base_bytes,
        "expected_delta_bytes": expected_delta_bytes,
        "runs": run_audits,
        "pass": all(audit["pass"] for audit in run_audits.values()),
    }


def _load_all_rows(raw_path: str | Path) -> list[dict]:
    path = Path(raw_path)
    if path.is_dir():
        path = path / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _protocols_complete(rows: list[dict], expected_cases: int) -> bool:
    by_id: dict[str, set[str]] = {}
    for row in rows:
        by_id.setdefault(row["id"], set()).add(row.get("protocol"))
    return len(by_id) == expected_cases and all(
        protocols == set(PROTOCOLS) for protocols in by_id.values()
    )


def _expected_packet_bytes(
    rows: list[dict],
    arm: str,
    *,
    expected_base_bytes: int,
    expected_delta_bytes: int,
) -> bool:
    expected = {
        f"{arm}_base_packet_bytes": expected_base_bytes,
        f"{arm}_delta_packet_bytes": expected_delta_bytes,
        f"{arm}_cold_packet_bytes": expected_base_bytes + expected_delta_bytes,
        f"{arm}_incremental_packet_bytes": expected_delta_bytes,
    }
    return all(
        all(row.get(field) == value for field, value in expected.items())
        for row in rows
    )


def _canonical_hashes_invariant(rows: list[dict]) -> bool:
    by_id: dict[str, list[dict]] = {}
    for row in rows:
        by_id.setdefault(row["id"], []).append(row)
    for case_rows in by_id.values():
        if len(case_rows) != len(PROTOCOLS):
            return False
        for arm in ("capsule", "shifted_capsule"):
            for suffix in ("base_packet_sha256", "delta_packet_sha256"):
                field = f"{arm}_{suffix}"
                if None in (values := {row.get(field) for row in case_rows}):
                    return False
                if len(values) != 1:
                    return False
    return True


if __name__ == "__main__":
    main()
