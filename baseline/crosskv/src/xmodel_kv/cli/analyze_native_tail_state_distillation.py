from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ("ce", "behavior", "state")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen equal-rate native Tail-state pilot."
    )
    for arm in ARMS:
        parser.add_argument(f"--{arm}-train", required=True)
        parser.add_argument(f"--{arm}-correct", required=True)
        parser.add_argument(f"--{arm}-shift", required=True)
    parser.add_argument("--expected-cases", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2094)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_native_tail_state_distillation(
        train_paths={arm: getattr(args, f"{arm}_train") for arm in ARMS},
        correct_paths={arm: getattr(args, f"{arm}_correct") for arm in ARMS},
        shift_paths={arm: getattr(args, f"{arm}_shift") for arm in ARMS},
        expected_cases=args.expected_cases,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def analyze_native_tail_state_distillation(
    *,
    train_paths,
    correct_paths,
    shift_paths,
    expected_cases: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if set(train_paths) != set(ARMS) or set(correct_paths) != set(ARMS) or set(
        shift_paths
    ) != set(ARMS):
        raise ValueError("all three path maps must contain exactly the frozen arms")
    if min(expected_cases, bootstrap_replicates) < 1:
        raise ValueError("case and bootstrap counts must be positive")

    internal = {
        arm: _load_rows(train_paths[arm], expected_cases=expected_cases)
        for arm in ARMS
    }
    correct = {
        arm: _load_rows(correct_paths[arm], expected_cases=expected_cases)
        for arm in ARMS
    }
    shift = {
        arm: _load_rows(shift_paths[arm], expected_cases=expected_cases)
        for arm in ARMS
    }
    reference_ids = set(internal["ce"])
    for tables in (internal, correct, shift):
        for arm, table in tables.items():
            if set(table) != reference_ids:
                raise ValueError(f"paired result IDs differ for {arm}")
    for case_id in reference_ids:
        gold = correct["ce"][case_id]["final_gold"]
        bridge = correct["ce"][case_id]["bridge_gold"]
        for tables in (internal, correct, shift):
            for arm in ARMS:
                row = tables[arm][case_id]
                if row["final_gold"] != gold or row["bridge_gold"] != bridge:
                    raise ValueError(f"gold fields differ for {case_id}")

    state_training = _training_curve(train_paths["state"])
    state_mse = _paired_delta(
        internal["state"],
        internal["ce"],
        field="native_tail_state_relative_mse",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    state_mean_mse = _mean(internal["state"], "native_tail_state_relative_mse")
    ce_mean_mse = _mean(internal["ce"], "native_tail_state_relative_mse")
    quality = {
        arm: {
            "final_vs_ce": _paired_delta(
                correct[arm],
                correct["ce"],
                field="student_f1",
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            ),
            "bridge_vs_ce": _paired_delta(
                correct[arm],
                correct["ce"],
                field="bridge_f1",
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            ),
        }
        for arm in ("behavior", "state")
    }
    causality = {
        arm: {
            "final_correct_minus_shift": _paired_delta(
                correct[arm],
                shift[arm],
                field="student_f1",
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            ),
            "bridge_correct_minus_shift": _paired_delta(
                correct[arm],
                shift[arm],
                field="bridge_f1",
                bootstrap_replicates=bootstrap_replicates,
                seed=seed,
            ),
        }
        for arm in ARMS
    }
    absolute = {
        arm: {
            "correct_final_f1": _mean(correct[arm], "student_f1"),
            "correct_bridge_f1": _mean(correct[arm], "bridge_f1"),
            "shift_final_f1": _mean(shift[arm], "student_f1"),
            "shift_bridge_f1": _mean(shift[arm], "bridge_f1"),
            "packet_bytes": _mean(correct[arm], "actual_packet_bytes"),
            "all_packets_target_invariant": all(
                row.get("packet_target_invariant") is True
                for row in correct[arm].values()
            ),
        }
        for arm in ARMS
    }
    mse_ratio = state_mean_mse / ce_mean_mse
    gates = {
        "training_state_loss_drops_30pct": state_training["last20_over_first20"]
        <= 0.70,
        "heldout_state_mse_improves_10pct_with_ci": (
            mse_ratio <= 0.90 and state_mse["bootstrap_95ci"][1] < 0
        ),
        "final_quality_nonnegative_with_margin": (
            quality["state"]["final_vs_ce"]["mean_delta"] >= 0
            and quality["state"]["final_vs_ce"]["bootstrap_95ci"][0] > -0.05
        ),
        "state_source_causality": (
            causality["state"]["final_correct_minus_shift"]["mean_delta"] > 0
            and causality["state"]["final_correct_minus_shift"][
                "bootstrap_95ci"
            ][0]
            > 0
        ),
    }
    return {
        "analysis": "equal_rate_native_tail_state_distillation",
        "cases": expected_cases,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "training_state_loss": state_training,
        "heldout_state_mse": {
            "ce_mean": ce_mean_mse,
            "state_mean": state_mean_mse,
            "state_over_ce": mse_ratio,
            "state_minus_ce": state_mse,
        },
        "absolute_real_canonical": absolute,
        "quality_real_canonical": quality,
        "source_causality_real_canonical": causality,
        "gates": {**gates, "authorize_cap8_run": all(gates.values())},
    }


def _load_rows(path, *, expected_cases: int) -> dict[str, dict]:
    path = Path(path)
    if path.is_dir():
        path = path / "results.jsonl"
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("quant_bits") != 4 or row.get("layer_pattern") != "all":
                continue
            case_id = row["id"]
            if case_id in rows:
                raise ValueError(f"duplicate result ID: {case_id}")
            rows[case_id] = row
    if len(rows) != expected_cases:
        raise ValueError(f"expected {expected_cases} cases, found {len(rows)}")
    return rows


def _training_curve(path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "training_log.jsonl"
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) < 40:
        raise ValueError("training curve requires at least 40 steps")
    first = float(np.mean([row["native_tail_state"] for row in rows[:20]]))
    last = float(np.mean([row["native_tail_state"] for row in rows[-20:]]))
    return {
        "steps": len(rows),
        "first20_mean": first,
        "last20_mean": last,
        "last20_over_first20": last / first,
        "relative_reduction": 1 - last / first,
    }


def _mean(table, field: str) -> float:
    return float(np.mean([row[field] for row in table.values()]))


def _paired_delta(
    candidate,
    reference,
    *,
    field: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if set(candidate) != set(reference):
        raise ValueError("paired result IDs differ")
    ids = sorted(candidate)
    delta = np.asarray(
        [candidate[case_id][field] - reference[case_id][field] for case_id in ids],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    sampled = delta[
        generator.integers(0, len(delta), size=(bootstrap_replicates, len(delta)))
    ].mean(axis=1)
    return {
        "field": field,
        "mean_delta": float(delta.mean()),
        "bootstrap_95ci": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
        "wins": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "losses": int(np.sum(delta < 0)),
    }


if __name__ == "__main__":
    main()
