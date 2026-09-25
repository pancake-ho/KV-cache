from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_native_tail_state_distillation import (
    _load_rows,
    _mean,
    _paired_delta,
    _training_curve,
)


ARMS = ("state4", "ce8", "state8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen cap8 scale-up.")
    for arm in ARMS:
        parser.add_argument(f"--{arm}-train", required=True)
        parser.add_argument(f"--{arm}-correct", required=True)
        parser.add_argument(f"--{arm}-shift", required=True)
    parser.add_argument("--expected-cases", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze_native_tail_cap8_scaleup(
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


def analyze_native_tail_cap8_scaleup(
    *,
    train_paths,
    correct_paths,
    shift_paths,
    expected_cases: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if any(set(paths) != set(ARMS) for paths in (train_paths, correct_paths, shift_paths)):
        raise ValueError("all path maps must contain exactly the frozen scale-up arms")
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
    ids = set(internal["state4"])
    if any(set(table) != ids for group in (internal, correct, shift) for table in group.values()):
        raise ValueError("paired result IDs differ")
    for case_id in ids:
        final_gold = correct["state4"][case_id]["final_gold"]
        bridge_gold = correct["state4"][case_id]["bridge_gold"]
        if any(
            table[case_id]["final_gold"] != final_gold
            or table[case_id]["bridge_gold"] != bridge_gold
            for group in (internal, correct, shift)
            for table in group.values()
        ):
            raise ValueError(f"gold fields differ for {case_id}")

    state8_training = _training_curve(train_paths["state8"])
    ce8_mse = _mean(internal["ce8"], "native_tail_state_relative_mse")
    state8_mse = _mean(internal["state8"], "native_tail_state_relative_mse")
    state8_vs_ce8_mse = _paired_delta(
        internal["state8"], internal["ce8"],
        field="native_tail_state_relative_mse",
        bootstrap_replicates=bootstrap_replicates, seed=seed,
    )
    state8_vs_ce8 = _paired_delta(
        correct["state8"], correct["ce8"], field="student_f1",
        bootstrap_replicates=bootstrap_replicates, seed=seed,
    )
    state8_vs_state4 = _paired_delta(
        correct["state8"], correct["state4"], field="student_f1",
        bootstrap_replicates=bootstrap_replicates, seed=seed,
    )
    state8_causality = _paired_delta(
        correct["state8"], shift["state8"], field="student_f1",
        bootstrap_replicates=bootstrap_replicates, seed=seed,
    )
    bridge = {
        "state8_minus_ce8": _paired_delta(
            correct["state8"], correct["ce8"], field="bridge_f1",
            bootstrap_replicates=bootstrap_replicates, seed=seed,
        ),
        "state8_minus_state4": _paired_delta(
            correct["state8"], correct["state4"], field="bridge_f1",
            bootstrap_replicates=bootstrap_replicates, seed=seed,
        ),
    }
    absolute = {
        arm: {
            "correct_final_f1": _mean(correct[arm], "student_f1"),
            "correct_bridge_f1": _mean(correct[arm], "bridge_f1"),
            "shift_final_f1": _mean(shift[arm], "student_f1"),
            "packet_bytes": _mean(correct[arm], "actual_packet_bytes"),
            "all_packets_target_invariant": all(
                row.get("packet_target_invariant") is True
                for row in correct[arm].values()
            ),
        }
        for arm in ARMS
    }
    ratio = state8_mse / ce8_mse
    gates = {
        "state8_mse_at_most_75pct_ce8": (
            ratio <= 0.75 and state8_vs_ce8_mse["bootstrap_95ci"][1] < 0
        ),
        "state8_noninferior_to_ce8": (
            state8_vs_ce8["mean_delta"] >= 0
            and state8_vs_ce8["bootstrap_95ci"][0] > -0.03
        ),
        "state8_capacity_gain_over_state4": (
            state8_vs_state4["mean_delta"] >= 0.02
            and state8_vs_state4["bootstrap_95ci"][0] > -0.02
        ),
        "state8_source_causality": (
            state8_causality["mean_delta"] > 0
            and state8_causality["bootstrap_95ci"][0] > 0
        ),
        "state8_training_loss_drops_30pct": (
            state8_training["last20_over_first20"] <= 0.70
        ),
    }
    return {
        "analysis": "native_tail_state_cap8_scaleup",
        "cases": expected_cases,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "state8_training_state_loss": state8_training,
        "heldout_state_mse": {
            "ce8_mean": ce8_mse,
            "state8_mean": state8_mse,
            "state8_over_ce8": ratio,
            "state8_minus_ce8": state8_vs_ce8_mse,
        },
        "absolute_real_canonical": absolute,
        "final_quality": {
            "state8_minus_ce8": state8_vs_ce8,
            "state8_minus_state4": state8_vs_state4,
        },
        "bridge_quality": bridge,
        "state8_source_causality": state8_causality,
        "gates": {**gates, "authorize_cross_task_run": all(gates.values())},
    }


if __name__ == "__main__":
    main()
