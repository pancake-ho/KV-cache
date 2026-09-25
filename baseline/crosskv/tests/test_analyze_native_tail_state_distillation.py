import json

from xmodel_kv.cli.analyze_native_tail_state_distillation import (
    analyze_native_tail_state_distillation,
)


def _write_arm(path, *, state_mse, correct, shifted, native_state):
    train = path / "train"
    good = path / "correct"
    bad = path / "shift"
    train.mkdir(parents=True)
    good.mkdir()
    bad.mkdir()
    with (train / "training_log.jsonl").open("w") as handle:
        for step in range(1, 41):
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "native_tail_state": native_state[0]
                        if step <= 20
                        else native_state[1],
                    }
                )
                + "\n"
            )
    for output, score in ((train, correct), (good, correct), (bad, shifted)):
        with (output / "results.jsonl").open("w") as handle:
            for index in range(4):
                handle.write(
                    json.dumps(
                        {
                            "id": f"case-{index}",
                            "quant_bits": 4,
                            "layer_pattern": "all",
                            "final_gold": "answer",
                            "bridge_gold": "bridge",
                            "student_f1": score,
                            "bridge_f1": score,
                            "native_tail_state_relative_mse": state_mse,
                            "actual_packet_bytes": 100,
                            "packet_target_invariant": True,
                        }
                    )
                    + "\n"
                )
    return train, good, bad


def test_native_tail_analysis_authorizes_passing_state_arm(tmp_path):
    paths = {
        "ce": _write_arm(
            tmp_path / "ce", state_mse=1.0, correct=0.5, shifted=0.1,
            native_state=(0.0, 0.0)
        ),
        "behavior": _write_arm(
            tmp_path / "behavior", state_mse=1.0, correct=0.4, shifted=0.1,
            native_state=(0.0, 0.0)
        ),
        "state": _write_arm(
            tmp_path / "state", state_mse=0.5, correct=0.7, shifted=0.1,
            native_state=(1.0, 0.5)
        ),
    }
    result = analyze_native_tail_state_distillation(
        train_paths={arm: values[0] for arm, values in paths.items()},
        correct_paths={arm: values[1] for arm, values in paths.items()},
        shift_paths={arm: values[2] for arm, values in paths.items()},
        expected_cases=4,
        bootstrap_replicates=100,
        seed=1,
    )

    assert result["heldout_state_mse"]["state_over_ce"] == 0.5
    assert result["gates"]["authorize_cap8_run"]
