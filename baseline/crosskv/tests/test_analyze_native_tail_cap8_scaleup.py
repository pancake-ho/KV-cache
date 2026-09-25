from test_analyze_native_tail_state_distillation import _write_arm

from xmodel_kv.cli.analyze_native_tail_cap8_scaleup import (
    analyze_native_tail_cap8_scaleup,
)


def test_cap8_scaleup_analysis_applies_all_frozen_gates(tmp_path):
    paths = {
        "state4": _write_arm(
            tmp_path / "state4", state_mse=0.8, correct=0.6, shifted=0.1,
            native_state=(1.0, 0.5)
        ),
        "ce8": _write_arm(
            tmp_path / "ce8", state_mse=1.0, correct=0.6, shifted=0.1,
            native_state=(0.0, 0.0)
        ),
        "state8": _write_arm(
            tmp_path / "state8", state_mse=0.5, correct=0.8, shifted=0.1,
            native_state=(1.0, 0.5)
        ),
    }
    result = analyze_native_tail_cap8_scaleup(
        train_paths={arm: values[0] for arm, values in paths.items()},
        correct_paths={arm: values[1] for arm, values in paths.items()},
        shift_paths={arm: values[2] for arm, values in paths.items()},
        expected_cases=4,
        bootstrap_replicates=100,
        seed=1,
    )

    assert result["heldout_state_mse"]["state8_over_ce8"] == 0.5
    assert result["gates"]["authorize_cross_task_run"]
