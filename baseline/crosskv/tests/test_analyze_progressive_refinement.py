import json

from xmodel_kv.cli.analyze_progressive_refinement import (
    analyze_progressive_refinement,
)


def test_progressive_feasibility_analysis_applies_all_gates(tmp_path):
    config = {
        "steps": 40,
        "trainable_parameter_count": 147456,
        "core_parameters_unchanged": True,
        "core_parameter_sha256_before": "core",
        "core_parameter_sha256_after": "core",
        "trainable_parameters_changed": True,
        "trainable_parameter_sha256_before": "before",
        "trainable_parameter_sha256_after": "after",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with (tmp_path / "training_log.jsonl").open("w") as handle:
        for step in range(1, 41):
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": 1.0 if step <= 20 else 0.5,
                        "grad_norm": 1.0,
                    }
                )
                + "\n"
            )
    with (tmp_path / "results.jsonl").open("w") as handle:
        for index in range(4):
            handle.write(
                json.dumps(
                    {
                        "id": f"case-{index}",
                        "source_id": f"case-{index}",
                        "shifted_source_id": f"case-{(index + 1) % 4}",
                        "core_em": 0,
                        "core_f1": 0.4,
                        "full_em": 1,
                        "full_f1": 0.8,
                        "shifted_full_em": 0,
                        "shifted_full_f1": 0.1,
                        "no_state_em": 0,
                        "no_state_f1": 0.0,
                        "core_packet_bytes": 67642,
                        "refinement_packet_bytes": 67642,
                    }
                )
                + "\n"
            )

    result = analyze_progressive_refinement(
        tmp_path,
        expected_cases=4,
        expected_steps=40,
        bootstrap_replicates=100,
        seed=1,
    )

    assert result["comparisons"]["full_minus_core"]["mean_f1_delta"] == 0.4
    assert result["gates"]["authorize_larger_development_run"]
