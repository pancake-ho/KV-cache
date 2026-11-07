import json

from xmodel_kv.cli.analyze_orbit_distillation import (
    analyze_orbit_distillation,
    audit_wire,
)


PROTOCOLS = ("state_readout", "question_conditioned")


def _write_training(path):
    path.mkdir()
    config = {
        "steps": 40,
        "trainable_parameter_count": 131072,
        "frozen_capsule_parameters_unchanged": True,
        "frozen_capsule_parameter_sha256_before": "frozen",
        "frozen_capsule_parameter_sha256_after": "frozen",
        "trainable_capsule_parameters_changed": True,
        "trainable_capsule_parameter_sha256_before": "before",
        "trainable_capsule_parameter_sha256_after": "after",
        "orbit_base_parameters_unchanged": True,
        "orbit_base_parameter_sha256_before": "base",
        "orbit_base_parameter_sha256_after": "base",
    }
    (path / "config.json").write_text(json.dumps(config))
    with (path / "training_log.jsonl").open("w") as handle:
        for step in range(1, 41):
            value = 0.2 if step <= 20 else 0.1
            row = {
                "step": step,
                "loss": 1.0,
                "stage_ce": 0.1,
                "bridge_ce": 0.1,
                "teacher_kl": 0.0,
                "preanswer_teacher_kl": 0.0,
                "orbit_kl": value,
                "grad_norm": 1.0,
            }
            handle.write(json.dumps(row) + "\n")


def _write_run(path, *, mode, capsule_f1, shifted_f1, delta_hash_suffix=""):
    path.mkdir()
    with (path / "results.jsonl").open("w") as handle:
        for index in range(4):
            for protocol in PROTOCOLS:
                row = {
                    "id": f"case-{index}",
                    "protocol": protocol,
                    "packet_position_mode": mode,
                    "key_codec": "cartesian",
                    "question": f"q-{index}",
                    "gold_answers": ["answer"],
                    "capsule": "answer",
                    "capsule_em": capsule_f1,
                    "capsule_f1": capsule_f1,
                    "shifted_capsule": "wrong",
                    "shifted_capsule_em": shifted_f1,
                    "shifted_capsule_f1": shifted_f1,
                }
                for arm in ("capsule", "shifted_capsule"):
                    row.update(
                        {
                            f"{arm}_base_packet_bytes": 67642,
                            f"{arm}_delta_packet_bytes": 4252,
                            f"{arm}_cold_packet_bytes": 71894,
                            f"{arm}_incremental_packet_bytes": 4252,
                        }
                    )
                    if mode == "canonical":
                        row[f"{arm}_base_packet_sha256"] = f"{arm}-base"
                        row[f"{arm}_delta_packet_sha256"] = (
                            f"{arm}-delta-{index}{delta_hash_suffix}"
                        )
                handle.write(json.dumps(row) + "\n")


def test_four_arm_analysis_applies_preregistered_gates(tmp_path):
    values = {
        "parent_target": ("target", 0.4, 0.1),
        "parent_canonical": ("canonical", 0.2, 0.1),
        "orbit_target": ("target", 0.4, 0.1),
        "orbit_canonical": ("canonical", 0.4, 0.1),
    }
    paths = {}
    for name, (mode, capsule, shifted) in values.items():
        path = tmp_path / name
        _write_run(path, mode=mode, capsule_f1=capsule, shifted_f1=shifted)
        paths[name] = path
    training = tmp_path / "training"
    _write_training(training)

    result = analyze_orbit_distillation(
        paths,
        training_dir=training,
        protocol="question_conditioned",
        expected_cases=4,
        expected_steps=40,
        bootstrap_replicates=100,
        seed=3,
    )

    assert result["training_audit"]["orbit_kl_fractional_reduction"] == 0.5
    assert result["difference_in_differences"]["f1"]["mean"] == 0.2
    assert result["gates"]["external_safety_pass"]
    assert result["gates"]["strong_mechanistic_repair_pass"]


def test_wire_audit_rejects_receiver_dependent_canonical_delta(tmp_path):
    paths = {}
    for name in ("parent_target", "orbit_target"):
        path = tmp_path / name
        _write_run(path, mode="target", capsule_f1=1, shifted_f1=0)
        paths[name] = path
    for name in ("parent_canonical", "orbit_canonical"):
        path = tmp_path / name
        _write_run(path, mode="canonical", capsule_f1=1, shifted_f1=0)
        paths[name] = path

    result_path = paths["orbit_canonical"] / "results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    rows[1]["capsule_delta_packet_sha256"] = "receiver-dependent"
    result_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    audit = audit_wire(paths, expected_cases=4)
    assert not audit["runs"]["orbit_canonical"]["pass"]
    assert not audit["pass"]
