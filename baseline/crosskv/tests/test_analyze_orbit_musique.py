import json

import pytest

from xmodel_kv.cli.analyze_orbit_musique import analyze_musique_orbit


def _write_run(path, *, mode, f1, source_shift=0):
    path.mkdir()
    with (path / "results.jsonl").open("w") as handle:
        for index in range(3):
            row = {
                "index": index,
                "id": f"case-{index}",
                "source_id": f"case-{(index + source_shift) % 3}",
                "quant_bits": 4,
                "layer_pattern": "first_16",
                "final_gold": "final",
                "bridge_gold": "bridge",
                "student_exact": f1,
                "student_f1": f1,
                "bridge_exact": f1,
                "bridge_f1": f1,
                "packet_position_mode": mode,
                "packet_target_invariant": mode == "canonical",
                "base_packet_bytes": 67642,
                "delta_packet_bytes": 4252,
                "actual_packet_bytes": 71894,
                "incremental_packet_bytes": 4252,
            }
            handle.write(json.dumps(row) + "\n")


def test_musique_characterization_preserves_four_arm_pairing(tmp_path):
    values = {
        "parent_target": ("target", 0.4),
        "parent_canonical": ("canonical", 0.2),
        "orbit_target": ("target", 0.4),
        "orbit_canonical": ("canonical", 0.4),
    }
    paths = {}
    for name, (mode, f1) in values.items():
        path = tmp_path / name
        _write_run(path, mode=mode, f1=f1)
        paths[name] = path

    result = analyze_musique_orbit(
        paths,
        expected_cases=3,
        bootstrap_replicates=100,
        seed=7,
    )

    stage = result["endpoints"]["student"]
    assert stage["comparisons"]["canonical_safety_Oc_minus_Ot"][
        "mean_f1_delta"
    ] == 0
    assert stage["difference_in_differences"]["f1"]["mean"] == pytest.approx(0.2)
    assert result["wire_audit"]["pass"]


def test_musique_characterization_includes_shifted_source_control(tmp_path):
    paths = {}
    for name in ("parent_target", "orbit_target"):
        path = tmp_path / name
        _write_run(path, mode="target", f1=0.5)
        paths[name] = path
    for name in ("parent_canonical", "orbit_canonical"):
        path = tmp_path / name
        _write_run(path, mode="canonical", f1=0.5)
        paths[name] = path
    shifted = tmp_path / "shifted"
    _write_run(shifted, mode="canonical", f1=0.0, source_shift=1)

    result = analyze_musique_orbit(
        paths,
        expected_cases=3,
        bootstrap_replicates=100,
        seed=7,
        shifted_path=shifted,
    )

    comparison = result["endpoints"]["bridge"]["comparisons"][
        "source_causality_Oc_correct_minus_shifted"
    ]
    assert comparison["mean_f1_delta"] == 0.5
