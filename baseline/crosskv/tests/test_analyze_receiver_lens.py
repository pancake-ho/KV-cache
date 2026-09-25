from xmodel_kv.cli.analyze_receiver_lens import analyze_groups


def _rows(base_final, base_bridge, hard_final, hard_bridge, lens_final, lens_bridge):
    return {
        str(index): {
            "base_student_f1": base_final,
            "base_bridge_f1": base_bridge,
            "hard4_student_f1": hard_final,
            "hard4_bridge_f1": hard_bridge,
            "lens4_student_f1": lens_final,
            "lens4_bridge_f1": lens_bridge,
        }
        for index in range(16)
    }


def test_receiver_lens_analyzer_authorizes_strong_safe_lens():
    groups = {
        "dev_correct": _rows(0.2, 0.4, 0.21, 0.39, 0.4, 0.5),
        "dev_shift": _rows(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        "musique_correct": _rows(0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
    }

    result = analyze_groups(groups, bootstrap_replicates=100, seed=1)

    assert all(result["gates"].values())
    assert result["eviction_test_authorized"] is True
    assert result["external_evaluation_authorized"] is False


def test_receiver_lens_analyzer_rejects_semantic_forgetting():
    groups = {
        "dev_correct": _rows(0.2, 0.5, 0.2, 0.5, 0.4, 0.2),
        "dev_shift": _rows(0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        "musique_correct": _rows(0.8, 0.8, 0.8, 0.8, 0.8, 0.2),
    }

    result = analyze_groups(groups, bootstrap_replicates=100, seed=1)

    assert result["gates"]["dev_bridge_safety"] is False
    assert result["gates"]["musique_safety"] is False
    assert result["eviction_test_authorized"] is False
