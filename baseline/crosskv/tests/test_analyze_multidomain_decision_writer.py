from xmodel_kv.cli.analyze_multidomain_decision_writer import analyze_groups


def _rows(final, bridge):
    return {
        str(index): {"student_f1": final, "bridge_f1": bridge}
        for index in range(8)
    }


def test_multidomain_analyzer_applies_all_frozen_gates():
    groups = {
        "parent_dev": _rows(0.5, 0.2),
        "parent_dev_shift": _rows(0.1, 0.0),
        "candidate_dev": _rows(0.6, 0.4),
        "candidate_dev_shift": _rows(0.1, 0.0),
        "parent_musique": _rows(0.8, 0.8),
        "candidate_musique": _rows(0.8, 0.8),
    }

    result = analyze_groups(groups, bootstrap_replicates=100, seed=1)

    assert all(result["gates"].values())
    assert result["external_evaluation_authorized"] is True


def test_multidomain_analyzer_rejects_bridge_forgetting():
    groups = {
        "parent_dev": _rows(0.5, 0.4),
        "parent_dev_shift": _rows(0.1, 0.0),
        "candidate_dev": _rows(0.6, 0.2),
        "candidate_dev_shift": _rows(0.1, 0.0),
        "parent_musique": _rows(0.8, 0.8),
        "candidate_musique": _rows(0.8, 0.2),
    }

    result = analyze_groups(groups, bootstrap_replicates=100, seed=1)

    assert result["gates"]["dev_bridge_improvement"] is False
    assert result["gates"]["musique_safety"] is False
    assert result["external_evaluation_authorized"] is False
