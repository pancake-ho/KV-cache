from xmodel_kv.cli.analyze_receiver_identity_lens import analyze_identity_groups


def rows(final, bridge, *, with_controls=False):
    result = {}
    for index in range(32):
        row = {"lens4_student_f1": final, "lens4_bridge_f1": bridge}
        if with_controls:
            row.update(
                {
                    "base_student_f1": 0.10,
                    "base_bridge_f1": 0.20,
                    "hard4_student_f1": 0.12,
                    "hard4_bridge_f1": 0.20,
                }
            )
        result[str(index)] = row
    return result


def integrity(value=True):
    return {"checks": {"synthetic_check": value}}


def test_identity_analyzer_authorizes_strong_specific_causal_lens():
    groups = {
        "identity_correct": rows(0.40, 0.30, with_controls=True),
        "identity_shift": rows(0.10, 0.05),
        "shared_correct": rows(0.20, 0.25),
    }
    result = analyze_identity_groups(
        groups, integrity=integrity(), bootstrap_replicates=100, seed=1
    )
    assert all(result["gates"].values())
    assert result["untouched_confirmation_authorized"] is True
    assert result["conditional_writer_authorized"] is False
    assert result["eviction_test_authorized"] is False


def test_identity_analyzer_rejects_no_gain_over_shared_and_bad_integrity():
    groups = {
        "identity_correct": rows(0.25, 0.30, with_controls=True),
        "identity_shift": rows(0.10, 0.05),
        "shared_correct": rows(0.24, 0.25),
    }
    result = analyze_identity_groups(
        groups, integrity=integrity(False), bootstrap_replicates=100, seed=2
    )
    assert result["gates"]["identity_final_vs_shared"] is False
    assert result["gates"]["bank_and_base_integrity"] is False
    assert result["untouched_confirmation_authorized"] is False
