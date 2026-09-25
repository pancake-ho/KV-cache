from xmodel_kv.cli.evaluate_diverse_refresh import (
    _pair,
    binding_only_refresh,
    compact_target_refresh,
    minimal_target_refresh,
)


def test_compact_target_refresh_contains_target_condition():
    text = compact_target_refresh(
        {
            "target_identity": "AGENT-B",
            "target_role": "executor",
            "target_policy": "tenant_scope",
            "target_policy_value": "TENANT-BETA",
        }
    )
    assert "AGENT-B" in text
    assert "tenant_scope" in text
    assert "TENANT-BETA" in text


def test_pair_counts_discordant_rows():
    rows = [
        {"native": {"target": {"compatible_em": True}}, "reuse": {"target": {"compatible_em": False}}},
        {"native": {"target": {"compatible_em": False}}, "reuse": {"target": {"compatible_em": True}}},
    ]
    assert _pair(rows, "native", "reuse") == {
        "both_correct": 0,
        "left_only_correct": 1,
        "right_only_correct": 1,
        "neither_correct": 0,
    }


def test_minimal_target_refresh_keeps_only_binding_condition_and_action_contract():
    text = minimal_target_refresh(
        {
            "target_policy": "tenant_scope",
            "target_policy_value": "TENANT-BETA",
        }
    )
    assert "tenant_scope=TENANT-BETA" in text
    assert "tool call" in text


def test_binding_only_refresh_is_canonical_policy_descriptor():
    assert binding_only_refresh(
        {"target_policy": "tenant_scope", "target_policy_value": "TENANT-BETA"}
    ) == "tenant_scope=TENANT-BETA"
