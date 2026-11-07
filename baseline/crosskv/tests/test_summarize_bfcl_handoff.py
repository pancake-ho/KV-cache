from xmodel_kv.cli.summarize_bfcl_handoff import summarize


def _row(source, target, hybrid, *, source_expected=None, target_expected=None):
    source_expected = source if source_expected is None else source_expected
    target_expected = target if target_expected is None else target_expected

    def call(value):
        return {"name": value, "arguments": {"x": value}, "valid_json": True}

    return {
        "bfcl_id": source + target,
        "direction": "forward",
        "source_expected": {"name": source_expected, "arguments": {"x": source_expected}},
        "target_expected": {"name": target_expected, "arguments": {"x": target_expected}},
        "source_native": {"tool_call": call(source)},
        "target_native": {"tool_call": call(target)},
        "identity": {"target_native_agreement": True},
        "hybrid": [{"tool_call": call(hybrid)}],
    }


def test_bfcl_handoff_summary_separates_end_to_end_and_causal_cells():
    result = summarize([_row("source", "target", "source")])
    overall = result["overall"]
    assert overall["paired_expected_accuracy"]["native_correct_reuse_wrong"] == 1
    assert overall["causal_contrast"]["rows"] == 1
    assert overall["causal_contrast"]["reuse_matches_source_native"]["value"] == 1.0
    assert overall["strict_both_policies_correct"]["rows"] == 1
    assert result["replay_curve"][0]["target_capable"][
        "reuse_matches_source_native"
    ]["value"] == 1.0


def test_bfcl_handoff_summary_excludes_incapable_target_from_contrast():
    result = summarize(
        [_row("source", "wrong", "source", target_expected="target")]
    )
    assert result["overall"]["target_capable"]["rows"] == 0
