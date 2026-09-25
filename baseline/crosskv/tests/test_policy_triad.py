from xmodel_kv.cli.policy_triad import _triad_group


def test_position_triad_has_nine_actions_and_distinct_third_policy():
    group = _triad_group(
        "position", "target_last", replicate=0, seed=3, record_count=9
    )
    assert len(group.record_ids) == 9
    assert group.target_record_id == group.record_ids[-1]
    assert group.third_record_id == group.record_ids[4]
    assert group.target_record_id != group.third_record_id
    assert group.cases["matched"].source_expected == group.cases["matched"].target_expected
    assert group.cases["third"].source_expected != group.cases["third"].target_expected
    enum = group.cases["third"].source_tools[0]["function"]["parameters"]["properties"][
        "record_id"
    ]["enum"]
    assert len(enum) == 9


def test_content_triad_randomizes_distinct_urgent_and_cheapest_positions():
    group = _triad_group(
        "content", "target_urgent", replicate=2, seed=7, record_count=9
    )
    assert group.metadata["urgent_position"] != group.metadata["cheapest_position"]
    assert group.target_record_id != group.third_record_id
    assert group.cases["third"].source_expected["arguments"]["record_id"] == group.third_record_id
    assert group.cases["matched"].target_expected["arguments"]["record_id"] == group.target_record_id
