from xmodel_kv.cli.summarize_policy_tail import _summarize


def test_empty_eligible_tail_cell_is_reported():
    row = {
        "triad": {"family": "content", "direction": "target_urgent", "layout": "clean"},
        "validity": {
            "front_target_correct": False,
            "front_source_correct": True,
            "tail_native_correct": True,
        },
        "token_lengths": {"source_receiver_prefix_shift": 0},
    }
    summary = _summarize(("content", "target_urgent", "clean"), [row])
    assert summary["rows"] == 1
    assert summary["eligible"] == 0
