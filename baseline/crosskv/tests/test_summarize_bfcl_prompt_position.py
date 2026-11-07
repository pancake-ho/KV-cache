from xmodel_kv.cli.summarize_bfcl_prompt_position import summarize


def _call(name):
    return {"name": name, "arguments": {"job_id": name}, "valid_json": True}


def _result(name):
    return {"tool_call": _call(name), "expected_full_call_em": True}


def _row(front_reuse="source", tail_reuse="target"):
    return {
        "bfcl_id": "q0",
        "direction": "forward",
        "token_lengths": {"history": 5_000},
        "source_expected": {"name": "source", "arguments": {"job_id": "source"}},
        "target_expected": {"name": "target", "arguments": {"job_id": "target"}},
        "source_native": _result("source"),
        "front": {
            "native": _result("target"),
            "identity": _result("target"),
            "reuse": _result(front_reuse),
        },
        "tail": {
            "user": {
                "native": _result("target"),
                "identity": _result("target"),
                "reuse_drop_source": _result(tail_reuse),
                "reuse_keep_source": _result(tail_reuse),
            }
        },
    }


def test_prompt_position_summary_uses_common_native_valid_subset():
    result = summarize([_row()])
    common = result["common_strict"]["user"]
    assert common["rows"] == 1
    assert common["front"]["matches_source_native"]["value"] == 1.0
    assert common["tail_drop_source"]["target_expected_accuracy"]["value"] == 1.0
    assert common["paired_front_vs_tail_target_accuracy"][
        "front_wrong_tail_correct"
    ] == 1
