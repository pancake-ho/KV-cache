from xmodel_kv.cli.screen_diverse_handoff import score_bfcl_call
from xmodel_kv.policy_imprinting import ToolCall


def test_bfcl_compatible_score_allows_legal_optional_arguments():
    call = ToolCall("area", {"base": 10, "height": 5, "unit": "m"}, True)
    score = score_bfcl_call(
        call,
        expected={"name": "area", "arguments": {"base": 10, "height": 5}},
        argument_options={"base": [10], "height": [5], "unit": ["", "m"]},
    )
    assert score["compatible_em"]
    assert not score["strict_arguments_em"]


def test_bfcl_compatible_score_rejects_wrong_required_argument():
    call = ToolCall("area", {"base": 11, "height": 5}, True)
    score = score_bfcl_call(
        call,
        expected={"name": "area", "arguments": {"base": 10, "height": 5}},
        argument_options={"base": [10], "height": [5]},
    )
    assert not score["compatible_em"]
