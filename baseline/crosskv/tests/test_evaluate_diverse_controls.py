from xmodel_kv.cli.evaluate_diverse_controls import (
    _appended_history_ids,
    _paired_counts,
)
from xmodel_kv.policy_imprinting import ChatSegments


def _result(correct):
    return {"target": {"compatible_em": correct}}


def test_appended_history_ids_preserves_prior_tokens():
    base = ChatSegments((1,), (2, 3), (9,))
    appended = ChatSegments((1,), (2, 3, 4, 5), (9,))
    assert _appended_history_ids(base, appended) == (4, 5)


def test_paired_counts_are_exact():
    rows = [
        {
            "front_full_reprefill": _result(True),
            "tail_full_reprefill": _result(False),
            "tail_kv_reuse": _result(False),
        },
        {
            "front_full_reprefill": _result(True),
            "tail_full_reprefill": _result(True),
            "tail_kv_reuse": _result(False),
        },
    ]
    paired = _paired_counts(rows)
    assert paired[
        "front_full_reprefill_vs_tail_full_reprefill"
    ] == {
        "both_correct": 1,
        "left_only_correct": 1,
        "right_only_correct": 0,
        "neither_correct": 0,
    }
