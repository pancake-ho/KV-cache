import pytest

from xmodel_kv.cli.policy_tail import _suffix, _tail_message


def test_tail_message_places_target_policy_in_final_user_message():
    message = _tail_message("Content policy: URGENT.")
    assert message["role"] == "user"
    assert message["content"].endswith("Content policy: URGENT.")
    assert "overrides" in message["content"]


def test_plain_tail_message_does_not_add_override_claim():
    message = _tail_message("Content policy: URGENT.", style="plain")
    assert message["content"] == "The batch is complete.\nContent policy: URGENT."


def test_suffix_extracts_appended_tokens():
    assert _suffix((1, 2, 3, 4), (1, 2)) == (3, 4)
    with pytest.raises(ValueError):
        _suffix((1, 9, 3), (1, 2))
