import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from xmodel_kv.cli.bfcl_authorization_handoff import authorization_cases
from xmodel_kv.cli.bfcl_authorization_prompt_position import (
    NEUTRAL_RECEIVER_PROMPT,
    _tail_ids,
)
from xmodel_kv.policy_imprinting import build_chat_segments


BFCL_ROOT = Path(
    os.environ.get(
        "XKV_BFCL_ROOT",
        "datasets/bfcl_official/berkeley-function-call-leaderboard/bfcl_eval/data",
    )
)
TEST_MODEL = os.environ.get("XKV_TEST_MODEL")


@pytest.mark.skipif(
    not BFCL_ROOT.exists() or not TEST_MODEL,
    reason="set XKV_BFCL_ROOT and XKV_TEST_MODEL to run this integration test",
)
def test_tail_target_policy_is_appended_without_rewriting_long_history():
    case = authorization_cases(
        BFCL_ROOT,
        max_cases=1,
        seed=9,
        background_records=40,
        post_switch_trigger=True,
    )[0][0]["forward"]
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    base = build_chat_segments(
        tokenizer,
        system_prompt=NEUTRAL_RECEIVER_PROMPT,
        tools=case.target_tools,
        history=case.history,
    )
    for role in ("user", "system"):
        appended = build_chat_segments(
            tokenizer,
            system_prompt=NEUTRAL_RECEIVER_PROMPT,
            tools=case.target_tools,
            history=[*case.history, {"role": role, "content": case.target_prompt}],
        )
        tail_ids = _tail_ids(base, appended)
        assert len(tail_ids) > 50
        assert len(base.history_ids) > 4_000
