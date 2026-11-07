import os
from pathlib import Path

import pytest

from xmodel_kv.cli.bfcl_authorization_handoff import (
    POST_SWITCH_TRIGGER,
    authorization_cases,
)


BFCL_ROOT = Path(
    os.environ.get(
        "XKV_BFCL_ROOT",
        "datasets/bfcl_official/berkeley-function-call-leaderboard/bfcl_eval/data",
    )
)
requires_bfcl = pytest.mark.skipif(
    not BFCL_ROOT.exists(), reason="set XKV_BFCL_ROOT to run BFCL integration tests"
)


@requires_bfcl
def test_authorization_cases_have_eight_real_jobs_and_mirrored_tenants():
    cases, metadata = authorization_cases(BFCL_ROOT, max_cases=1, seed=9)[0]
    forward = cases["forward"]
    reverse = cases["reverse"]
    assert metadata["job_count"] == 8
    assert metadata["chance_level"] == 0.125
    assert len(forward.history) == 17
    assert forward.source_expected == reverse.target_expected
    assert forward.target_expected == reverse.source_expected
    assert forward.source_expected != forward.target_expected
    assert "not yet executed" in forward.history[-1]["content"]


@requires_bfcl
def test_authorization_post_switch_trigger_is_a_separate_final_user_message():
    cases, _ = authorization_cases(
        BFCL_ROOT, max_cases=1, seed=9, post_switch_trigger=True
    )[0]
    forward = cases["forward"]
    assert len(forward.history) == 18
    assert forward.history[-2]["role"] == "assistant"
    assert "not yet executed" in forward.history[-2]["content"]
    assert forward.history[-1] == {"role": "user", "content": POST_SWITCH_TRIGGER}
