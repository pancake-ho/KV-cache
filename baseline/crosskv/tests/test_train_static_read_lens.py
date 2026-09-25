import os
from pathlib import Path

import pytest

from xmodel_kv.cli.train_static_read_lens import (
    _diverse_cases,
    _parse_layers,
)


BFCL_ROOT = Path(
    os.environ.get(
        "XKV_BFCL_ROOT",
        "datasets/bfcl_official/berkeley-function-call-leaderboard/bfcl_eval/data",
    )
)


def test_parse_static_lens_layers():
    assert _parse_layers("18-20,23") == (18, 19, 20, 23)


@pytest.mark.skipif(
    not BFCL_ROOT.exists(), reason="set XKV_BFCL_ROOT to run BFCL integration tests"
)
def test_static_lens_training_uses_disjoint_background_pools():
    cases = _diverse_cases(
        BFCL_ROOT, train_cases=1, val_cases=1, background_records=8, seed=17
    )
    train_messages = {m["content"] for m in cases[0][2].history[:16] if m["role"] == "user"}
    validation_messages = {
        m["content"] for m in cases[1][2].history[:16] if m["role"] == "user"
    }
    assert train_messages.isdisjoint(validation_messages)
