import json

import pytest

from xmodel_kv.cli.compare_hotpot_transfer_runs import load_rows


def test_load_rows_keys_by_id_and_protocol(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [
        {"index": 0, "id": "a", "protocol": "state_readout", "capsule_em": 1},
        {"index": 1, "id": "a", "protocol": "question_conditioned", "capsule_em": 0},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert set(load_rows(path)) == {
        ("a", "state_readout"),
        ("a", "question_conditioned"),
    }
    assert set(load_rows(path, min_index=1)) == {("a", "question_conditioned")}
    path.write_text("".join(json.dumps(rows[0]) + "\n" for _ in range(2)))
    with pytest.raises(ValueError, match="duplicate"):
        load_rows(path)
