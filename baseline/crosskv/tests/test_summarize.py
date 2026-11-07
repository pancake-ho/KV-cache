import pytest

from xmodel_kv.cli.summarize_hellaswag import summarize


def test_summarize_checks_coverage_and_computes_retention():
    rows = [
        {"index": 0, "label": 1, "standalone_prediction": 1, "transfer_prediction": 1},
        {"index": 1, "label": 2, "standalone_prediction": 2, "transfer_prediction": 0},
        {"index": 2, "label": 3, "standalone_prediction": 0, "transfer_prediction": 3},
        {"index": 3, "label": 0, "standalone_prediction": 0, "transfer_prediction": 0},
    ]
    result = summarize(rows, expected_documents=4)
    assert result["standalone_acc_norm"] == 0.75
    assert result["transfer_acc_norm"] == 0.75
    assert result["retention_percent"] == 100.0
    assert result["floor_normalized_retention_percent"] == 100.0

    with pytest.raises(ValueError, match="coverage mismatch"):
        summarize(rows[:-1], expected_documents=4)
