import pytest

from xmodel_kv.cli.analyze_paired_transfer import (
    _load_rows,
    _two_sided_binomial_p,
    cluster_bootstrap_mean_ci,
    paired_summary,
)
import json


def test_exact_binomial_p_handles_one_sided_and_tied_discordance():
    assert _two_sided_binomial_p(0, 3) == 0.25
    assert _two_sided_binomial_p(2, 4) == 1.0
    assert _two_sided_binomial_p(0, 0) == 1.0


def test_paired_summary_preserves_pairing_and_bootstrap_delta():
    summary = paired_summary(
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1.0, 0.5, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        bootstrap_replicates=1000,
        seed=7,
    )

    assert summary["candidate_accuracy"] == 0.5
    assert summary["reference_accuracy"] == 0.5
    assert summary["candidate_only_correct"] == 1
    assert summary["reference_only_correct"] == 1
    assert summary["both_correct"] == 1
    assert summary["neither_correct"] == 1
    assert summary["oracle_union_accuracy"] == 0.75
    assert summary["mean_f1_delta"] == pytest.approx(0.125)


def test_cluster_bootstrap_resamples_whole_groups():
    interval, clusters = cluster_bootstrap_mean_ci(
        [1.0, 1.0, -1.0],
        ["same", "same", "other"],
        bootstrap_replicates=2000,
        seed=3,
    )

    assert clusters == 2
    assert interval[0] <= -1.0
    assert interval[1] >= 1.0


def test_load_rows_applies_inclusive_holdout_index_floor(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {
            "index": index,
            "id": f"case-{index}",
            "quant_bits": 4,
            "layer_pattern": "first_16",
        }
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selected = _load_rows(path, 4, layer_pattern="first_16", min_index=1)
    assert set(selected) == {"case-1", "case-2"}
