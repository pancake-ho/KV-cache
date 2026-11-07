import json

import pytest

from xmodel_kv.cli.analyze_hotpot_summary_transfer import analyze, load_and_validate


def _row(index, protocol):
    capsule = int(index == 0)
    shifted = int(index == 1)
    row = {
        "id": f"id-{index}",
        "index": index,
        "protocol": protocol,
        "gold_answers": [f"answer-{index}"],
        "source_answer": f"answer-{index}",
        "source_answer_em": 1,
        "source_answer_f1": 1,
        "source_tokens": 100,
        "capsule_tokens": 4,
        "generated_tail_tokens": 5,
        "generated_tail": f"answer-{index}",
    }
    for arm, score in (
        ("no_summary", 0),
        ("capsule", capsule),
        ("shifted_capsule", shifted),
        ("generated_tail", 1),
    ):
        row[f"{arm}_em"] = score
        row[f"{arm}_f1"] = score
    return row


def test_load_and_validate_requires_complete_paired_protocols(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [
        _row(index, protocol)
        for index in range(2)
        for protocol in ("state_readout", "question_conditioned")
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = load_and_validate([path], expected_cases=2)
    assert len(loaded) == 4
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="protocol"):
        load_and_validate([path], expected_cases=2)


def test_analyze_reports_frozen_pairwise_comparisons():
    rows = [
        _row(index, protocol)
        for index in range(2)
        for protocol in ("state_readout", "question_conditioned")
    ]
    summary = analyze(
        rows,
        input_paths=["synthetic"],
        bootstrap_replicates=100,
        seed=1,
    )["by_protocol"]["state_readout"]
    assert summary["cases"] == 2
    assert summary["arms"]["generated_tail"]["em"] == 1
    paired = summary["comparisons"]["capsule_vs_shifted_capsule"]
    assert paired["candidate_only_correct"] == 1
    assert paired["reference_only_correct"] == 1
