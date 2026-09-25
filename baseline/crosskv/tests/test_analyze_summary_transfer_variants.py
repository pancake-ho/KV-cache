import json

import pytest

from xmodel_kv.cli.analyze_summary_transfer_variants import (
    compare_variants,
    load_protocol_rows,
)


def _row(case_id, value):
    return {
        "id": case_id,
        "protocol": "question_conditioned",
        "question": f"q-{case_id}",
        "gold_answers": ["yes"],
        "capsule_em": value,
        "capsule_f1": value,
        "capsule": f"answer-{value}",
        "source_answer": "source",
        "source_answer_em": 1,
        "source_answer_f1": 1,
    }


def test_variant_comparison_is_paired():
    candidate = {"a": _row("a", 1), "b": _row("b", 1)}
    reference = {"a": _row("a", 0), "b": _row("b", 0)}
    result = compare_variants(
        candidate,
        reference,
        candidate_arm="capsule",
        reference_arm="capsule",
        bootstrap_replicates=100,
        seed=1,
    )
    assert result["mean_f1_delta"] == 1
    assert result["candidate_only_correct"] == 2
    assert result["agreement"]["arm_output"]["different"] == 2
    assert result["agreement"]["source_answer"]["different"] == 0


def test_variant_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(_row("a", 1)) + "\n" for _ in range(2)))
    with pytest.raises(ValueError, match="duplicate"):
        load_protocol_rows([path], protocol="question_conditioned", expected_cases=1)
