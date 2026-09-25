import json

import pytest

from xmodel_kv.cli.evaluate_hotpot_plaintext_baselines import (
    format_full_recompute_user,
    format_plaintext_user,
    load_source_rows,
    summarize,
)


def test_load_source_rows_selects_question_protocol(tmp_path):
    rows = [
        {"id": "a", "protocol": "state_readout"},
        {
            "id": "a",
            "protocol": "question_conditioned",
            "question": "q",
            "source_answer": "answer",
            "source_answer_em": 1.0,
            "source_answer_f1": 1.0,
            "shifted_capsule_source_id": "b",
        },
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = load_source_rows(path)
    assert list(selected) == ["a"]
    assert selected["a"]["source_answer"] == "answer"


def test_plaintext_and_full_prompts_keep_information_separate():
    plain = format_plaintext_user("compact answer", "which one?")
    full = format_full_recompute_user(
        {"context": "long dossier", "input": "which one?"}
    )
    assert "compact answer" in plain and "long dossier" not in plain
    assert "long dossier" in full and "compact answer" not in full


def test_summarize_reports_cost_and_quality():
    row = {}
    for position, arm in enumerate(
        (
            "plaintext_source",
            "plaintext_shifted",
            "plaintext_gold",
            "full_recompute",
        )
    ):
        row.update(
            {
                f"{arm}_em": float(position == 0),
                f"{arm}_f1": 0.25 * position,
                f"{arm}_prompt_tokens": 10 + position,
                f"{arm}_prefill_ms": 1 + position,
                f"{arm}_generation_ms": 2 + position,
            }
        )
        if arm != "full_recompute":
            row[f"{arm}_handoff_tokens"] = 3 + position
            row[f"{arm}_handoff_utf8_bytes"] = 4 + position
    result = summarize([row])
    assert result["arms"]["plaintext_source"]["em"] == 1.0
    assert result["arms"]["plaintext_gold"]["mean_handoff_tokens"] == 5.0
    assert "mean_handoff_tokens" not in result["arms"]["full_recompute"]


def test_load_source_rows_rejects_missing_protocol(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"id": "a", "protocol": "state_readout"}) + "\n")
    with pytest.raises(ValueError, match="no question-conditioned"):
        load_source_rows(path)
