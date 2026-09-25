import pytest

from xmodel_kv.cli.analyze_hotpot_baseline_matrix import analyze, join_rows


def _capsule(index, case_id):
    row = {
        "index": index,
        "id": case_id,
        "question": f"q{index}",
        "gold_answers": ["a"],
        "source_answer": "a",
        "source_answer_em": 1.0,
        "source_answer_f1": 1.0,
        "source_tokens": 100,
        "source_decode_ms": 3.0,
        "capsule_emission_ms": 1.0,
        "capsule_prepare_ms": 1.0,
        "capsule_generation_ms": 1.0,
        "generated_tail_prepare_ms": 1.0,
        "generated_tail_generation_ms": 1.0,
        "generated_tail_tokens": 2,
    }
    for arm, value in (
        ("no_summary", 0.0),
        ("capsule", 0.5),
        ("shifted_capsule", 0.0),
        ("generated_tail", 1.0),
    ):
        row[f"{arm}_em"] = value
        row[f"{arm}_f1"] = value
    return row


def _baseline(index, case_id):
    row = {
        "index": index,
        "id": case_id,
        "question": f"q{index}",
        "gold_answers": ["a"],
        "source_answer": "a",
    }
    for arm, value in (
        ("plaintext_source", 1.0),
        ("plaintext_shifted", 0.0),
        ("plaintext_gold", 1.0),
        ("full_recompute", 1.0),
    ):
        row.update(
            {
                arm: "a",
                f"{arm}_em": value,
                f"{arm}_f1": value,
                f"{arm}_prompt_tokens": 10,
                f"{arm}_prefill_ms": 2.0,
                f"{arm}_generation_ms": 3.0,
            }
        )
        if arm != "full_recompute":
            row[f"{arm}_handoff_tokens"] = 1
            row[f"{arm}_handoff_utf8_bytes"] = 1
    return row


def test_join_and_analyze_matrix():
    capsule = {"a": _capsule(0, "a"), "b": _capsule(1, "b")}
    baseline = {"a": _baseline(0, "a"), "b": _baseline(1, "b")}
    rows = join_rows(capsule, baseline, expected_cases=2)
    result = analyze(
        rows,
        bootstrap_replicates=20,
        seed=7,
        capsule_payload_bytes=64,
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
    )
    assert result["quality"]["source_full_kv"]["em"] == 1.0
    assert result["transport"]["full_kv_mean_tokens"] == 100
    assert result["timing_ms"]["capsule_post_source_prefill_ms"] == 3.0


def test_join_rejects_different_ids():
    with pytest.raises(ValueError, match="ID sets differ"):
        join_rows({"a": _capsule(0, "a")}, {"b": _baseline(0, "b")}, expected_cases=1)
