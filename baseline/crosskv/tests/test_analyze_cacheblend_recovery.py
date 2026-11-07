from xmodel_kv.cli.analyze_cacheblend_recovery import (
    analyze_recovery,
    join_recovery_rows,
)


def _cold(index: int, text: str, exact: float, f1: float):
    return {
        "index": index,
        "id": str(index),
        "question": f"q{index}",
        "gold_answers": ["answer"],
        "prompt_tokens": 100,
        "document_tokens": 80,
        "document_kv_bf16_bytes": 1000,
        "cold_full": text,
        "cold_full_em": exact,
        "cold_full_f1": f1,
    }


def _blend(index: int, text: str, exact: float, f1: float):
    row = _cold(index, text, exact, f1)
    row.update(
        {
            "cacheblend": text,
            "cacheblend_em": exact,
            "cacheblend_f1": f1,
            "recompute_ratio": 1.0,
            "blend_check_layer": 1,
        }
    )
    return row


def test_exact_recovery_passes_equivalence_gate():
    cold = {"0": _cold(0, "The Answer.", 1, 1), "1": _cold(1, "No", 0, 0)}
    blend = {"0": _blend(0, "Answer", 1, 1), "1": _blend(1, "No", 0, 0)}
    rows = join_recovery_rows(cold, blend, expected_cases=2)
    result = analyze_recovery(
        rows,
        equivalence_margin=0.05,
        minimum_answer_agreement=0.8,
        bootstrap_replicates=100,
        seed=1,
    )
    assert result["strict_exact_recovery"]
    assert result["recovery_gate_pass"]
    assert result["raw_answer_agreement"] == 0.5


def test_large_quality_loss_fails_recovery_gate():
    cold = {str(i): _cold(i, "answer", 1, 1) for i in range(4)}
    blend = {str(i): _blend(i, "wrong", 0, 0) for i in range(4)}
    result = analyze_recovery(
        join_recovery_rows(cold, blend, expected_cases=4),
        equivalence_margin=0.05,
        minimum_answer_agreement=0.8,
        bootstrap_replicates=100,
        seed=1,
    )
    assert not result["metric_equivalence"]
    assert not result["recovery_gate_pass"]
