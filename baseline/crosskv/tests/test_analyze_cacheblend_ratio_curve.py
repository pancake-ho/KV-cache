from xmodel_kv.cli.analyze_cacheblend_ratio_curve import (
    analyze_curve,
    join_curve,
    max_ngram_count,
)


def _cold(index):
    return {
        "index": index,
        "id": str(index),
        "question": f"q{index}",
        "gold_answers": ["yes"],
        "prompt_tokens": 100,
        "document_tokens": 80,
        "document_kv_bf16_bytes": 1000,
        "cold_full": "Yes",
        "cold_full_em": 1,
        "cold_full_f1": 1,
        "cold_full_wall_ms": 10,
        "cold_full_ttft_ms": 8,
    }


def _blend(index, ratio, *, correct):
    row = _cold(index)
    row.update(
        {
            "recompute_ratio": ratio,
            "cacheblend": "Yes" if correct else "no no no no no",
            "cacheblend_em": int(correct),
            "cacheblend_f1": int(correct),
            "cacheblend_wall_ms": 5,
            "cacheblend_ttft_ms": 4,
            "cacheblend_cached_tokens": 90,
            "cacheblend_cached_fraction": 0.9,
        }
    )
    return row


def test_curve_join_and_analysis_preserve_fixed_ratios():
    cold = {str(i): _cold(i) for i in range(2)}
    ratios = {
        "ratio_015": {str(i): _blend(i, 0.15, correct=False) for i in range(2)},
        "ratio_050": {str(i): _blend(i, 0.50, correct=bool(i)) for i in range(2)},
        "ratio_100": {str(i): _blend(i, 1.0, correct=True) for i in range(2)},
    }
    result = analyze_curve(
        join_curve(cold, ratios, expected_cases=2),
        bootstrap_replicates=100,
        seed=1,
    )
    assert result["quality_and_timing"]["ratio_015"]["f1"] == 0
    assert result["quality_and_timing"]["ratio_100"]["f1"] == 1
    assert result["length_strata"]["le_4096"]["cases"] == 2
    assert result["output_stability"]["ratio_100"][
        "normalized_agreement_with_cold"
    ] == 1


def test_max_ngram_count_detects_repetition():
    assert max_ngram_count("x y z x y z x y z", n=3) == 3
    assert max_ngram_count("short", n=3) == 0
