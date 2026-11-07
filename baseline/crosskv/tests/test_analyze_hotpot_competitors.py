import pytest

from xmodel_kv.cli.analyze_hotpot_competitors import analyze, join_rows


def _fixtures():
    base, packet, cold, blend = {}, {}, {}, {}
    for index in range(2):
        case_id = f"case-{index}"
        exact = float(index == 0)
        base[case_id] = {
            "id": case_id,
            "index": index,
            "question": f"question {index}",
            "gold_answers": [f"gold {index}"],
            "capsule_em": exact,
            "capsule_f1": 0.25 + 0.5 * exact,
            "generated_tail_em": exact,
            "generated_tail_f1": 0.3 + 0.5 * exact,
            "plaintext_source_em": exact,
            "plaintext_source_f1": 0.4 + 0.5 * exact,
            "full_recompute_em": 1.0,
            "full_recompute_f1": 1.0,
            "capsule_prepare_ms": 2.0,
            "capsule_generation_ms": 8.0,
            "capsule_emission_ms": 3.0,
        }
        packet_row = {
            "id": case_id,
            "index": index,
            "question": f"question {index}",
            "gold_answers": [f"gold {index}"],
            "document_tokens": 100 + index,
            "raw_document_kv_bf16_bytes": 1000 + index,
            "packet_kv_bf16_bytes": 1100 + index,
        }
        for arm, arm_exact, f1 in (
            ("full_recompute", 1.0, 1.0),
            ("no_recompute", 0.0, 0.2),
            ("kv_packet", exact, 0.4 + 0.5 * exact),
        ):
            packet_row.update(
                {
                    arm: f"{arm}-{index}",
                    f"{arm}_em": arm_exact,
                    f"{arm}_f1": f1,
                    f"{arm}_ttft_ms": 10.0 + index,
                    f"{arm}_flops": 100,
                    f"{arm}_online_total_flops": 200,
                    f"{arm}_peak_gpu_delta_bytes": 300,
                }
            )
        packet[case_id] = packet_row
        shared = {
            "id": case_id,
            "index": index,
            "question": f"question {index}",
            "gold_answers": [f"gold {index}"],
            "prompt_tokens": 120 + index,
            "document_tokens": 100 + index,
            "document_kv_bf16_bytes": 1000 + index,
        }
        cold[case_id] = {
            **shared,
            "cold_full": f"cold-{index}",
            "cold_full_em": 1.0,
            "cold_full_f1": 1.0,
            "cold_full_wall_ms": 20.0,
            "cold_full_ttft_ms": 15.0,
        }
        blend[case_id] = {
            **shared,
            "cacheblend": f"blend-{index}",
            "cacheblend_em": exact,
            "cacheblend_f1": 0.5 + 0.5 * exact,
            "cacheblend_wall_ms": 12.0,
            "cacheblend_ttft_ms": 9.0,
            "cacheblend_cached_tokens": 110 + index,
            "cacheblend_uncached_tokens": 10,
            "cacheblend_cached_fraction": (110 + index) / (120 + index),
            "population_wall_ms": 11.0,
            "population_prompt_tokens": 115 + index,
        }
    return base, packet, cold, blend


def test_join_and_analyze_competitor_matrix():
    base, packet, cold, blend = _fixtures()
    rows = join_rows(base, packet, cold, blend, expected_cases=2)
    summary = analyze(
        rows,
        bootstrap_replicates=100,
        seed=7,
        capsule_payload_bytes=10,
    )
    assert len(rows) == 2
    assert summary["cases"] == 2
    assert summary["quality"]["kvpacket"]["em"] == 0.5
    assert summary["quality"]["cacheblend_cold"]["f1"] == 1.0
    assert summary["cacheblend"]["zero_hit_cases"] == 0
    assert summary["transport"]["kvpacket_to_capsule_payload_ratio"] == 110.05


def test_join_rejects_cold_blend_prompt_mismatch():
    base, packet, cold, blend = _fixtures()
    blend["case-0"]["prompt_tokens"] += 1
    with pytest.raises(ValueError, match="cold/blend mismatch"):
        join_rows(base, packet, cold, blend, expected_cases=2)
