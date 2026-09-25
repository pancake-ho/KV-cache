import json
from types import SimpleNamespace

import pytest

from xmodel_kv.cli.evaluate_hotpot_summary_transfer import (
    format_source_user,
    load_slice,
    require_canonical_packet_invariance,
    summarize,
    wire_payload_bytes,
)


def test_format_source_user_preserves_question_and_context():
    text = format_source_user({"context": "Alpha evidence", "input": "Who?"})
    assert "<DOSSIER>\nAlpha evidence\n</DOSSIER>" in text
    assert "Question: Who?" in text


def test_load_slice_requires_hotpot_schema(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"input": f"q{i}", "context": f"c{i}", "answers": [f"a{i}"]})
            for i in range(3)
        )
        + "\n"
    )
    assert [index for index, _ in load_slice(path, offset=1, count=2)] == [1, 2]
    with pytest.raises(ValueError, match="requested slice"):
        load_slice(path, offset=2, count=2)


def test_wire_payload_matches_frozen_qwen_operating_point():
    config = SimpleNamespace(
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=4096,
        num_attention_heads=32,
    )
    assert wire_payload_bytes(config, tokens=4, active_layers=30, bits=4) == 126_720


def test_summarize_keeps_causal_controls_separate():
    config = SimpleNamespace(
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=4096,
        num_attention_heads=32,
    )
    rows = []
    for index, values in enumerate(((1, 0, 0, 1), (0, 1, 0, 1))):
        no_summary, capsule, shifted, tail = values
        row = {
            "protocol": "state_readout",
            "source_answer_em": 1,
            "source_answer_f1": 1,
            "source_tokens": 100,
            "source_prefill_ms": 10,
            "capsule_emission_ms": 2,
            "source_decode_ms": 3,
            "generated_tail_tokens": 5,
            "no_summary_generation_ms": 1,
        }
        for arm, score in zip(
            ("no_summary", "capsule", "shifted_capsule", "generated_tail"),
            (no_summary, capsule, shifted, tail),
            strict=True,
        ):
            row[f"{arm}_em"] = score
            row[f"{arm}_f1"] = score
            if arm != "no_summary":
                row[f"{arm}_generation_ms"] = 1
                row[f"{arm}_prepare_ms"] = 1
        rows.append(row)
    output = summarize(
        rows,
        config={"quant_bits": 4},
        model_config=config,
        slots=4,
        active_layers=30,
        generated_tail_active_layers=36,
        training_config={},
    )["by_protocol"]["state_readout"]
    assert output["arms"]["capsule"]["em"] == 0.5
    assert output["arms"]["shifted_capsule"]["em"] == 0
    assert output["arms"]["capsule"]["wire_payload_bytes"] == 126_720
    assert output["arms"]["generated_tail"]["mean_wire_payload_bytes"] == 190_080


def test_canonical_packet_audit_requires_identical_hashes_across_protocols():
    rows = [
        {
            "id": "case-1",
            "protocol": protocol,
            "capsule_base_packet_sha256": "base",
            "capsule_delta_packet_sha256": "delta",
            "shifted_capsule_base_packet_sha256": "shift-base",
            "shifted_capsule_delta_packet_sha256": "shift-delta",
        }
        for protocol in ("state_readout", "question_conditioned")
    ]

    assert require_canonical_packet_invariance(
        rows, protocols=("state_readout", "question_conditioned")
    )

    rows[1]["capsule_delta_packet_sha256"] = "different"
    with pytest.raises(RuntimeError, match="depends on receiver protocol"):
        require_canonical_packet_invariance(
            rows, protocols=("state_readout", "question_conditioned")
        )
