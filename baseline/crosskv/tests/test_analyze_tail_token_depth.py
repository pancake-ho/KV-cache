import json

import pytest

from xmodel_kv.cli.analyze_tail_token_depth import analyze_tail_token_depth


def _write_rows(path, *, cap, score, packet_bytes):
    path.mkdir()
    answers = ["short answer", "one two three four five", " ".join(["x"] * 10)]
    with (path / "results.jsonl").open("w") as handle:
        for index, answer in enumerate(answers):
            handle.write(
                json.dumps(
                    {
                        "id": f"case-{index}",
                        "protocol": "question_conditioned",
                        "question": f"q-{index}",
                        "gold_answers": [answer],
                        "generated_tail_em": score,
                        "generated_tail_f1": score,
                        "generated_tail_tokens": cap,
                        "generated_tail_packet_bytes": packet_bytes,
                        "no_summary_em": 0.0,
                        "no_summary_f1": 0.0,
                    }
                )
                + "\n"
            )


def test_token_depth_factorial_applies_frontier_gates(tmp_path):
    first16 = {}
    all32 = {}
    for cap, first_score, all_score, packet_bytes in (
        (4, 0.2, 0.3, 400),
        (8, 0.3, 0.5, 800),
        (16, 0.35, 0.55, 1600),
    ):
        first_path = tmp_path / f"first16-cap{cap}"
        all_path = tmp_path / f"all32-cap{cap}"
        _write_rows(
            first_path, cap=cap, score=first_score, packet_bytes=packet_bytes / 2
        )
        _write_rows(all_path, cap=cap, score=all_score, packet_bytes=packet_bytes)
        first16[cap] = first_path
        all32[cap] = all_path

    cap24 = tmp_path / "all32-cap24"
    _write_rows(cap24, cap=24, score=0.5, packet_bytes=1600)
    result = analyze_tail_token_depth(
        first16,
        all32,
        all_cap24=cap24,
        protocol="question_conditioned",
        expected_cases=3,
        bootstrap_replicates=100,
        seed=1,
    )

    assert result["all32_minus_first16"]["8"]["mean_f1_delta"] == pytest.approx(0.2)
    assert result["bytes"]["ratio"] == 0.5
    assert result["gates"]["paper_frontier_authorized"]
