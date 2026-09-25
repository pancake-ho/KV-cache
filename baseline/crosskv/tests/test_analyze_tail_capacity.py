import json

from xmodel_kv.cli.analyze_tail_capacity import analyze_tail_capacity


def _write_rows(path, cap, values):
    path.mkdir()
    with (path / "results.jsonl").open("w") as handle:
        for index, (answer, f1) in enumerate(values):
            row = {
                "id": f"case-{index}",
                "protocol": "question_conditioned",
                "question": f"q-{index}",
                "gold_answers": [answer],
                "generated_tail_em": f1,
                "generated_tail_f1": f1,
                "generated_tail_tokens": cap,
                "generated_tail_packet_bytes": cap * 100,
            }
            handle.write(json.dumps(row) + "\n")


def test_capacity_analysis_pairs_global_and_length_strata(tmp_path):
    answers = ["short answer", "one two three four five", " ".join(["x"] * 10)]
    paths = {}
    for cap, scores in ((4, (0, 0, 0)), (8, (0, 0.5, 0.5)), (16, (0, 1, 1))):
        path = tmp_path / f"cap{cap}"
        _write_rows(path, cap, list(zip(answers, scores, strict=True)))
        paths[cap] = path

    result = analyze_tail_capacity(
        paths,
        protocol="question_conditioned",
        expected_cases=3,
        bootstrap_replicates=100,
        seed=1,
    )

    assert result["comparisons"]["cap16_minus_cap4"]["mean_f1_delta"] == 2 / 3
    assert result["answer_length_strata"]["9_plus_words"][
        "cap16_minus_cap4"
    ]["mean_f1_delta"] == 1
    assert result["long_minus_short_interaction"]["mean"] == 1
