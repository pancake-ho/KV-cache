from xmodel_kv.cli.probe_tail_receiver_effect import analyze_receiver_effect_rows


def test_receiver_effect_analysis_reports_paired_reversal():
    rows = []
    for index in range(4):
        rows.append(
            {
                "arms": {
                    "state4": {
                        "effect_total": 2.0,
                        "effect_direction": 1.0,
                        "effect_log_norm": 10.0,
                        "final_f1": 0.5,
                    },
                    "ce8": {
                        "effect_total": 1.0,
                        "effect_direction": 0.8,
                        "effect_log_norm": 2.0,
                        "final_f1": 1.0,
                    },
                    "state8": {
                        "effect_total": 1.5,
                        "effect_direction": 0.9,
                        "effect_log_norm": 6.0,
                        "final_f1": 0.0,
                    },
                }
            }
        )

    result = analyze_receiver_effect_rows(rows, bootstrap_replicates=100, seed=1)

    delta = result["comparisons"]["state8_minus_ce8"]
    assert delta["effect_total"]["mean_delta"] == 0.5
    assert delta["effect_total"]["positive_fraction"] == 1
    assert delta["final_f1"]["mean_delta"] == -1
