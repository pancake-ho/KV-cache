from xmodel_kv.cli.probe_tail_task_gradient_alignment import analyze_gradient_rows


def test_gradient_row_analysis_reports_conflict_and_bridge_gap():
    rows = []
    for index in range(4):
        rows.append(
            {
                "coordinate_final_cosine": -0.5,
                "coordinate_bridge_cosine": 0.5,
                "operator_final_cosine": -0.25,
                "operator_bridge_cosine": 0.25,
                "final_bridge_cosine": 0.0,
            }
        )
    result = analyze_gradient_rows(rows, bootstrap_replicates=100, seed=1)

    assert result["alignments"]["coordinate_final"]["negative_fraction"] == 1
    assert result["bridge_minus_final_alignment"][
        "coordinate_bridge_minus_final"
    ]["mean"] == 1
