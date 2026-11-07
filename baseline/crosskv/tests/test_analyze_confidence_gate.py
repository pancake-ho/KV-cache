from xmodel_kv.cli.analyze_confidence_gate import evaluate_gate, select_threshold


def _row(score, exact):
    return {
        "source_answer_mean_logprob": score,
        "student_exact": float(exact),
        "student_f1": float(exact),
    }


def test_confidence_gate_selects_tail_for_high_scores_and_generalizes_rule():
    writer = {"a": _row(0, 1), "b": _row(0, 0), "c": _row(0, 0)}
    tail = {"a": _row(-3, 0), "b": _row(-0.1, 1), "c": _row(-0.2, 1)}

    threshold, calibration = select_threshold(
        writer, tail, feature="source_answer_mean_logprob"
    )
    summary = evaluate_gate(
        writer,
        tail,
        feature="source_answer_mean_logprob",
        threshold=threshold,
        bootstrap_replicates=100,
        seed=1,
    )

    assert -3 < threshold < -0.2
    assert calibration["gate_accuracy"] == 1.0
    assert summary["candidate_accuracy"] == 1.0
    assert summary["reference_accuracy"] == 2 / 3
