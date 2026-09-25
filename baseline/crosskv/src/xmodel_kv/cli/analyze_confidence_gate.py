from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .analyze_paired_transfer import _load_rows, paired_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate a generation-confidence gate between writer and Tail-KV."
    )
    parser.add_argument("--calibration-writer", required=True)
    parser.add_argument("--calibration-tail", required=True)
    parser.add_argument("--test-writer", required=True)
    parser.add_argument("--test-tail", required=True)
    parser.add_argument("--writer-layer-pattern", default="first_30")
    parser.add_argument("--tail-layer-pattern", default="all")
    parser.add_argument("--quant-bits", type=int, default=4)
    parser.add_argument("--feature", default="source_answer_mean_logprob")
    parser.add_argument("--bootstrap-replicates", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    calibration_writer, calibration_tail = _paired_inputs(
        args.calibration_writer,
        args.calibration_tail,
        quant_bits=args.quant_bits,
        writer_layer_pattern=args.writer_layer_pattern,
        tail_layer_pattern=args.tail_layer_pattern,
    )
    threshold, calibration = select_threshold(
        calibration_writer, calibration_tail, feature=args.feature
    )
    test_writer, test_tail = _paired_inputs(
        args.test_writer,
        args.test_tail,
        quant_bits=args.quant_bits,
        writer_layer_pattern=args.writer_layer_pattern,
        tail_layer_pattern=args.tail_layer_pattern,
    )
    test = evaluate_gate(
        test_writer,
        test_tail,
        feature=args.feature,
        threshold=threshold,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    summary = {
        "feature": args.feature,
        "rule": "use_tail_if_feature_greater_than_or_equal_to_threshold_else_writer",
        "threshold": threshold,
        "calibration": calibration,
        "test": test,
        "config": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2) + "\n"
    output.write_text(rendered)
    print(rendered, end="")


def _paired_inputs(
    writer_path,
    tail_path,
    *,
    quant_bits: int,
    writer_layer_pattern: str,
    tail_layer_pattern: str,
):
    writer = _load_rows(
        writer_path, quant_bits, layer_pattern=writer_layer_pattern
    )
    tail = _load_rows(tail_path, quant_bits, layer_pattern=tail_layer_pattern)
    if writer.keys() != tail.keys():
        raise ValueError("writer and tail result IDs must match exactly")
    return writer, tail


def select_threshold(writer: dict, tail: dict, *, feature: str):
    scores = sorted({float(row[feature]) for row in tail.values()})
    thresholds = [-math.inf]
    thresholds.extend((left + right) / 2 for left, right in zip(scores, scores[1:]))
    thresholds.append(math.inf)
    candidates = []
    for threshold in thresholds:
        result = _gate_vectors(writer, tail, feature=feature, threshold=threshold)
        accuracy = sum(result["gate_exact"]) / len(result["gate_exact"])
        writer_cases = result["writer_cases"]
        candidates.append((accuracy, -writer_cases, threshold, result))
    accuracy, _, threshold, selected = max(candidates, key=lambda item: item[:2])
    return threshold, {
        "cases": len(selected["gate_exact"]),
        "gate_accuracy": accuracy,
        "writer_cases": selected["writer_cases"],
        "writer_fraction": selected["writer_cases"] / len(selected["gate_exact"]),
        "tail_accuracy": sum(selected["tail_exact"]) / len(selected["tail_exact"]),
        "writer_accuracy": sum(selected["writer_exact"]) / len(selected["writer_exact"]),
    }


def evaluate_gate(
    writer: dict,
    tail: dict,
    *,
    feature: str,
    threshold: float,
    bootstrap_replicates: int,
    seed: int,
):
    vectors = _gate_vectors(writer, tail, feature=feature, threshold=threshold)
    summary = paired_summary(
        vectors["gate_exact"],
        vectors["tail_exact"],
        vectors["gate_f1"],
        vectors["tail_f1"],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    summary["writer_cases"] = vectors["writer_cases"]
    summary["writer_fraction"] = vectors["writer_cases"] / len(vectors["gate_exact"])
    summary["standalone_writer_accuracy"] = sum(vectors["writer_exact"]) / len(
        vectors["writer_exact"]
    )
    return summary


def _gate_vectors(writer: dict, tail: dict, *, feature: str, threshold: float):
    gate_exact = []
    gate_f1 = []
    writer_exact = []
    tail_exact = []
    tail_f1 = []
    writer_cases = 0
    for case_id in sorted(writer):
        use_tail = float(tail[case_id][feature]) >= threshold
        selected = tail[case_id] if use_tail else writer[case_id]
        writer_cases += int(not use_tail)
        gate_exact.append(selected["student_exact"])
        gate_f1.append(selected["student_f1"])
        writer_exact.append(writer[case_id]["student_exact"])
        tail_exact.append(tail[case_id]["student_exact"])
        tail_f1.append(tail[case_id]["student_f1"])
    return {
        "gate_exact": gate_exact,
        "gate_f1": gate_f1,
        "writer_exact": writer_exact,
        "tail_exact": tail_exact,
        "tail_f1": tail_f1,
        "writer_cases": writer_cases,
    }


if __name__ == "__main__":
    main()
