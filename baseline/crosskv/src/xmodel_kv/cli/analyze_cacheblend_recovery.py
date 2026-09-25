from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import string

from .analyze_hotpot_competitors import load_unique_rows
from .analyze_paired_transfer import paired_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether full CacheBlend recomputation recovers cold quality."
    )
    parser.add_argument("--cold-results", nargs="+", required=True)
    parser.add_argument("--blend-results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--equivalence-margin", type=float, default=0.05)
    parser.add_argument("--minimum-answer-agreement", type=float, default=0.80)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if min(args.expected_cases, args.bootstrap_replicates) < 1:
        parser.error("case and bootstrap counts must be positive")
    if not 0 < args.equivalence_margin < 1:
        parser.error("equivalence-margin must be in (0, 1)")
    if not 0 <= args.minimum_answer_agreement <= 1:
        parser.error("minimum-answer-agreement must be in [0, 1]")

    cold = load_unique_rows(args.cold_results, name="CacheBlend cold")
    blend = load_unique_rows(args.blend_results, name="CacheBlend recovery")
    rows = join_recovery_rows(cold, blend, expected_cases=args.expected_cases)
    analysis = analyze_recovery(
        rows,
        equivalence_margin=args.equivalence_margin,
        minimum_answer_agreement=args.minimum_answer_agreement,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    analysis["inputs"] = {
        "cold_results": args.cold_results,
        "blend_results": args.blend_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


def join_recovery_rows(cold, blend, *, expected_cases: int) -> list[dict]:
    if len(blend) != expected_cases:
        raise ValueError("recovery input does not contain the expected cases")
    if not set(blend) <= set(cold):
        raise ValueError("recovery IDs are not a subset of cold IDs")
    rows = []
    for case_id in sorted(blend, key=lambda value: int(blend[value]["index"])):
        reference = cold[case_id]
        candidate = blend[case_id]
        for key in (
            "index",
            "question",
            "gold_answers",
            "prompt_tokens",
            "document_tokens",
            "document_kv_bf16_bytes",
        ):
            if reference[key] != candidate[key]:
                raise ValueError(f"cold/recovery mismatch for {case_id}: {key}")
        if float(candidate["recompute_ratio"]) != 1.0:
            raise ValueError("recovery analysis requires recompute_ratio=1.0")
        rows.append({"cold": reference, "blend": candidate})
    return rows


def analyze_recovery(
    rows,
    *,
    equivalence_margin: float,
    minimum_answer_agreement: float,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    ratios = {float(row["blend"]["recompute_ratio"]) for row in rows}
    check_layers = {int(row["blend"]["blend_check_layer"]) for row in rows}
    if ratios != {1.0} or len(check_layers) != 1:
        raise ValueError("recovery rows must share ratio=1 and one check layer")
    comparison = paired_summary(
        [row["blend"]["cacheblend_em"] for row in rows],
        [row["cold"]["cold_full_em"] for row in rows],
        [row["blend"]["cacheblend_f1"] for row in rows],
        [row["cold"]["cold_full_f1"] for row in rows],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    raw_agreements = [
        row["blend"]["cacheblend"] == row["cold"]["cold_full"] for row in rows
    ]
    normalized_agreements = [
        normalize_answer(row["blend"]["cacheblend"])
        == normalize_answer(row["cold"]["cold_full"])
        for row in rows
    ]
    accuracy_ci = comparison["accuracy_delta_bootstrap_95ci"]
    f1_ci = comparison["mean_f1_delta_bootstrap_95ci"]
    metric_equivalence = all(
        -equivalence_margin <= endpoint <= equivalence_margin
        for endpoint in (*accuracy_ci, *f1_ci)
    )
    normalized_agreement = sum(normalized_agreements) / len(rows)
    return {
        "cases": len(rows),
        "recompute_ratio": ratios.pop(),
        "blend_check_layer": check_layers.pop(),
        "equivalence_margin": equivalence_margin,
        "minimum_answer_agreement": minimum_answer_agreement,
        "raw_answer_agreement": sum(raw_agreements) / len(rows),
        "normalized_answer_agreement": normalized_agreement,
        "strict_exact_recovery": all(normalized_agreements),
        "metric_equivalence": metric_equivalence,
        "recovery_gate_pass": (
            metric_equivalence
            and normalized_agreement >= minimum_answer_agreement
        ),
        "comparison": comparison,
        "mismatched_indices": [
            int(row["blend"]["index"])
            for row, agrees in zip(rows, normalized_agreements, strict=True)
            if not agrees
        ],
    }


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


if __name__ == "__main__":
    main()
