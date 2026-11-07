from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_paired_transfer import paired_summary


ARMS = ("no_summary", "capsule", "shifted_capsule", "generated_tail")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired comparison of arms from two summary-transfer variants."
    )
    parser.add_argument("--candidate-results", nargs="+", required=True)
    parser.add_argument("--candidate-arm", choices=ARMS, default="capsule")
    parser.add_argument("--reference-results", nargs="+", required=True)
    parser.add_argument("--reference-arm", choices=ARMS, default="capsule")
    parser.add_argument(
        "--protocol",
        choices=("state_readout", "question_conditioned"),
        default="question_conditioned",
    )
    parser.add_argument("--expected-cases", type=int, default=200)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.expected_cases, args.bootstrap_replicates) < 1:
        parser.error("case and bootstrap counts must be positive")

    candidate = load_protocol_rows(
        args.candidate_results,
        protocol=args.protocol,
        expected_cases=args.expected_cases,
    )
    reference = load_protocol_rows(
        args.reference_results,
        protocol=args.protocol,
        expected_cases=args.expected_cases,
    )
    result = compare_variants(
        candidate,
        reference,
        candidate_arm=args.candidate_arm,
        reference_arm=args.reference_arm,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    result.update(
        {
            "candidate_results": args.candidate_results,
            "candidate_arm": args.candidate_arm,
            "reference_results": args.reference_results,
            "reference_arm": args.reference_arm,
            "protocol": args.protocol,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def load_protocol_rows(paths, *, protocol: str, expected_cases: int) -> dict[str, dict]:
    selected = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            path = path / "results.jsonl"
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("protocol") != protocol:
                    continue
                case_id = row["id"]
                if case_id in selected:
                    raise ValueError(f"duplicate result ID: {case_id}")
                selected[case_id] = row
    if len(selected) != expected_cases:
        raise ValueError(
            f"expected {expected_cases} {protocol} cases, found {len(selected)}"
        )
    return selected


def compare_variants(
    candidate: dict[str, dict],
    reference: dict[str, dict],
    *,
    candidate_arm: str,
    reference_arm: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if set(candidate) != set(reference):
        raise ValueError("variant result IDs differ")
    ids = sorted(candidate)
    for case_id in ids:
        left, right = candidate[case_id], reference[case_id]
        for field in ("question", "gold_answers"):
            if left.get(field) != right.get(field):
                raise ValueError(f"variant field differs for {case_id}: {field}")
    result = paired_summary(
        [candidate[case_id][f"{candidate_arm}_em"] for case_id in ids],
        [reference[case_id][f"{reference_arm}_em"] for case_id in ids],
        [candidate[case_id][f"{candidate_arm}_f1"] for case_id in ids],
        [reference[case_id][f"{reference_arm}_f1"] for case_id in ids],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    agreement_fields = {
        "arm_output": (candidate_arm, reference_arm),
        "arm_em": (f"{candidate_arm}_em", f"{reference_arm}_em"),
        "arm_f1": (f"{candidate_arm}_f1", f"{reference_arm}_f1"),
        "source_answer": ("source_answer", "source_answer"),
        "source_answer_em": ("source_answer_em", "source_answer_em"),
        "source_answer_f1": ("source_answer_f1", "source_answer_f1"),
    }
    agreement = {}
    for label, (candidate_field, reference_field) in agreement_fields.items():
        if not all(
            candidate_field in candidate[case_id]
            and reference_field in reference[case_id]
            for case_id in ids
        ):
            continue
        differing_ids = [
            case_id
            for case_id in ids
            if candidate[case_id][candidate_field]
            != reference[case_id][reference_field]
        ]
        agreement[label] = {
            "equal": len(ids) - len(differing_ids),
            "different": len(differing_ids),
            "fraction_equal": (len(ids) - len(differing_ids)) / len(ids),
            "differing_ids": differing_ids,
        }
    result["agreement"] = agreement
    return result


if __name__ == "__main__":
    main()
