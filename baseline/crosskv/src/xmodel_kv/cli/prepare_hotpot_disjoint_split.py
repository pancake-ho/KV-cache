from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove question overlap from a candidate HotpotQA JSONL split."
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    candidate = load_jsonl(args.candidate)
    reference = load_jsonl(args.reference)
    kept, excluded = disjoint_questions(candidate, reference)
    if not kept:
        raise ValueError("question de-duplication removed every candidate row")
    rendered = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    summary = {
        "candidate": args.candidate,
        "reference": args.reference,
        "candidate_rows": len(candidate),
        "reference_rows": len(reference),
        "kept_rows": len(kept),
        "excluded_question_overlaps": len(excluded),
        "excluded": [
            {"id": row.get("_id"), "input": row["input"]} for row in excluded
        ],
        "output_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any("input" not in row for row in rows):
        raise ValueError("Hotpot JSONL must contain non-empty rows with input fields")
    return rows


def disjoint_questions(candidate: list[dict], reference: list[dict]):
    reference_questions = {normalize_question(row["input"]) for row in reference}
    kept = []
    excluded = []
    seen = set()
    for row in candidate:
        question = normalize_question(row["input"])
        if question in reference_questions or question in seen:
            excluded.append(row)
        else:
            kept.append(row)
            seen.add(question)
    return kept, excluded


if __name__ == "__main__":
    main()
