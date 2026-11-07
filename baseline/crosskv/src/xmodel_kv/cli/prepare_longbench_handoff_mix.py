from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


PASSAGE = re.compile(r"(?m)^Passage\s+\d+:\s*\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-audited LongBench handoff cases with an eight-way "
            "receiver lookup, optionally mixed with existing MuSiQue training rows."
        )
    )
    parser.add_argument("--hotpot", required=True)
    parser.add_argument("--twowiki", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--musique-train")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--dev-per-source", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2100)
    args = parser.parse_args()
    if min(args.dev_per_source, args.candidate_count) < 2:
        parser.error("dev size and candidate count must be at least two")

    excluded_questions = set()
    for path in args.exclude:
        excluded_questions.update(_questions(path))
    source_rows = {}
    for name, path in (("hotpotqa", args.hotpot), ("2wikimqa", args.twowiki)):
        rows = load_longbench_rows(path, dataset=name)
        source_rows[name] = [
            row for row in rows if normalize_question(row["input"]) not in excluded_questions
        ]

    generator = random.Random(args.seed)
    synthetic_train = []
    synthetic_dev = []
    split_counts = {}
    for source_position, (name, rows) in enumerate(source_rows.items()):
        rows = list(rows)
        random.Random(args.seed + source_position).shuffle(rows)
        if len(rows) <= args.dev_per_source:
            raise ValueError(f"{name} has too few leakage-free rows")
        dev_rows = rows[: args.dev_per_source]
        train_rows = rows[args.dev_per_source :]
        source_train = build_handoff_cases(
            train_rows,
            candidate_count=args.candidate_count,
            seed=args.seed + 10 + source_position,
        )
        source_dev = build_handoff_cases(
            dev_rows,
            candidate_count=args.candidate_count,
            seed=args.seed + 20 + source_position,
        )
        synthetic_train.extend(source_train)
        synthetic_dev.extend(source_dev)
        split_counts[name] = {
            "raw": len(load_longbench_rows(
                args.hotpot if name == "hotpotqa" else args.twowiki,
                dataset=name,
            )),
            "excluded_overlap": len(load_longbench_rows(
                args.hotpot if name == "hotpotqa" else args.twowiki,
                dataset=name,
            ))
            - len(rows),
            "handoff_train": len(source_train),
            "handoff_dev": len(source_dev),
        }

    generator.shuffle(synthetic_train)
    generator.shuffle(synthetic_dev)
    train_rows = list(synthetic_train)
    musique_rows = []
    if args.musique_train:
        musique_rows = _read_jsonl(args.musique_train)
        train_rows.extend(musique_rows)
        generator.shuffle(train_rows)

    _require_unique_ids(train_rows, label="train")
    _require_unique_ids(synthetic_dev, label="dev")
    overlap = {row["id"] for row in train_rows} & {
        row["id"] for row in synthetic_dev
    }
    if overlap:
        raise ValueError(f"train/dev ID overlap: {len(overlap)}")
    dev_questions = {
        normalize_question(row["global_question"]) for row in synthetic_dev
    }
    train_questions = {
        normalize_question(row["global_question"])
        for row in synthetic_train
    }
    if train_questions & dev_questions:
        raise ValueError("synthetic train/dev question overlap")
    if dev_questions & excluded_questions or train_questions & excluded_questions:
        raise ValueError("excluded evaluation question leaked into synthetic data")

    _write_jsonl(args.train_output, train_rows)
    _write_jsonl(args.dev_output, synthetic_dev)
    summary = {
        "builder": "longbench_handoff_mix_v1",
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "dev_per_source_requested": args.dev_per_source,
        "excluded_question_count": len(excluded_questions),
        "source_splits": split_counts,
        "synthetic_train_cases": len(synthetic_train),
        "musique_train_cases": len(musique_rows),
        "combined_train_cases": len(train_rows),
        "synthetic_dev_cases": len(synthetic_dev),
        "train_sha256": _sha256(args.train_output),
        "dev_sha256": _sha256(args.dev_output),
    }
    Path(args.summary_output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def load_longbench_rows(path: str, *, dataset: str) -> list[dict]:
    rows = []
    for raw in _read_jsonl(path):
        answers = raw.get("answers")
        if not isinstance(raw.get("input"), str) or not raw["input"].strip():
            raise ValueError("LongBench input must be non-empty text")
        if not isinstance(raw.get("context"), str) or not raw["context"].strip():
            raise ValueError("LongBench context must be non-empty text")
        if not isinstance(answers, list) or not answers or not str(answers[0]).strip():
            raise ValueError("LongBench answers must contain a non-empty first answer")
        rows.append(
            {
                "dataset": dataset,
                "input": raw["input"].strip(),
                "context": raw["context"].strip(),
                "answer": str(answers[0]).strip(),
                "raw_id": str(raw.get("_id", "")),
            }
        )
    return rows


def build_handoff_cases(
    rows: list[dict], *, candidate_count: int, seed: int
) -> list[dict]:
    groups = unique_answer_groups(rows, size=candidate_count, seed=seed)
    cases = []
    for group_index, group in enumerate(groups):
        final_answers = [group[(index + 1) % len(group)]["answer"] for index in range(len(group))]
        candidates = []
        for index, (row, final_answer) in enumerate(zip(group, final_answers, strict=True), 1):
            candidates.append(
                {
                    "source_id": stable_id(row),
                    "bridge_answer": row["answer"],
                    "final_answer": final_answer,
                    "document": {
                        "source_id": stable_id(row),
                        "title": row["dataset"],
                        "text": row["input"],
                    },
                    "is_gold": False,
                    "candidate_id": f"CAND-{index}",
                }
            )
        for row_index, row in enumerate(group):
            row_candidates = [dict(candidate) for candidate in candidates]
            row_candidates[row_index]["is_gold"] = True
            documents = parse_passages(row["context"], source_id=stable_id(row))
            cases.append(
                {
                    "benchmark": "longbench_decision_handoff_v1",
                    "id": stable_id(row),
                    "source_dataset": row["dataset"],
                    "global_question": row["input"],
                    "relation_key": "synthetic paired-result lookup",
                    "agent_a": {
                        "question": row["input"],
                        "gold_answer": row["answer"],
                        "documents": documents,
                    },
                    "agent_b": {
                        "question_template": "#1 >> paired result",
                        "gold_answer": final_answers[row_index],
                        "gold_candidate_id": f"CAND-{row_index + 1}",
                        "candidates": row_candidates,
                        "candidate_count": candidate_count,
                        "chance_accuracy": 1 / candidate_count,
                    },
                    "group_index": group_index,
                }
            )
    return cases


def unique_answer_groups(rows: list[dict], *, size: int, seed: int) -> list[list[dict]]:
    if size < 2:
        raise ValueError("group size must be at least two")
    remaining = list(rows)
    random.Random(seed).shuffle(remaining)
    groups = []
    while len(remaining) >= size:
        group = []
        used = set()
        kept = []
        for row in remaining:
            answer = normalize_answer(row["answer"])
            if len(group) < size and answer not in used:
                group.append(row)
                used.add(answer)
            else:
                kept.append(row)
        if len(group) < size:
            break
        groups.append(group)
        remaining = kept
    return groups


def parse_passages(context: str, *, source_id: str) -> list[dict]:
    parts = [part.strip() for part in PASSAGE.split(context) if part.strip()]
    if not parts:
        parts = [context.strip()]
    documents = []
    for index, part in enumerate(parts, 1):
        title, separator, text = part.partition("\n")
        if not separator:
            title, text = f"Passage {index}", part
        documents.append(
            {
                "source_id": f"{source_id}__p{index}",
                "title": title.strip() or f"Passage {index}",
                "text": text.strip(),
            }
        )
    return documents


def stable_id(row: dict) -> str:
    digest = hashlib.sha256(
        (row["dataset"] + "\n" + normalize_question(row["input"])).encode()
    ).hexdigest()[:24]
    return f"longbench__{row['dataset']}__{digest}"


def normalize_question(value: str) -> str:
    return " ".join(value.casefold().split())


def normalize_answer(value: str) -> str:
    return " ".join(value.casefold().split())


def _questions(path: str) -> set[str]:
    return {
        normalize_question(row["input"])
        for row in _read_jsonl(path)
        if isinstance(row.get("input"), str)
    }


def _read_jsonl(path: str) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _require_unique_ids(rows: list[dict], *, label: str) -> None:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} IDs are not unique")


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
