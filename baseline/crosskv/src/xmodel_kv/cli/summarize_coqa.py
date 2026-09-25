from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly merge and summarize CoQA handoff JSONL")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--expected-conversations", type=int, default=100)
    parser.add_argument("--expected-turns", default="1,3,5,7,10")
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = []
    for pattern in args.inputs:
        paths = sorted(Path().glob(pattern)) if any(c in pattern for c in "*?[") else [Path(pattern)]
        for path in paths:
            with path.open() as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    turns = tuple(int(value) for value in args.expected_turns.split(","))
    result = summarize(rows, expected_conversations=args.expected_conversations, expected_turns=turns)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered)
    print(rendered, end="")


def summarize(
    rows: list[dict], *, expected_conversations: int, expected_turns: tuple[int, ...]
) -> dict:
    keys = [(int(row["index"]), int(row["turn"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (conversation index, turn) rows")
    expected_rows = expected_conversations * len(expected_turns)
    if len(rows) != expected_rows:
        raise ValueError(f"coverage mismatch: got {len(rows)} rows; expected {expected_rows}")

    by_turn = defaultdict(list)
    for row in rows:
        by_turn[int(row["turn"])].append(row)
    if set(by_turn) != set(expected_turns):
        raise ValueError(f"turn coverage mismatch: got {sorted(by_turn)}; expected {list(expected_turns)}")

    turn_results = []
    for turn in expected_turns:
        turn_rows = by_turn[turn]
        if len(turn_rows) != expected_conversations:
            raise ValueError(
                f"turn {turn} has {len(turn_rows)} conversations; expected {expected_conversations}"
            )
        standalone = sum(float(row["standalone_f1"]) for row in turn_rows) / len(turn_rows)
        transfer = sum(float(row["transfer_f1"]) for row in turn_rows) / len(turn_rows)
        turn_results.append(
            {
                "turn": turn,
                "conversations": len(turn_rows),
                "standalone_f1": standalone,
                "transfer_f1": transfer,
                "drift_percentage_points": 100.0 * (standalone - transfer),
            }
        )

    overall_standalone = sum(float(row["standalone_f1"]) for row in rows) / len(rows)
    overall_transfer = sum(float(row["transfer_f1"]) for row in rows) / len(rows)
    domains = sorted({str(row["domain"]) for row in rows})
    return {
        "conversations": expected_conversations,
        "turns": list(expected_turns),
        "rows": len(rows),
        "domains": domains,
        "overall_standalone_f1": overall_standalone,
        "overall_transfer_f1": overall_transfer,
        "overall_drift_percentage_points": 100.0 * (overall_standalone - overall_transfer),
        "per_turn": turn_results,
    }


if __name__ == "__main__":
    main()
