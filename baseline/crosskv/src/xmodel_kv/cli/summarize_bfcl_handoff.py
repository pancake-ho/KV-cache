from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize_policy_triad import _exact_mcnemar, _rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize BFCL planner-to-executor KV handoff experiments"
    )
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [row for path in args.inputs for row in _jsonl(Path(path))]
    keys = [(row["bfcl_id"], row["direction"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate BFCL id/direction rows detected")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(rows)}")
    summary = summarize(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def summarize(rows):
    if not rows:
        raise ValueError("cannot summarize an empty BFCL handoff run")
    arm_count = len(rows[0]["hybrid"])
    if any(len(row["hybrid"]) != arm_count for row in rows):
        raise ValueError("all rows must contain the same replay arms")
    return {
        "rows": len(rows),
        "overall": _cell(rows),
        "by_direction": {
            direction: _cell([row for row in rows if row["direction"] == direction])
            for direction in sorted({row["direction"] for row in rows})
        },
        "replay_curve": [
            _replay_cell(rows, hybrid_index) for hybrid_index in range(arm_count)
        ],
    }


def _cell(rows):
    target_capable = [
        row
        for row in rows
        if _correct(row["target_native"]["tool_call"], row["target_expected"])
        and row["identity"]["target_native_agreement"]
    ]
    causal_contrast = [
        row
        for row in target_capable
        if not _same_call(
            row["source_native"]["tool_call"], row["target_native"]["tool_call"]
        )
    ]
    strict = [
        row
        for row in causal_contrast
        if _correct(row["source_native"]["tool_call"], row["source_expected"])
    ]
    native_correct = [
        _correct(row["target_native"]["tool_call"], row["target_expected"])
        for row in rows
    ]
    reuse_correct = [
        _correct(row["hybrid"][0]["tool_call"], row["target_expected"])
        for row in rows
    ]
    native_only = sum(a and not b for a, b in zip(native_correct, reuse_correct, strict=True))
    reuse_only = sum(b and not a for a, b in zip(native_correct, reuse_correct, strict=True))
    result = {
        "rows": len(rows),
        "target_native_expected_accuracy": _rate(sum(native_correct), len(rows)),
        "reuse_expected_accuracy": _rate(sum(reuse_correct), len(rows)),
        "paired_expected_accuracy": {
            "native_correct_reuse_wrong": native_only,
            "native_wrong_reuse_correct": reuse_only,
            "exact_mcnemar_two_sided_p": _exact_mcnemar(native_only, reuse_only),
        },
        "target_capable": _contrast_metrics(target_capable),
        "causal_contrast": _contrast_metrics(causal_contrast),
        "strict_both_policies_correct": _contrast_metrics(strict),
    }
    return result


def _replay_cell(rows, hybrid_index):
    arm_rows = [row["hybrid"][hybrid_index] for row in rows]
    requested = {
        arm.get("requested_replay_tokens", hybrid_index) for arm in arm_rows
    }
    if len(requested) != 1:
        raise ValueError(f"replay arm {hybrid_index} has inconsistent requested lengths")
    target_capable = [
        row
        for row in rows
        if _correct(row["target_native"]["tool_call"], row["target_expected"])
        and row["identity"]["target_native_agreement"]
        and not _same_call(
            row["source_native"]["tool_call"], row["target_native"]["tool_call"]
        )
    ]
    strict = [
        row
        for row in target_capable
        if _correct(row["source_native"]["tool_call"], row["source_expected"])
    ]
    actual = [
        arm.get("replay_tokens", next(iter(requested))) for arm in arm_rows
    ]
    native_correct = [
        _correct(row["target_native"]["tool_call"], row["target_expected"])
        for row in rows
    ]
    reuse_correct = [
        _correct(row["hybrid"][hybrid_index]["tool_call"], row["target_expected"])
        for row in rows
    ]
    native_only = sum(a and not b for a, b in zip(native_correct, reuse_correct, strict=True))
    reuse_only = sum(b and not a for a, b in zip(native_correct, reuse_correct, strict=True))
    result = {
        "requested_replay_tokens": next(iter(requested)),
        "actual_replay_tokens_min": min(actual),
        "actual_replay_tokens_max": max(actual),
        "reuse_expected_accuracy": _rate(sum(reuse_correct), len(rows)),
        "native_correct_reuse_wrong": native_only,
        "native_wrong_reuse_correct": reuse_only,
        "exact_mcnemar_two_sided_p": _exact_mcnemar(native_only, reuse_only),
        "target_capable": _replay_contrast(target_capable, hybrid_index),
        "strict_both_policies_correct": _replay_contrast(strict, hybrid_index),
    }
    return result


def _replay_contrast(rows, hybrid_index):
    if not rows:
        return {"rows": 0}
    target_matches = sum(
        _same_call(
            row["hybrid"][hybrid_index]["tool_call"],
            row["target_native"]["tool_call"],
        )
        for row in rows
    )
    source_matches = sum(
        _same_call(
            row["hybrid"][hybrid_index]["tool_call"],
            row["source_native"]["tool_call"],
        )
        for row in rows
    )
    return {
        "rows": len(rows),
        "reuse_matches_target_native": _rate(target_matches, len(rows)),
        "reuse_matches_source_native": _rate(source_matches, len(rows)),
    }


def _contrast_metrics(rows):
    if not rows:
        return {"rows": 0}
    target_matches = sum(
        _same_call(row["hybrid"][0]["tool_call"], row["target_native"]["tool_call"])
        for row in rows
    )
    source_matches = sum(
        _same_call(row["hybrid"][0]["tool_call"], row["source_native"]["tool_call"])
        for row in rows
    )
    target_expected = sum(
        _correct(row["hybrid"][0]["tool_call"], row["target_expected"])
        for row in rows
    )
    return {
        "rows": len(rows),
        "reuse_target_expected_accuracy": _rate(target_expected, len(rows)),
        "reuse_matches_target_native": _rate(target_matches, len(rows)),
        "reuse_matches_source_native": _rate(source_matches, len(rows)),
        "reuse_matches_neither": _rate(
            len(rows)
            - sum(
                _same_call(row["hybrid"][0]["tool_call"], row["target_native"]["tool_call"])
                or _same_call(
                    row["hybrid"][0]["tool_call"], row["source_native"]["tool_call"]
                )
                for row in rows
            ),
            len(rows),
        ),
    }


def _correct(call, expected):
    return (
        call.get("valid_json", False)
        and call.get("name") == expected.get("name")
        and call.get("arguments") == expected.get("arguments")
    )


def _same_call(first, second):
    return (
        first.get("valid_json", False)
        and second.get("valid_json", False)
        and first.get("name") == second.get("name")
        and first.get("arguments") == second.get("arguments")
    )


def _jsonl(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
