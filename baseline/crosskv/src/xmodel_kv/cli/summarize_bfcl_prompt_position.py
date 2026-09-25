from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bfcl_policy_imprinting import _jsonl
from .summarize_policy_triad import _exact_mcnemar, _rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize long-history front-vs-tail agent-policy placement"
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
        raise ValueError("cannot summarize an empty prompt-position run")
    tail_roles = tuple(sorted(rows[0]["tail"]))
    if any(tuple(sorted(row["tail"])) != tail_roles for row in rows):
        raise ValueError("rows have inconsistent tail roles")
    return {
        "rows": len(rows),
        "history_tokens": {
            "min": min(row["token_lengths"]["history"] for row in rows),
            "max": max(row["token_lengths"]["history"] for row in rows),
        },
        "all_rows": _all_rows(rows, tail_roles),
        "layout_specific": {
            "front": _layout_cell(rows, "front"),
            **{
                f"tail_{role}": _layout_cell(rows, "tail", role)
                for role in tail_roles
            },
        },
        "common_strict": {
            role: _common_cell(rows, role) for role in tail_roles
        },
        "by_direction": {
            direction: {
                role: _common_cell(
                    [row for row in rows if row["direction"] == direction], role
                )
                for role in tail_roles
            }
            for direction in sorted({row["direction"] for row in rows})
        },
    }


def _all_rows(rows, tail_roles):
    return {
        "source_native_expected_accuracy": _accuracy(rows, ("source_native",)),
        "front_native_expected_accuracy": _accuracy(rows, ("front", "native")),
        "front_reuse_expected_accuracy": _accuracy(rows, ("front", "reuse")),
        **{
            f"tail_{role}_native_expected_accuracy": _accuracy(
                rows, ("tail", role, "native")
            )
            for role in tail_roles
        },
        **{
            f"tail_{role}_reuse_expected_accuracy": _accuracy(
                rows, ("tail", role, "reuse_drop_source")
            )
            for role in tail_roles
        },
    }


def _layout_cell(rows, layout, role=None):
    if layout == "front":
        native_path = ("front", "native")
        identity_path = ("front", "identity")
        reuse_path = ("front", "reuse")
    else:
        native_path = ("tail", role, "native")
        identity_path = ("tail", role, "identity")
        reuse_path = ("tail", role, "reuse_drop_source")
    eligible = [
        row
        for row in rows
        if _correct(_get(row, ("source_native",)), row["source_expected"])
        and _correct(_get(row, native_path), row["target_expected"])
        and _same(_get(row, identity_path), _get(row, native_path))
        and not _same(_get(row, ("source_native",)), _get(row, native_path))
    ]
    return _contrast(eligible, reuse_path)


def _common_cell(rows, role):
    eligible = [
        row
        for row in rows
        if _correct(_get(row, ("source_native",)), row["source_expected"])
        and _correct(_get(row, ("front", "native")), row["target_expected"])
        and _same(_get(row, ("front", "identity")), _get(row, ("front", "native")))
        and _correct(_get(row, ("tail", role, "native")), row["target_expected"])
        and _same(
            _get(row, ("tail", role, "identity")),
            _get(row, ("tail", role, "native")),
        )
        and not _same(_get(row, ("source_native",)), _get(row, ("front", "native")))
        and not _same(
            _get(row, ("source_native",)), _get(row, ("tail", role, "native"))
        )
    ]
    front = [
        _correct(_get(row, ("front", "reuse")), row["target_expected"])
        for row in eligible
    ]
    tail = [
        _correct(
            _get(row, ("tail", role, "reuse_drop_source")), row["target_expected"]
        )
        for row in eligible
    ]
    front_only = sum(a and not b for a, b in zip(front, tail, strict=True))
    tail_only = sum(b and not a for a, b in zip(front, tail, strict=True))
    return {
        "rows": len(eligible),
        "front": _contrast(eligible, ("front", "reuse")),
        "tail_drop_source": _contrast(
            eligible, ("tail", role, "reuse_drop_source")
        ),
        "tail_keep_source": _contrast(
            eligible, ("tail", role, "reuse_keep_source")
        ),
        "paired_front_vs_tail_target_accuracy": {
            "front_correct_tail_wrong": front_only,
            "front_wrong_tail_correct": tail_only,
            "exact_mcnemar_two_sided_p": _exact_mcnemar(front_only, tail_only),
        },
    }


def _contrast(rows, path):
    if not rows:
        return {"rows": 0}
    calls = [_get(row, path) for row in rows]
    target = sum(
        _correct(call, row["target_expected"])
        for call, row in zip(calls, rows, strict=True)
    )
    source = sum(
        _same(call, _get(row, ("source_native",)))
        for call, row in zip(calls, rows, strict=True)
    )
    return {
        "rows": len(rows),
        "target_expected_accuracy": _rate(target, len(rows)),
        "matches_source_native": _rate(source, len(rows)),
    }


def _accuracy(rows, path):
    correct = sum(
        _correct(_get(row, path), row["target_expected"])
        for row in rows
    )
    if path == ("source_native",):
        correct = sum(
            _correct(_get(row, path), row["source_expected"]) for row in rows
        )
    return _rate(correct, len(rows))


def _get(row, path):
    value = row
    for key in path:
        value = value[key]
    return value["tool_call"] if "tool_call" in value else value


def _correct(call, expected):
    return (
        call.get("valid_json", False)
        and call.get("name") == expected["name"]
        and call.get("arguments") == expected["arguments"]
    )


def _same(first, second):
    return (
        first.get("valid_json", False)
        and second.get("valid_json", False)
        and first.get("name") == second.get("name")
        and first.get("arguments") == second.get("arguments")
    )


if __name__ == "__main__":
    main()
