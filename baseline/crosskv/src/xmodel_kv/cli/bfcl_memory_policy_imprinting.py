from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bfcl_policy_imprinting import _jsonl, _normalize_schema_types
from .policy_imprinting import ControlledCase, _run_case


DOMAINS = ("customer", "finance", "healthcare", "notetaker", "student")
MODES = ("tagged", "verbatim")
POLICY_STYLES = ("strong", "minimal")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "First/last policy-imprinting test on official BFCL memory-prerequisite "
            "conversations"
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--mode", choices=MODES, default="tagged")
    parser.add_argument("--policy-style", choices=POLICY_STYLES, default="strong")
    parser.add_argument("--directions", default="forward,reverse")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--replay-tokens", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    domains = tuple(part.strip() for part in args.domains.split(",") if part.strip())
    unknown_domains = set(domains) - set(DOMAINS)
    if not domains or unknown_domains:
        parser.error(f"unknown or empty domains: {sorted(unknown_domains)}")
    directions = tuple(
        part.strip() for part in args.directions.split(",") if part.strip()
    )
    if not directions or set(directions) - {"forward", "reverse"}:
        parser.error("directions must contain forward and/or reverse")
    try:
        replay_tokens = tuple(int(part) for part in args.replay_tokens.split(","))
    except ValueError as error:
        raise SystemExit("replay-tokens must be comma-separated integers") from error
    if any(value < 0 for value in replay_tokens):
        parser.error("replay-tokens must be non-negative")
    if args.max_cases < 1 or args.max_new_tokens < 1:
        parser.error("max-cases and max-new-tokens must be positive")

    root = Path(args.dataset_root)
    candidates = memory_cases(
        root,
        domains=domains,
        mode=args.mode,
        policy_style=args.policy_style,
        max_cases=args.max_cases,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(output_path)
    pending = [
        (selection_index, bfcl_id, domain, direction, case, metadata)
        for selection_index, (bfcl_id, domain, cases, metadata) in enumerate(candidates)
        for direction, case in cases.items()
        if direction in directions
        and (bfcl_id, args.mode, args.policy_style, direction) not in completed
    ]
    if not pending:
        print({"processed_this_run": 0, "message": "all selected memory cases complete"})
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()

    processed = 0
    with output_path.open("a") as output:
        for selection_index, bfcl_id, domain, direction, case, metadata in pending:
            row = _run_case(
                model,
                tokenizer,
                case,
                replay_tokens=replay_tokens,
                max_new_tokens=args.max_new_tokens,
            )
            row.update(
                {
                    "bfcl_id": bfcl_id,
                    "bfcl_domain": domain,
                    "bfcl_mode": args.mode,
                    "bfcl_policy_style": args.policy_style,
                    "bfcl_selection_index": selection_index,
                    "bfcl_memory_metadata": metadata,
                }
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            processed += 1
            zero_replay = row["hybrid"][0]
            print(
                {
                    "bfcl_id": bfcl_id,
                    "domain": domain,
                    "direction": direction,
                    "history_tokens": row["token_lengths"]["history"],
                    "native_contrast": row["native_contrast"],
                    "source_expected": row["source_native"]["expected_full_call_em"],
                    "target_expected": row["target_native"]["expected_full_call_em"],
                    "hybrid_target_agreement": zero_replay["target_native_agreement"],
                    "hybrid_source_exact": zero_replay["tool_call"]
                    == row["source_native"]["tool_call"],
                    "source_leakage_shift": zero_replay["source_leakage_shift"],
                },
                flush=True,
            )
    print({"processed_this_run": processed, "output": str(output_path)})


def memory_cases(
    root: Path,
    *,
    domains: tuple[str, ...] = DOMAINS,
    mode: str = "tagged",
    policy_style: str = "strong",
    max_cases: int | None = None,
):
    if mode not in MODES:
        raise ValueError(f"unknown memory mode: {mode}")
    if policy_style not in POLICY_STYLES:
        raise ValueError(f"unknown policy style: {policy_style}")
    tool = _memory_add_tool(root)
    selected = []
    for domain in domains:
        path = root / "memory_prereq_conversation" / f"memory_{domain}.json"
        for document in _jsonl(path):
            records = [
                message["content"]
                for turn in document["question"]
                for message in turn
                if message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ]
            if len(records) < 2 or records[0] == records[-1]:
                continue
            cases, metadata = _memory_document_cases(
                document,
                domain=domain,
                records=records,
                tool=tool,
                mode=mode,
                policy_style=policy_style,
            )
            selected.append((document["id"], domain, cases, metadata))
            if max_cases is not None and len(selected) >= max_cases:
                return selected
    return selected


def memory_case(
    root: Path,
    *,
    bfcl_id: str,
    mode: str,
    direction: str,
    policy_style: str = "strong",
):
    for document_id, domain, cases, metadata in memory_cases(
        root, mode=mode, policy_style=policy_style, max_cases=None
    ):
        if document_id == bfcl_id:
            return cases[direction], domain, metadata
    raise KeyError(f"BFCL memory case not found: {bfcl_id}")


def _memory_document_cases(
    document, *, domain: str, records, tool, mode: str, policy_style: str = "strong"
):
    identifiers = [
        _record_identifier(document["id"], index) for index in range(len(records))
    ]
    history: list[dict[str, str]] = []
    for index, record in enumerate(records):
        content = record
        if mode == "tagged":
            content = f"Memory record identifier: {identifiers[index]}\n{record}"
        history.append({"role": "user", "content": content})
        history.append(
            {
                "role": "assistant",
                "content": "Memory record received; waiting for the retention handoff.",
            }
        )
    history.append(
        {
            "role": "user",
            "content": "The memory batch is complete. Apply the active retention policy now.",
        }
    )

    if mode == "tagged":
        values = {"first": identifiers[0], "last": identifiers[-1]}
        selection_instruction = (
            "Set the text argument to exactly the opaque identifier at the beginning of "
            "the selected record; do not copy or summarize its body."
        )
    else:
        values = {"first": records[0], "last": records[-1]}
        selection_instruction = (
            "Set the text argument to the complete selected user record verbatim, without "
            "its chat wrapper and without paraphrasing."
        )
    if policy_style == "strong":
        prompt = (
            "You are a memory-retention agent receiving a batch of user memory records. Your "
            "active policy is to retain only the {choice} record in the batch and reject the "
            "{other} record for this handoff. When the batch-complete trigger arrives, call "
            "archival_memory_add exactly once. {selection_instruction} Return exactly one tool "
            "call and no prose."
        )
    elif policy_style == "minimal":
        if mode == "tagged":
            payload = (
                "Set text to only that record's exact opaque identifier, not its body."
            )
        else:
            payload = "Set text to that record's complete text verbatim."
        prompt = (
            "Retention position: {choice}. At batch completion, call archival_memory_add "
            f"once. {payload} Return only the tool call."
        )
    else:
        raise ValueError(f"unknown policy style: {policy_style}")

    def build(direction: str, source_choice: str, target_choice: str) -> ControlledCase:
        return ControlledCase(
            scenario=f"bfcl_memory_{mode}",
            direction=direction,
            history_turn_pairs=len(records),
            source_prompt=prompt.format(
                choice=source_choice,
                other=target_choice,
                selection_instruction=selection_instruction,
            ),
            target_prompt=prompt.format(
                choice=target_choice,
                other=source_choice,
                selection_instruction=selection_instruction,
            ),
            source_tools=[tool],
            target_tools=[tool],
            history=history,
            source_expected={
                "name": "archival_memory_add",
                "arguments": {"text": values[source_choice]},
            },
            target_expected={
                "name": "archival_memory_add",
                "arguments": {"text": values[target_choice]},
            },
        )

    metadata = {
        "topic": document.get("topic"),
        "record_count": len(records),
        "original_characters": sum(len(record) for record in records),
        "first_characters": len(records[0]),
        "last_characters": len(records[-1]),
        "first_identifier": identifiers[0],
        "last_identifier": identifiers[-1],
        "policy_style": policy_style,
    }
    return {
        "forward": build("forward", "first", "last"),
        "reverse": build("reverse", "last", "first"),
    }, metadata


def _memory_add_tool(root: Path) -> dict[str, Any]:
    path = root / "multi_turn_func_doc" / "memory_vector.json"
    for function in _jsonl(path):
        if function["name"] != "archival_memory_add":
            continue
        function = copy.deepcopy(function)
        function.pop("response", None)
        _normalize_schema_types(function.get("parameters"))
        return {"type": "function", "function": function}
    raise ValueError(f"archival_memory_add missing from {path}")


def _record_identifier(document_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{document_id}:{index}".encode()).hexdigest()[:8].upper()
    return f"MEM-{digest}"


def _completed(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    return {
        (
            row["bfcl_id"],
            row["bfcl_mode"],
            row.get("bfcl_policy_style", "strong"),
            row["direction"],
        )
        for row in _jsonl(path)
        if "bfcl_id" in row and "bfcl_mode" in row
    }


if __name__ == "__main__":
    main()
