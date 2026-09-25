from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .policy_imprinting import ControlledCase, _run_case


CLASS_DOCUMENTS = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "TwitterAPI": "posting_api.json",
    "VehicleControlAPI": "vehicle_control.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Natural-history policy-imprinting test on official BFCL multi-turn data"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--category", choices=("base", "long_context"), default="base")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--directions", default="forward,reverse")
    parser.add_argument("--max-cases", type=int, default=20, help="BFCL conversations before direction expansion")
    parser.add_argument("--replay-tokens", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--handoff-proposal",
        action="store_true",
        help=(
            "append Agent A's first-step proposal and evaluate a planner-to-final-executor "
            "handoff instead of the original active-function control"
        ),
    )
    parser.add_argument(
        "--background-records",
        type=int,
        default=0,
        help=(
            "prepend this many real BFCL memory-conversation records as completed "
            "background history for length-stress experiments"
        ),
    )
    parser.add_argument(
        "--handoff-trigger-mode",
        choices=("shared", "none"),
        default="shared",
        help=(
            "shared appends an orchestrator handoff message; none models an automatic "
            "A-proposal-to-B-readout switch with no post-switch user tokens"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directions = tuple(part.strip() for part in args.directions.split(",") if part.strip())
    if not directions or set(directions) - {"forward", "reverse"}:
        parser.error("directions must contain forward and/or reverse")
    try:
        replay_tokens = tuple(int(part) for part in args.replay_tokens.split(","))
    except ValueError as error:
        raise SystemExit("replay-tokens must be comma-separated integers") from error
    if args.max_cases < 1 or args.max_new_tokens < 1:
        parser.error("max-cases and max-new-tokens must be positive")
    if args.background_records < 0:
        parser.error("background-records must be non-negative")

    root = Path(args.dataset_root)
    candidates = bfcl_cases(
        root,
        args.category,
        max_cases=args.max_cases,
        handoff_proposal=args.handoff_proposal,
        background_records=args.background_records,
        handoff_trigger_mode=args.handoff_trigger_mode,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(output_path)
    pending = [
        (index, bfcl_id, direction, case)
        for index, (bfcl_id, cases) in enumerate(candidates)
        for direction, case in cases.items()
        if direction in directions and (bfcl_id, direction) not in completed
    ]
    if not pending:
        print({"processed_this_run": 0, "message": "all selected BFCL cases complete"})
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
        for index, bfcl_id, direction, case in pending:
            row = _run_case(
                model,
                tokenizer,
                case,
                replay_tokens=replay_tokens,
                max_new_tokens=args.max_new_tokens,
            )
            row["bfcl_id"] = bfcl_id
            row["bfcl_category"] = args.category
            row["bfcl_selection_index"] = index
            row["bfcl_handoff_proposal"] = args.handoff_proposal
            row["bfcl_background_records"] = args.background_records
            row["bfcl_handoff_trigger_mode"] = args.handoff_trigger_mode
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            processed += 1
            zero_replay = row["hybrid"][0]
            print(
                {
                    "bfcl_id": bfcl_id,
                    "direction": direction,
                    "source": row["source_native"]["tool_call"]["name"],
                    "target": row["target_native"]["tool_call"]["name"],
                    "hybrid": zero_replay["tool_call"]["name"],
                    "native_contrast": row["native_contrast"],
                    "target_agreement": zero_replay["target_native_agreement"],
                    "source_leakage_shift": zero_replay["source_leakage_shift"],
                },
                flush=True,
            )
    print({"processed_this_run": processed, "output": str(output_path)})


def bfcl_cases(
    root: Path,
    category: str,
    *,
    max_cases: int,
    handoff_proposal: bool = False,
    background_records: int = 0,
    handoff_trigger_mode: str = "shared",
):
    if handoff_trigger_mode not in {"shared", "none"}:
        raise ValueError(f"unknown handoff trigger mode: {handoff_trigger_mode}")
    question_path = root / f"BFCL_v4_multi_turn_{category}.json"
    answer_path = root / "possible_answer" / f"BFCL_v4_multi_turn_{category}.json"
    questions = {row["id"]: row for row in _jsonl(question_path)}
    answers = {row["id"]: row for row in _jsonl(answer_path)}
    selected = []
    for bfcl_id, document in questions.items():
        ground_truth = answers[bfcl_id]["ground_truth"]
        if len(ground_truth) < 2 or len(ground_truth[0]) != 1 or len(ground_truth[-1]) != 1:
            continue
        first = _parse_call(ground_truth[0][0])
        last = _parse_call(ground_truth[-1][0])
        if first is None or last is None or first == last:
            continue
        tools = _tools_for_document(root, document)
        tool_names = {tool["function"]["name"] for tool in tools}
        if first["name"] not in tool_names or last["name"] not in tool_names:
            continue
        history = [
            *_background_history(root, background_records),
            *_natural_history(document["question"]),
        ]
        if handoff_proposal:
            history = _handoff_history(
                history,
                first,
                include_trigger=handoff_trigger_mode == "shared",
            )
            source_prompt, target_prompt = _handoff_prompts()
        else:
            source_prompt = target_prompt = None
        prompt = (
            "For this specialist-agent handoff, {active} is the active function and {inactive} "
            "is inactive. You must call {active} exactly once and derive its arguments from the "
            "matching user request in the conversation history. Never call {inactive}. Return "
            "exactly one tool call and no prose."
        )
        forward = ControlledCase(
            scenario=f"bfcl_{category}",
            direction="forward",
            history_turn_pairs=len(document["question"]),
            source_prompt=(
                source_prompt
                if handoff_proposal
                else prompt.format(active=first["name"], inactive=last["name"])
            ),
            target_prompt=(
                target_prompt
                if handoff_proposal
                else prompt.format(active=last["name"], inactive=first["name"])
            ),
            source_tools=tools,
            target_tools=tools,
            history=history,
            source_expected=first,
            target_expected=last,
            replicate=len(selected),
        )
        reverse = ControlledCase(
            scenario=f"bfcl_{category}",
            direction="reverse",
            history_turn_pairs=len(document["question"]),
            source_prompt=(
                target_prompt
                if handoff_proposal
                else prompt.format(active=last["name"], inactive=first["name"])
            ),
            target_prompt=(
                source_prompt
                if handoff_proposal
                else prompt.format(active=first["name"], inactive=last["name"])
            ),
            source_tools=tools,
            target_tools=tools,
            history=history,
            source_expected=last,
            target_expected=first,
            replicate=len(selected),
        )
        selected.append((bfcl_id, {"forward": forward, "reverse": reverse}))
        if len(selected) >= max_cases:
            break
    return selected


def _handoff_history(history, first_call, *, include_trigger: bool = True):
    proposal = json.dumps(first_call, ensure_ascii=False, separators=(",", ":"))
    result = [
        *history,
        {
            "role": "assistant",
            "content": (
                "Agent A initial-step proposal (preserved for the audit trail): "
                f"{proposal}\nThe orchestrator confirms that this initial-stage operation "
                "has already been handled upstream."
            ),
        },
    ]
    if include_trigger:
        result.append(
            {
                "role": "user",
                "content": (
                    "The workflow is now at the specialist handoff. Execute exactly one "
                    "operation for your assigned stage. Return exactly one tool call and no prose."
                ),
            }
        )
    return result


def _handoff_prompts():
    planner = (
        "You are Agent A, the initial-action planner in a sequential tool workflow. Your "
        "assigned stage is the earliest user request in the conversation. Preserve the "
        "initial-step proposal recorded in the audit trail. Whenever it is your turn to issue "
        "an operation, issue exactly that initial proposed operation, even "
        "when later requests exist. Return exactly one tool call and no prose."
    )
    executor = (
        "You are Agent B, the final-stage executor in a sequential tool workflow. Earlier "
        "planner operations recorded in the audit trail have already been handled and must "
        "not be repeated. Use the full conversation to identify the latest outstanding user "
        "request. When your turn begins after the planner proposal, issue exactly one "
        "appropriate operation for that final stage with its "
        "correct arguments. Return exactly one tool call and no prose."
    )
    return planner, executor


def _background_history(root: Path, count: int):
    if count <= 0:
        return []
    records = []
    for domain in ("notetaker", "customer", "healthcare", "student", "finance"):
        path = root / "memory_prereq_conversation" / f"memory_{domain}.json"
        for document in _jsonl(path):
            for turn in document["question"]:
                for message in turn:
                    content = message.get("content")
                    if message.get("role") != "user" or not isinstance(content, str):
                        continue
                    records.extend(
                        [
                            {
                                "role": "user",
                                "content": (
                                    "Completed background record for continuity only; it has "
                                    "no pending tool operation:\n" + content
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    "Background record archived. It does not change the active "
                                    "workflow stage."
                                ),
                            },
                        ]
                    )
                    if len(records) // 2 >= count:
                        return records
    raise ValueError(f"requested {count} BFCL background records but only found {len(records) // 2}")


def _natural_history(turns) -> list[dict[str, str]]:
    history = []
    for turn_index, turn in enumerate(turns):
        for message in turn:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                history.append({"role": "user", "content": message["content"]})
        if turn_index != len(turns) - 1:
            history.append(
                {
                    "role": "assistant",
                    "content": "Request recorded. I will wait for the execution policy.",
                }
            )
    return history


def _tools_for_document(root: Path, document) -> list[dict[str, Any]]:
    excluded = set(document.get("excluded_function", []))
    tools = []
    for class_name in document["involved_classes"]:
        path = root / "multi_turn_func_doc" / CLASS_DOCUMENTS[class_name]
        for function in _jsonl(path):
            if function["name"] in excluded:
                continue
            function = copy.deepcopy(function)
            function.pop("response", None)
            _normalize_schema_types(function.get("parameters"))
            tools.append({"type": "function", "function": function})
    return tools


def _normalize_schema_types(value) -> None:
    if isinstance(value, dict):
        if value.get("type") == "dict":
            value["type"] = "object"
        for child in value.values():
            _normalize_schema_types(child)
    elif isinstance(value, list):
        for child in value:
            _normalize_schema_types(child)


def _parse_call(expression: str) -> dict[str, Any] | None:
    try:
        node = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not isinstance(node, ast.Call) or node.args or any(item.arg is None for item in node.keywords):
        return None
    name_node = node.func
    while isinstance(name_node, ast.Attribute):
        name_node = name_node.attr
    if isinstance(name_node, ast.Name):
        name = name_node.id
    elif isinstance(name_node, str):
        name = name_node
    else:
        return None
    try:
        arguments = {item.arg: ast.literal_eval(item.value) for item in node.keywords}
    except (ValueError, TypeError):
        return None
    return {"name": name, "arguments": arguments}


def _jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _completed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (row["bfcl_id"], row["direction"])
        for row in _jsonl(path)
        if "bfcl_id" in row
    }


if __name__ == "__main__":
    main()
