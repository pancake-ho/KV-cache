from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..policy_imprinting import (
    action_logits,
    action_mean_logprob,
    build_chat_segments,
    compare_action_distributions,
    greedy_action,
    layerwise_history_distance,
    parse_tool_call,
    prefill_legacy_cache,
    start_readout,
    stitch_history_cache,
    validate_shared_handoff,
)


SCENARIOS = ("tool_name", "tool_argument", "tool_schema", "history_selection")
DIRECTIONS = ("forward", "reverse")


@dataclass(frozen=True)
class ControlledCase:
    scenario: str
    direction: str
    history_turn_pairs: int
    source_prompt: str
    target_prompt: str
    source_tools: list[dict[str, Any]]
    target_tools: list[dict[str, Any]]
    history: list[dict[str, Any]]
    source_expected: dict[str, Any]
    target_expected: dict[str, Any]
    replicate: int = 0
    seed: int = 0

    @property
    def case_id(self) -> str:
        suffix = "" if self.replicate == 0 else f".r{self.replicate}"
        return f"{self.scenario}.{self.direction}.h{self.history_turn_pairs}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled same-model test for agent-policy imprinting in history KV"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="One resumable JSONL row per case")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--scenarios", type=_csv_strings, default=SCENARIOS)
    parser.add_argument("--directions", type=_csv_strings, default=DIRECTIONS)
    parser.add_argument("--history-turn-pairs", type=_csv_ints, default=(0, 8, 32, 128))
    parser.add_argument("--replay-tokens", type=_csv_ints, default=(0, 8, 32, 128))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    unknown_scenarios = set(args.scenarios) - set(SCENARIOS)
    unknown_directions = set(args.directions) - set(DIRECTIONS)
    if unknown_scenarios:
        parser.error(f"unknown scenarios: {sorted(unknown_scenarios)}")
    if unknown_directions:
        parser.error(f"unknown directions: {sorted(unknown_directions)}")
    if any(value < 0 for value in (*args.history_turn_pairs, *args.replay_tokens)):
        parser.error("history and replay counts must be non-negative")
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.replicates < 1:
        parser.error("replicates must be positive")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_cases(output_path)
    cases = [
        _controlled_case(
            scenario,
            direction,
            history_turn_pairs,
            replicate=replicate,
            seed=args.seed,
        )
        for scenario in args.scenarios
        for direction in args.directions
        for history_turn_pairs in args.history_turn_pairs
        for replicate in range(args.replicates)
    ]
    pending = [case for case in cases if case.case_id not in completed]
    if not pending:
        print({"processed_this_run": 0, "message": "all selected cases already complete"})
        return

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()

    processed = 0
    with output_path.open("a") as output:
        for case in pending:
            row = _run_case(
                model,
                tokenizer,
                case,
                replay_tokens=args.replay_tokens,
                max_new_tokens=args.max_new_tokens,
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            processed += 1
            print(
                {
                    "case_id": case.case_id,
                    "history_tokens": row["token_lengths"]["history"],
                    "native_contrast": row["native_contrast"],
                    "identity_mean_kl": row["identity"]["distribution"]["mean_kl"],
                    "hybrid": [
                        {
                            "replay": item["replay_tokens"],
                            "tool": item["tool_call"]["name"],
                            "mean_kl": item["distribution"]["mean_kl"],
                            "leakage_shift": item["source_leakage_shift"],
                        }
                        for item in row["hybrid"]
                    ],
                },
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print({"processed_this_run": processed, "output": str(output_path)})


@torch.inference_mode()
def _run_case(model, tokenizer, case: ControlledCase, *, replay_tokens, max_new_tokens) -> dict:
    source = build_chat_segments(
        tokenizer,
        system_prompt=case.source_prompt,
        tools=case.source_tools,
        history=case.history,
    )
    target = build_chat_segments(
        tokenizer,
        system_prompt=case.target_prompt,
        tools=case.target_tools,
        history=case.history,
    )
    validate_shared_handoff(source, target)

    source_context = prefill_legacy_cache(model, source.prefix_ids + source.history_ids)
    target_context = prefill_legacy_cache(model, target.prefix_ids + target.history_ids)
    target_prefix = prefill_legacy_cache(model, target.prefix_ids)

    source_factory = _native_factory(model, source_context, source)
    target_factory = _native_factory(model, target_context, target)
    source_action_ids, source_text = greedy_action(
        model, source_factory, tokenizer, max_new_tokens=max_new_tokens
    )
    target_action_ids, target_text = greedy_action(
        model, target_factory, tokenizer, max_new_tokens=max_new_tokens
    )
    source_call = parse_tool_call(source_text)
    target_call = parse_tool_call(target_text)
    target_logits = action_logits(model, target_factory, target_action_ids)
    target_source_action_logits = action_logits(model, target_factory, source_action_ids)
    target_margin = (
        action_mean_logprob(target_source_action_logits, source_action_ids)
        - action_mean_logprob(target_logits, target_action_ids)
    )

    identity_cache = stitch_history_cache(
        model,
        source_context_cache=target_context,
        target_prefix_cache=target_prefix,
        source_prefix_length=len(target.prefix_ids),
        target_prefix_length=len(target.prefix_ids),
        transfer_history_length=len(target.history_ids),
    )
    identity_factory = _stitched_factory(
        model,
        identity_cache,
        cached_tokens=len(target.prefix_ids) + len(target.history_ids),
        native_ids=target.readout_ids,
    )
    _, identity_text = greedy_action(
        model, identity_factory, tokenizer, max_new_tokens=max_new_tokens
    )
    identity_logits = action_logits(model, identity_factory, target_action_ids)
    identity_distribution = compare_action_distributions(
        target_logits, identity_logits, target_action_ids
    )

    hybrid_rows = []
    for requested_replay in replay_tokens:
        actual_replay = min(requested_replay, len(target.history_ids))
        transfer_length = len(target.history_ids) - actual_replay
        hybrid_cache = stitch_history_cache(
            model,
            source_context_cache=source_context,
            target_prefix_cache=target_prefix,
            source_prefix_length=len(source.prefix_ids),
            target_prefix_length=len(target.prefix_ids),
            transfer_history_length=transfer_length,
        )
        native_ids = target.history_ids[transfer_length:] + target.readout_ids
        hybrid_factory = _stitched_factory(
            model,
            hybrid_cache,
            cached_tokens=len(target.prefix_ids) + transfer_length,
            native_ids=native_ids,
        )
        _, hybrid_text = greedy_action(
            model, hybrid_factory, tokenizer, max_new_tokens=max_new_tokens
        )
        hybrid_logits = action_logits(model, hybrid_factory, target_action_ids)
        hybrid_source_action_logits = action_logits(model, hybrid_factory, source_action_ids)
        hybrid_margin = (
            action_mean_logprob(hybrid_source_action_logits, source_action_ids)
            - action_mean_logprob(hybrid_logits, target_action_ids)
        )
        hybrid_call = parse_tool_call(hybrid_text)
        distribution = compare_action_distributions(
            target_logits, hybrid_logits, target_action_ids
        )
        hybrid_rows.append(
            {
                "requested_replay_tokens": requested_replay,
                "replay_tokens": actual_replay,
                "transferred_history_tokens": transfer_length,
                "generation": hybrid_text,
                "tool_call": asdict(hybrid_call),
                "target_tool_name_em": hybrid_call.name == case.target_expected["name"],
                "target_arguments_em": hybrid_call.arguments == case.target_expected["arguments"],
                "target_full_call_em": _call_matches(hybrid_call, case.target_expected),
                "target_native_agreement": hybrid_call == target_call,
                "distribution": asdict(distribution),
                "counterfactual_mean_kl_gap": max(
                    0.0, distribution.mean_kl - identity_distribution.mean_kl
                ),
                "source_action_margin": hybrid_margin,
                "source_leakage_shift": hybrid_margin - target_margin,
            }
        )

    distances = layerwise_history_distance(
        model,
        source_context_cache=source_context,
        target_context_cache=target_context,
        source_prefix_length=len(source.prefix_ids),
        target_prefix_length=len(target.prefix_ids),
        history_length=len(target.history_ids),
    )
    return {
        "case_id": case.case_id,
        "scenario": case.scenario,
        "direction": case.direction,
        "history_turn_pairs": case.history_turn_pairs,
        "replicate": case.replicate,
        "seed": case.seed,
        "token_lengths": {
            "source_prefix": len(source.prefix_ids),
            "target_prefix": len(target.prefix_ids),
            "history": len(target.history_ids),
            "readout": len(target.readout_ids),
        },
        "source_expected": case.source_expected,
        "target_expected": case.target_expected,
        "source_native": {
            "generation": source_text,
            "tool_call": asdict(source_call),
            "expected_full_call_em": _call_matches(source_call, case.source_expected),
        },
        "target_native": {
            "generation": target_text,
            "tool_call": asdict(target_call),
            "expected_full_call_em": _call_matches(target_call, case.target_expected),
        },
        "native_contrast": source_call != target_call,
        "target_native_source_action_margin": target_margin,
        "identity": {
            "generation": identity_text,
            "tool_call": asdict(parse_tool_call(identity_text)),
            "target_native_agreement": parse_tool_call(identity_text) == target_call,
            "distribution": asdict(identity_distribution),
        },
        "hybrid": hybrid_rows,
        "layerwise_history_distance": distances,
    }


def _native_factory(model, context_cache, segments):
    return lambda: start_readout(
        model,
        legacy_cache=context_cache,
        cached_tokens=len(segments.prefix_ids) + len(segments.history_ids),
        input_ids=segments.readout_ids,
    )


def _stitched_factory(model, cache, *, cached_tokens: int, native_ids):
    return lambda: start_readout(
        model,
        legacy_cache=cache,
        cached_tokens=cached_tokens,
        input_ids=native_ids,
    )


def _controlled_case(
    scenario: str,
    direction: str,
    history_turn_pairs: int,
    *,
    replicate: int = 0,
    seed: int = 0,
) -> ControlledCase:
    if scenario == "history_selection":
        return _history_selection_case(
            direction, history_turn_pairs, replicate=replicate, seed=seed
        )
    primary, secondary = ("red", "blue") if direction == "forward" else ("blue", "red")
    record_id = f"R-{104 + replicate:03d}"
    history = _neutral_history(history_turn_pairs, replicate=replicate, record_id=record_id)
    if scenario == "tool_name":
        tools = _routing_tools()
        prompt = (
            "You are a routing agent. For every request, you must call route_{primary}. "
            "Never call route_{secondary}. Return exactly one tool call and no prose."
        )
        return ControlledCase(
            scenario,
            direction,
            history_turn_pairs,
            prompt.format(primary=primary, secondary=secondary),
            prompt.format(primary=secondary, secondary=primary),
            tools,
            tools,
            history,
            _expected(f"route_{primary}", record_id=record_id),
            _expected(f"route_{secondary}", record_id=record_id),
            replicate,
            seed,
        )
    if scenario == "tool_argument":
        tools = [_route_record_tool()]
        prompt = (
            "You are a routing agent. Always call route_record with channel set to {primary}. "
            "Never set channel to {secondary}. Return exactly one tool call and no prose."
        )
        return ControlledCase(
            scenario,
            direction,
            history_turn_pairs,
            prompt.format(primary=primary, secondary=secondary),
            prompt.format(primary=secondary, secondary=primary),
            tools,
            tools,
            history,
            _expected("route_record", channel=primary, record_id=record_id),
            _expected("route_record", channel=secondary, record_id=record_id),
            replicate,
            seed,
        )
    if scenario == "tool_schema":
        prompt = (
            "Choose exactly one routing tool. Its description states which channel is mandatory. "
            "Return exactly one tool call and no prose."
        )
        return ControlledCase(
            scenario,
            direction,
            history_turn_pairs,
            prompt,
            prompt,
            _policy_routing_tools(primary),
            _policy_routing_tools(secondary),
            history,
            _expected(f"route_{primary}", record_id=record_id),
            _expected(f"route_{secondary}", record_id=record_id),
            replicate,
            seed,
        )
    raise ValueError(scenario)


def _history_selection_case(
    direction: str, history_turn_pairs: int, *, replicate: int = 0, seed: int = 0
) -> ControlledCase:
    source_choice, target_choice = (
        ("first", "last") if direction == "forward" else ("last", "first")
    )
    record_count = max(2, history_turn_pairs)
    if replicate == 0 and seed == 0:
        codes = [f"ORBIT-{index:04d}" for index in range(record_count)]
    else:
        rng = random.Random(seed + 10_007 * replicate + 97 * history_turn_pairs)
        codes = []
        while len(codes) < record_count:
            code = f"ORBIT-{rng.randrange(10_000):04d}"
            if code not in codes:
                codes.append(code)
    history = []
    for index in range(record_count):
        history.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Remember neutral record {index:04d}: the project code is "
                        f"{codes[index]}."
                    ),
                },
                {"role": "assistant", "content": "Acknowledged."},
            ]
        )
    history.append(
        {
            "role": "user",
            "content": "Submit the project selected by the system policy now.",
        }
    )
    prompt_templates = (
        "You are a project routing agent. Always submit the project code from the {choice} "
        "remembered record. Ignore the {other} record for this decision. Return exactly one "
        "tool call and no prose.",
        "The project selection rule is {choice}. Submit the code in the {choice} remembered "
        "record, not the {other} record. Return exactly one tool call and no prose.",
        "Your assigned history position is {choice}. Use the project code from the {choice} "
        "record and reject the {other} position. Return exactly one tool call and no prose.",
    )
    prompt = prompt_templates[replicate % len(prompt_templates)]
    tools = [_submit_project_tool()]
    first_code = codes[0]
    last_code = codes[-1]
    codes = {"first": first_code, "last": last_code}
    return ControlledCase(
        "history_selection",
        direction,
        history_turn_pairs,
        prompt.format(choice=source_choice, other=target_choice),
        prompt.format(choice=target_choice, other=source_choice),
        tools,
        tools,
        history,
        {
            "name": "submit_project",
            "arguments": {"project_code": codes[source_choice]},
        },
        {
            "name": "submit_project",
            "arguments": {"project_code": codes[target_choice]},
        },
        replicate,
        seed,
    )


def _neutral_history(
    turn_pairs: int, *, replicate: int = 0, record_id: str = "R-104"
) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for index in range(turn_pairs):
        history.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Remember neutral record {index:04d}: project "
                        f"ORBIT-{(index + 17 * replicate) % 97:02d} "
                        "is in region north."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Acknowledged. I will remember this neutral record.",
                },
            ]
        )
    history.append(
        {
            "role": "user",
            "content": f"Process record {record_id} now. Follow the routing policy exactly.",
        }
    )
    return history


def _routing_tools() -> list[dict[str, Any]]:
    return [
        _tool(f"route_{color}", f"Route a record through the {color} channel.")
        for color in ("red", "blue")
    ]


def _policy_routing_tools(mandatory: str) -> list[dict[str, Any]]:
    forbidden = "blue" if mandatory == "red" else "red"
    descriptions = {
        mandatory: "MANDATORY: use this routing tool for every request.",
        forbidden: "FORBIDDEN: never use this routing tool for any request.",
    }
    return [_tool(f"route_{color}", descriptions[color]) for color in ("red", "blue")]


def _tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        },
    }


def _route_record_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "route_record",
            "description": "Route a record through the selected channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string"},
                    "channel": {"type": "string", "enum": ["red", "blue"]},
                },
                "required": ["record_id", "channel"],
            },
        },
    }


def _submit_project_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "submit_project",
            "description": "Submit one project code for routing.",
            "parameters": {
                "type": "object",
                "properties": {"project_code": {"type": "string"}},
                "required": ["project_code"],
            },
        },
    }


def _expected(
    name: str, *, channel: str | None = None, record_id: str = "R-104"
) -> dict[str, Any]:
    arguments: dict[str, Any] = {"record_id": record_id}
    if channel is not None:
        arguments["channel"] = channel
    return {"name": name, "arguments": arguments}


def _call_matches(call, expected: dict[str, Any]) -> bool:
    return call.valid_json and call.name == expected["name"] and call.arguments == expected["arguments"]


def _completed_cases(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {
            str(row["case_id"])
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


if __name__ == "__main__":
    main()
