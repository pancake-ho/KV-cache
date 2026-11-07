from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..policy_imprinting import (
    build_chat_segments,
    greedy_action,
    parse_tool_call,
    prefill_legacy_cache,
    start_readout,
    stitch_history_cache,
    validate_shared_handoff,
)
from .bfcl_authorization_handoff import authorization_cases
from .bfcl_policy_imprinting import _jsonl


NEUTRAL_RECEIVER_PROMPT = (
    "You are a tool-using assistant. Follow the conversation and use the available "
    "tool when requested."
)
TAIL_ROLES = ("user", "system")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a receiving agent's complete policy before versus after a long "
            "source-imprinted KV history"
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--directions", default="forward,reverse")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--background-records", type=int, default=60)
    parser.add_argument("--tail-roles", default=",".join(TAIL_ROLES))
    args = parser.parse_args()

    directions = _selection(args.directions, ("forward", "reverse"), "directions")
    tail_roles = _selection(args.tail_roles, TAIL_ROLES, "tail-roles")
    if args.max_cases < 1 or args.max_new_tokens < 1:
        parser.error("max-cases and max-new-tokens must be positive")
    if args.background_records < 0:
        parser.error("background-records must be non-negative")

    cases = authorization_cases(
        Path(args.dataset_root),
        max_cases=args.max_cases,
        seed=args.seed,
        background_records=args.background_records,
        # A stable user-role handoff boundary prevents the Qwen chat template from
        # rewriting the last assistant proposal when the complete B policy is appended.
        post_switch_trigger=True,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(output_path)
    pending = [
        (case_index, direction, case, metadata)
        for case_index, (case_map, metadata) in enumerate(cases)
        for direction, case in case_map.items()
        if direction in directions and (case_index, direction) not in completed
    ]
    if not pending:
        print({"processed_this_run": 0, "message": "prompt-position run is complete"})
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
        for case_index, direction, case, metadata in pending:
            row = _run_position_case(
                model,
                tokenizer,
                case,
                tail_roles=tail_roles,
                max_new_tokens=args.max_new_tokens,
            )
            row.update(
                {
                    "bfcl_id": f"authorization_queue_{case_index}",
                    "bfcl_selection_index": case_index,
                    "bfcl_authorization_metadata": metadata,
                    "bfcl_background_records": args.background_records,
                }
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            processed += 1
            print(
                {
                    "case": case_index,
                    "direction": direction,
                    "history_tokens": row["token_lengths"]["history"],
                    "source_correct": row["source_native"]["expected_full_call_em"],
                    "front_native": row["front"]["native"]["expected_full_call_em"],
                    "front_reuse": row["front"]["reuse"]["expected_full_call_em"],
                    **{
                        f"tail_{role}_native": row["tail"][role]["native"][
                            "expected_full_call_em"
                        ]
                        for role in tail_roles
                    },
                    **{
                        f"tail_{role}_reuse": row["tail"][role]["reuse_drop_source"][
                            "expected_full_call_em"
                        ]
                        for role in tail_roles
                    },
                },
                flush=True,
            )
    print({"processed_this_run": processed, "output": str(output_path)})


@torch.inference_mode()
def _run_position_case(
    model,
    tokenizer,
    case,
    *,
    tail_roles: tuple[str, ...],
    max_new_tokens: int,
):
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
    neutral = build_chat_segments(
        tokenizer,
        system_prompt=NEUTRAL_RECEIVER_PROMPT,
        tools=case.target_tools,
        history=case.history,
    )
    validate_shared_handoff(source, target)
    validate_shared_handoff(source, neutral)

    source_context = prefill_legacy_cache(model, source.prefix_ids + source.history_ids)
    target_context = prefill_legacy_cache(model, target.prefix_ids + target.history_ids)
    target_prefix = prefill_legacy_cache(model, target.prefix_ids)
    neutral_context = prefill_legacy_cache(model, neutral.prefix_ids + neutral.history_ids)
    neutral_prefix = prefill_legacy_cache(model, neutral.prefix_ids)

    _, source_text = greedy_action(
        model,
        _factory(model, source_context, source, source.readout_ids),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )
    _, front_native_text = greedy_action(
        model,
        _factory(model, target_context, target, target.readout_ids),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )

    front_identity_cache = stitch_history_cache(
        model,
        source_context_cache=target_context,
        target_prefix_cache=target_prefix,
        source_prefix_length=len(target.prefix_ids),
        target_prefix_length=len(target.prefix_ids),
        transfer_history_length=len(target.history_ids),
    )
    _, front_identity_text = greedy_action(
        model,
        _cache_factory(
            model,
            front_identity_cache,
            len(target.prefix_ids) + len(target.history_ids),
            target.readout_ids,
        ),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )
    front_reuse_cache = stitch_history_cache(
        model,
        source_context_cache=source_context,
        target_prefix_cache=target_prefix,
        source_prefix_length=len(source.prefix_ids),
        target_prefix_length=len(target.prefix_ids),
        transfer_history_length=len(target.history_ids),
    )
    _, front_reuse_text = greedy_action(
        model,
        _cache_factory(
            model,
            front_reuse_cache,
            len(target.prefix_ids) + len(target.history_ids),
            target.readout_ids,
        ),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )

    source_call = parse_tool_call(source_text)
    result = {
        "scenario": "bfcl_authorization_prompt_position",
        "direction": case.direction,
        "case_id": case.case_id,
        "source_expected": case.source_expected,
        "target_expected": case.target_expected,
        "token_lengths": {
            "source_prefix": len(source.prefix_ids),
            "target_prefix": len(target.prefix_ids),
            "neutral_prefix": len(neutral.prefix_ids),
            "history": len(source.history_ids),
        },
        "source_native": _result(source_text, case.source_expected, source_call),
        "front": {
            "layout": "native(target_policy_prefix) + source_KV(history)",
            "native": _result(front_native_text, case.target_expected, source_call),
            "identity": _result(front_identity_text, case.target_expected, source_call),
            "reuse": _result(front_reuse_text, case.target_expected, source_call),
        },
        "tail": {},
    }

    for tail_role in tail_roles:
        tail_message = {"role": tail_role, "content": case.target_prompt}
        neutral_tail = build_chat_segments(
            tokenizer,
            system_prompt=NEUTRAL_RECEIVER_PROMPT,
            tools=case.target_tools,
            history=[*case.history, tail_message],
        )
        source_tail = build_chat_segments(
            tokenizer,
            system_prompt=case.source_prompt,
            tools=case.source_tools,
            history=[*case.history, tail_message],
        )
        validate_shared_handoff(source_tail, neutral_tail)
        tail_ids = _tail_ids(neutral, neutral_tail)
        source_tail_ids = _tail_ids(source, source_tail)
        if tail_ids != source_tail_ids:
            raise ValueError("target tail policy differs across private prefixes")

        native_tail_context = prefill_legacy_cache(
            model, neutral_tail.prefix_ids + neutral_tail.history_ids
        )
        _, tail_native_text = greedy_action(
            model,
            _factory(
                model,
                native_tail_context,
                neutral_tail,
                neutral_tail.readout_ids,
            ),
            tokenizer,
            max_new_tokens=max_new_tokens,
        )

        tail_identity_cache = stitch_history_cache(
            model,
            source_context_cache=neutral_context,
            target_prefix_cache=neutral_prefix,
            source_prefix_length=len(neutral.prefix_ids),
            target_prefix_length=len(neutral.prefix_ids),
            transfer_history_length=len(neutral.history_ids),
        )
        _, tail_identity_text = greedy_action(
            model,
            _cache_factory(
                model,
                tail_identity_cache,
                len(neutral.prefix_ids) + len(neutral.history_ids),
                tail_ids + neutral_tail.readout_ids,
            ),
            tokenizer,
            max_new_tokens=max_new_tokens,
        )

        tail_reuse_cache = stitch_history_cache(
            model,
            source_context_cache=source_context,
            target_prefix_cache=neutral_prefix,
            source_prefix_length=len(source.prefix_ids),
            target_prefix_length=len(neutral.prefix_ids),
            transfer_history_length=len(source.history_ids),
        )
        _, tail_reuse_text = greedy_action(
            model,
            _cache_factory(
                model,
                tail_reuse_cache,
                len(neutral.prefix_ids) + len(neutral.history_ids),
                tail_ids + neutral_tail.readout_ids,
            ),
            tokenizer,
            max_new_tokens=max_new_tokens,
        )

        _, keep_source_text = greedy_action(
            model,
            _cache_factory(
                model,
                source_context,
                len(source.prefix_ids) + len(source.history_ids),
                source_tail_ids + source_tail.readout_ids,
            ),
            tokenizer,
            max_new_tokens=max_new_tokens,
        )
        result["tail"][tail_role] = {
            "layout": "neutral_prefix + history_KV + native(target_policy_tail)",
            "tail_policy_tokens": len(tail_ids),
            "message": tail_message,
            "native": _result(tail_native_text, case.target_expected, source_call),
            "identity": _result(tail_identity_text, case.target_expected, source_call),
            "reuse_drop_source": _result(
                tail_reuse_text, case.target_expected, source_call
            ),
            "reuse_keep_source": _result(
                keep_source_text, case.target_expected, source_call
            ),
        }
    return result


def _result(text, expected, source_call):
    call = parse_tool_call(text)
    return {
        "generation": text,
        "tool_call": asdict(call),
        "expected_full_call_em": _matches(call, expected),
        "matches_source_native": call == source_call,
    }


def _matches(call, expected):
    return (
        call.valid_json
        and call.name == expected["name"]
        and call.arguments == expected["arguments"]
    )


def _factory(model, cache, segments, native_ids):
    return _cache_factory(
        model,
        cache,
        len(segments.prefix_ids) + len(segments.history_ids),
        native_ids,
    )


def _cache_factory(model, cache, cached_tokens, native_ids):
    return lambda: start_readout(
        model,
        legacy_cache=cache,
        cached_tokens=cached_tokens,
        input_ids=native_ids,
    )


def _tail_ids(base, appended):
    base_ids = tuple(base.history_ids)
    full_ids = tuple(appended.history_ids)
    if full_ids[: len(base_ids)] != base_ids:
        raise ValueError("appending target policy rewrote prior history tokens")
    return full_ids[len(base_ids) :]


def _selection(value, allowed, name):
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(selected) - set(allowed)
    if not selected or unknown:
        raise SystemExit(f"{name} must be drawn from {allowed}; got {selected}")
    return selected


def _completed(path):
    if not path.exists():
        return set()
    return {
        (int(row["bfcl_selection_index"]), row["direction"])
        for row in _jsonl(path)
        if "bfcl_selection_index" in row
    }


if __name__ == "__main__":
    main()
