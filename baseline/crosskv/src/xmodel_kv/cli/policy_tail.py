from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..policy_imprinting import (
    action_logits,
    action_mean_logprob,
    build_chat_segments,
    compare_action_distributions,
    greedy_action,
    history_message_spans,
    parse_tool_call,
    prefill_legacy_cache,
    start_readout,
    stitch_history_cache,
    validate_shared_handoff,
)
from .policy_triad import (
    FAMILIES,
    LAYOUTS,
    _directions,
    _length_match_group,
    _shift_prompt,
    _triad_group,
)


TAIL_STYLES = ("plain", "override")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test a receiving-agent policy appended after source-imprinted history"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    parser.add_argument("--replicates", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--record-count", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--tail-style", choices=TAIL_STYLES, default="override")
    parser.add_argument("--shift-padding-tokens", type=int, default=32)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    families = _csv(args.families, FAMILIES, "families")
    layouts = _csv(args.layouts, LAYOUTS, "layouts")
    if args.replicates < 1 or args.max_new_tokens < 1:
        parser.error("replicates and max-new-tokens must be positive")
    if args.record_count < 8:
        parser.error("record-count must be at least 8")
    if args.shift_padding_tokens < 1:
        parser.error("shift-padding-tokens must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num-shards >= 1 and 0 <= shard-index < num-shards")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prepared = []
    group_index = 0
    for family in families:
        for direction in _directions(family):
            for replicate in range(args.replicates):
                group = _triad_group(
                    family,
                    direction,
                    replicate=replicate,
                    seed=args.seed,
                    record_count=args.record_count,
                )
                if group_index % args.num_shards == args.shard_index:
                    clean_cases, _ = _length_match_group(tokenizer, group)
                    for layout in layouts:
                        prepared.append((group, clean_cases, layout))
                group_index += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(output_path)
    pending = [
        item
        for item in prepared
        if _tail_id(item[0], item[2], args.tail_style) not in completed
    ]
    if not pending:
        print({"processed_this_run": 0, "message": "selected tail-policy shard is complete"})
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    processed = 0
    with output_path.open("a") as output:
        for group, clean_cases, layout in pending:
            row = _run_tail_group(
                model,
                tokenizer,
                group,
                clean_cases,
                layout=layout,
                tail_style=args.tail_style,
                shift_padding_tokens=args.shift_padding_tokens,
                max_new_tokens=args.max_new_tokens,
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            processed += 1
            print(
                {
                    "tail_id": row["tail_id"],
                    "shift": row["token_lengths"]["source_receiver_prefix_shift"],
                    "A_front": _record(row["front"]["target_native"]),
                    "S_front": _record(row["front"]["source_native"]),
                    "B_front": _record(row["front"]["hybrid"]),
                    "A_tail": _record(row["tail"]["native"]),
                    "B_tail": _record(row["tail"]["stitched"]),
                    "keep_source_tail": _record(row["tail"]["keep_source_prefix"]),
                },
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(
        {
            "processed_this_run": processed,
            "shard": f"{args.shard_index}/{args.num_shards}",
            "output": str(output_path),
        }
    )


@torch.inference_mode()
def _run_tail_group(
    model,
    tokenizer,
    group,
    clean_cases,
    *,
    layout: str,
    tail_style: str,
    shift_padding_tokens: int,
    max_new_tokens: int,
):
    third = clean_cases["third"]
    neutral = clean_cases["null"]
    source_prompt = third.source_prompt
    if layout == "shifted":
        source_prompt = _shift_prompt(source_prompt, shift_padding_tokens)
    target_prompt = third.target_prompt
    receiver_prompt = neutral.source_prompt
    tools = third.target_tools
    full_history = third.history
    base_history = full_history[:-1]
    generic_query = full_history[-1]
    tail_query = _tail_message(
        group.cases["third"].target_prompt, style=tail_style
    )

    front_source = build_chat_segments(
        tokenizer,
        system_prompt=source_prompt,
        tools=tools,
        history=full_history,
    )
    front_target = build_chat_segments(
        tokenizer,
        system_prompt=target_prompt,
        tools=tools,
        history=full_history,
    )
    validate_shared_handoff(front_source, front_target)

    receiver_tail = build_chat_segments(
        tokenizer,
        system_prompt=receiver_prompt,
        tools=tools,
        history=[*base_history, tail_query],
    )
    source_tail = build_chat_segments(
        tokenizer,
        system_prompt=source_prompt,
        tools=tools,
        history=[*base_history, tail_query],
    )
    validate_shared_handoff(source_tail, receiver_tail)
    spans = history_message_spans(
        tokenizer,
        system_prompt=receiver_prompt,
        tools=tools,
        history=[*base_history, tail_query],
    )
    tail_start = spans[-1][0]
    base_history_ids = receiver_tail.history_ids[:tail_start]
    tail_ids = receiver_tail.history_ids[tail_start:]
    source_base_history_ids = source_tail.history_ids[:tail_start]
    source_tail_ids = source_tail.history_ids[tail_start:]
    if base_history_ids != source_base_history_ids or tail_ids != source_tail_ids:
        raise ValueError("late target instruction differs outside the private prefix")
    generic_query_ids = _suffix(front_target.history_ids, base_history_ids)

    front_source_cache = prefill_legacy_cache(
        model, front_source.prefix_ids + front_source.history_ids
    )
    front_target_cache = prefill_legacy_cache(
        model, front_target.prefix_ids + front_target.history_ids
    )
    front_target_prefix_cache = prefill_legacy_cache(model, front_target.prefix_ids)
    source_base_cache = prefill_legacy_cache(
        model, source_tail.prefix_ids + source_base_history_ids
    )
    receiver_base_cache = prefill_legacy_cache(
        model, receiver_tail.prefix_ids + base_history_ids
    )
    receiver_prefix_cache = prefill_legacy_cache(model, receiver_tail.prefix_ids)
    receiver_tail_cache = prefill_legacy_cache(
        model, receiver_tail.prefix_ids + receiver_tail.history_ids
    )

    front_source_factory = _factory(
        model,
        front_source_cache,
        len(front_source.prefix_ids) + len(front_source.history_ids),
        front_source.readout_ids,
    )
    front_target_factory = _factory(
        model,
        front_target_cache,
        len(front_target.prefix_ids) + len(front_target.history_ids),
        front_target.readout_ids,
    )
    source_ids, source_text = greedy_action(
        model, front_source_factory, tokenizer, max_new_tokens=max_new_tokens
    )
    front_target_ids, front_target_text = greedy_action(
        model, front_target_factory, tokenizer, max_new_tokens=max_new_tokens
    )

    front_hybrid_cache = stitch_history_cache(
        model,
        source_context_cache=front_source_cache,
        target_prefix_cache=front_target_prefix_cache,
        source_prefix_length=len(front_source.prefix_ids),
        target_prefix_length=len(front_target.prefix_ids),
        transfer_history_length=len(front_target.history_ids),
    )
    front_hybrid_factory = _factory(
        model,
        front_hybrid_cache,
        len(front_target.prefix_ids) + len(front_target.history_ids),
        front_target.readout_ids,
    )
    _, front_hybrid_text = greedy_action(
        model, front_hybrid_factory, tokenizer, max_new_tokens=max_new_tokens
    )

    tail_native_factory = _factory(
        model,
        receiver_tail_cache,
        len(receiver_tail.prefix_ids) + len(receiver_tail.history_ids),
        receiver_tail.readout_ids,
    )
    tail_target_ids, tail_native_text = greedy_action(
        model, tail_native_factory, tokenizer, max_new_tokens=max_new_tokens
    )
    tail_target_logits = action_logits(model, tail_native_factory, tail_target_ids)
    tail_source_logits = action_logits(model, tail_native_factory, source_ids)
    tail_native_margin = _margin(
        tail_target_logits, tail_source_logits, tail_target_ids, source_ids
    )

    tail_stitched_cache = stitch_history_cache(
        model,
        source_context_cache=source_base_cache,
        target_prefix_cache=receiver_prefix_cache,
        source_prefix_length=len(source_tail.prefix_ids),
        target_prefix_length=len(receiver_tail.prefix_ids),
        transfer_history_length=len(base_history_ids),
    )
    tail_stitched_factory = _factory(
        model,
        tail_stitched_cache,
        len(receiver_tail.prefix_ids) + len(base_history_ids),
        tail_ids + receiver_tail.readout_ids,
    )
    _, tail_stitched_text = greedy_action(
        model, tail_stitched_factory, tokenizer, max_new_tokens=max_new_tokens
    )

    tail_identity_cache = stitch_history_cache(
        model,
        source_context_cache=receiver_base_cache,
        target_prefix_cache=receiver_prefix_cache,
        source_prefix_length=len(receiver_tail.prefix_ids),
        target_prefix_length=len(receiver_tail.prefix_ids),
        transfer_history_length=len(base_history_ids),
    )
    tail_identity_factory = _factory(
        model,
        tail_identity_cache,
        len(receiver_tail.prefix_ids) + len(base_history_ids),
        tail_ids + receiver_tail.readout_ids,
    )
    _, tail_identity_text = greedy_action(
        model, tail_identity_factory, tokenizer, max_new_tokens=max_new_tokens
    )

    keep_source_factory = _factory(
        model,
        source_base_cache,
        len(source_tail.prefix_ids) + len(source_base_history_ids),
        source_tail_ids + source_tail.readout_ids,
    )
    _, keep_source_text = greedy_action(
        model, keep_source_factory, tokenizer, max_new_tokens=max_new_tokens
    )

    front_target_call = parse_tool_call(front_target_text)
    source_call = parse_tool_call(source_text)
    tail_native_call = parse_tool_call(tail_native_text)
    front_hybrid = _result(front_hybrid_text, front_target_call, source_call)
    tail_stitched = _tail_result(
        model,
        tail_stitched_factory,
        tail_stitched_text,
        tail_target_ids,
        source_ids,
        tail_target_logits,
        tail_native_margin,
        tail_native_call,
        source_call,
    )
    keep_source = _tail_result(
        model,
        keep_source_factory,
        keep_source_text,
        tail_target_ids,
        source_ids,
        tail_target_logits,
        tail_native_margin,
        tail_native_call,
        source_call,
    )
    return {
        "tail_id": _tail_id(group, layout, tail_style),
        "triad": {
            "family": group.family,
            "direction": group.direction,
            "layout": layout,
            "replicate": group.replicate,
            "seed": group.seed,
            "record_count": len(group.record_ids),
            "chance_level": 1.0 / len(group.record_ids),
            "target_record_id": group.target_record_id,
            "source_record_id": group.third_record_id,
            "target_policy": group.target_policy,
            "source_policy": group.third_policy,
            "tail_style": tail_style,
            **group.metadata,
        },
        "token_lengths": {
            "source_prefix": len(source_tail.prefix_ids),
            "receiver_prefix": len(receiver_tail.prefix_ids),
            "source_receiver_prefix_shift": (
                len(source_tail.prefix_ids) - len(receiver_tail.prefix_ids)
            ),
            "base_history": len(base_history_ids),
            "generic_query": len(generic_query_ids),
            "tail_policy_query": len(tail_ids),
        },
        "front": {
            "target_native": _result(front_target_text, front_target_call, source_call),
            "source_native": _result(source_text, front_target_call, source_call),
            "hybrid": front_hybrid,
        },
        "tail": {
            "native": _result(tail_native_text, tail_native_call, source_call),
            "identity": _result(tail_identity_text, tail_native_call, source_call),
            "stitched": tail_stitched,
            "keep_source_prefix": keep_source,
        },
        "validity": {
            "front_target_correct": _expected(front_target_call, group.target_record_id),
            "front_source_correct": _expected(source_call, group.third_record_id),
            "tail_native_correct": _expected(tail_native_call, group.target_record_id),
        },
        "messages": {
            "generic_query": generic_query,
            "tail_query": tail_query,
        },
    }


def _tail_result(
    model,
    factory,
    text,
    target_ids,
    source_ids,
    target_logits,
    target_margin,
    target_call,
    source_call,
):
    candidate_logits = action_logits(model, factory, target_ids)
    candidate_source_logits = action_logits(model, factory, source_ids)
    candidate_margin = _margin(
        candidate_logits, candidate_source_logits, target_ids, source_ids
    )
    result = _result(text, target_call, source_call)
    result["distribution"] = asdict(
        compare_action_distributions(target_logits, candidate_logits, target_ids)
    )
    result["source_leakage_shift"] = candidate_margin - target_margin
    return result


def _result(text, target_call, source_call):
    call = parse_tool_call(text)
    return {
        "generation": text,
        "tool_call": asdict(call),
        "matches_target_native": call == target_call,
        "matches_source_native": call == source_call,
    }


def _margin(target_logits, source_logits, target_ids, source_ids):
    return action_mean_logprob(source_logits, source_ids) - action_mean_logprob(
        target_logits, target_ids
    )


def _expected(call, record_id):
    return (
        call.valid_json
        and call.name == "choose_record"
        and call.arguments == {"record_id": record_id}
    )


def _tail_message(target_policy: str, *, style: str = "override"):
    if style == "plain":
        introduction = "The batch is complete.\n"
    elif style == "override":
        introduction = (
            "The batch is complete. This is the receiving agent's final handoff policy; "
            "it overrides any earlier selection rule.\n"
        )
    else:
        raise ValueError(f"unknown tail style: {style}")
    return {
        "role": "user",
        "content": introduction + target_policy,
    }


def _suffix(full, prefix):
    if tuple(full[: len(prefix)]) != tuple(prefix):
        raise ValueError("appending the tail instruction rewrote earlier history tokens")
    return tuple(full[len(prefix) :])


def _factory(model, cache, cached_tokens, native_ids):
    return lambda: start_readout(
        model,
        legacy_cache=cache,
        cached_tokens=cached_tokens,
        input_ids=native_ids,
    )


def _tail_id(group, layout, style="override"):
    suffix = "" if style == "override" else f".{style}"
    return f"tail.{group.family}.{group.direction}.r{group.replicate}.{layout}{suffix}"


def _record(result):
    arguments = result["tool_call"].get("arguments")
    return arguments.get("record_id") if isinstance(arguments, dict) else None


def _csv(value, allowed, name):
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(selected) - set(allowed)
    if not selected or unknown:
        raise SystemExit(f"invalid {name}: {sorted(unknown)}")
    return selected


def _completed(path: Path):
    if not path.exists():
        return set()
    with path.open() as handle:
        return {
            row["tail_id"]
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
            if "tail_id" in row
        }


if __name__ == "__main__":
    main()
