from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..diverse_handoff import load_manifest
from ..policy_imprinting import (
    ToolCall,
    action_logits,
    build_chat_segments,
    greedy_action,
    parse_tool_call,
    prefill_legacy_cache,
    start_readout,
    stitch_history_cache,
    validate_shared_handoff,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen Agent-KV V2 cases with source/target native prefills"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    if args.max_new_tokens < 1 or args.num_shards < 1:
        parser.error("max-new-tokens and num-shards must be positive")

    rows = [
        row
        for index, row in enumerate(load_manifest(Path(args.manifest)))
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        row["id"]
        for row in _jsonl(output_path)
    } if output_path.exists() else set()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    counts = {"processed": 0, "eligible": 0}
    with output_path.open("a") as output:
        for row in rows:
            if row["id"] in completed:
                continue
            result = _screen_one(
                model, tokenizer, row, max_new_tokens=args.max_new_tokens
            )
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            counts["processed"] += 1
            counts["eligible"] += int(result["eligible"])
            print(
                {
                    "id": row["id"],
                    "history_tokens": row["history_tokens"],
                    "source_correct": result["source_native"]["compatible_em"],
                    "target_correct": result["target_native"]["compatible_em"],
                    "eligible": result["eligible"],
                    "running_eligible": counts["eligible"],
                },
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(counts, flush=True)


@torch.inference_mode()
def _screen_one(model, tokenizer, row, *, max_new_tokens):
    source = build_chat_segments(
        tokenizer,
        system_prompt=row["source_prompt"],
        tools=row["source_tools"],
        history=row["history"],
    )
    target = build_chat_segments(
        tokenizer,
        system_prompt=row["target_prompt"],
        tools=row["target_tools"],
        history=row["history"],
    )
    validate_shared_handoff(source, target)
    source_cache = prefill_legacy_cache(model, source.prefix_ids + source.history_ids)
    source_ids, source_text = greedy_action(
        model,
        _factory(model, source_cache, source),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )
    source_call = parse_tool_call(source_text)
    source_score = score_bfcl_call(
        source_call,
        expected=row["source_expected"],
        argument_options=row["source_argument_options"],
    )
    target_cache = prefill_legacy_cache(model, target.prefix_ids + target.history_ids)
    target_ids, target_text = greedy_action(
        model,
        _factory(model, target_cache, target),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )
    target_call = parse_tool_call(target_text)
    target_score = score_bfcl_call(
        target_call,
        expected=row["target_expected"],
        argument_options=row["target_argument_options"],
    )
    teacher_logits = action_logits(
        model, _factory(model, target_cache, target), target_ids
    ).float()
    teacher_log_probs = teacher_logits.log_softmax(dim=-1)
    topk_log_probs, topk_indices = teacher_log_probs.topk(
        k=min(32, teacher_log_probs.shape[-1]), dim=-1
    )
    topk_mass = topk_log_probs.exp().sum(dim=-1)
    target_prefix_cache = prefill_legacy_cache(model, target.prefix_ids)
    stitched = stitch_history_cache(
        model,
        source_context_cache=source_cache,
        target_prefix_cache=target_prefix_cache,
        source_prefix_length=len(source.prefix_ids),
        target_prefix_length=len(target.prefix_ids),
        transfer_history_length=len(target.history_ids),
    )
    _, direct_text = greedy_action(
        model,
        _factory(model, stitched, target),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )
    direct_call = parse_tool_call(direct_text)
    direct_target_score = score_bfcl_call(
        direct_call,
        expected=row["target_expected"],
        argument_options=row["target_argument_options"],
    )
    direct_source_score = score_bfcl_call(
        direct_call,
        expected=row["source_expected"],
        argument_options=row["source_argument_options"],
    )
    del source_cache, target_cache, target_prefix_cache, stitched
    eligible = (
        source_score["compatible_em"]
        and target_score["compatible_em"]
        and row["source_expected"]["name"] != row["target_expected"]["name"]
    )
    return {
        "id": row["id"],
        "split": row["split"],
        "eligible": eligible,
        "source_native": {
            **source_score,
            "call": asdict(source_call),
            "text": source_text,
        },
        "target_native": {
            **target_score,
            "call": asdict(target_call),
            "text": target_text,
            "action_ids": list(target_ids),
            "teacher_topk_indices": topk_indices.squeeze(0).cpu().tolist(),
            "teacher_topk_log_probs": topk_log_probs.squeeze(0).cpu().tolist(),
            "teacher_topk_mass": topk_mass.squeeze(0).cpu().tolist(),
        },
        "direct_reuse": {
            "target": direct_target_score,
            "source": direct_source_score,
            "call": asdict(direct_call),
            "text": direct_text,
        },
        "token_lengths": {
            "source_prefix": len(source.prefix_ids),
            "target_prefix": len(target.prefix_ids),
            "history": len(target.history_ids),
        },
    }


def score_bfcl_call(
    call: ToolCall,
    *,
    expected: dict,
    argument_options: dict[str, list],
) -> dict[str, bool]:
    name_em = call.valid_json and call.name == expected["name"]
    strict_arguments_em = call.arguments == expected["arguments"]
    required_arguments_em = bool(name_em and isinstance(call.arguments, dict))
    if required_arguments_em:
        for name, canonical in expected["arguments"].items():
            actual = call.arguments.get(name)
            allowed = argument_options.get(name, [canonical])
            if not any(_equivalent(actual, value) for value in allowed if value != ""):
                required_arguments_em = False
                break
    return {
        "valid_json": call.valid_json,
        "name_em": bool(name_em),
        "strict_arguments_em": bool(strict_arguments_em),
        "compatible_em": bool(name_em and required_arguments_em),
    }


def _equivalent(left, right) -> bool:
    if left == right:
        return True
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return str(left).strip().casefold() == str(right).strip().casefold()


def _factory(model, cache, segments):
    return lambda: start_readout(
        model,
        legacy_cache=cache,
        cached_tokens=len(segments.prefix_ids) + len(segments.history_ids),
        input_ids=segments.readout_ids,
    )


def _jsonl(path: Path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


if __name__ == "__main__":
    main()
