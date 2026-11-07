from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..diverse_handoff import load_manifest
from ..diverse_training import load_screen_rows, select_eligible_rows
from ..policy_imprinting import (
    build_chat_segments,
    greedy_action,
    parse_tool_call,
    prefill_legacy_cache,
    start_readout,
    stitch_history_cache,
    validate_shared_handoff,
)
from .screen_diverse_handoff import score_bfcl_call


NEUTRAL_RECEIVER_PROMPT = (
    "You are a tool-using assistant. Follow the conversation and use the available "
    "tool when requested."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full target re-prefill and a target-policy-at-tail KV reuse control"
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--screen", required=True, nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="iid_test")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.rows < 1 or args.max_new_tokens < 1:
        parser.error("rows and max-new-tokens must be positive")

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        global_rank, world_size, device = 0, 1, args.device
    is_main = global_rank == 0

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    manifest = load_manifest(Path(args.manifest))
    screens = load_screen_rows([Path(path) for path in args.screen])
    selected, selection_audit = select_eligible_rows(
        manifest, screens, limits={args.split: args.rows}
    )
    rows = selected[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": device},
        attn_implementation="sdpa",
    ).eval()

    shard_path = output_dir / f"rows.rank{global_rank}.jsonl"
    counts = {
        name: {"target": 0, "source": 0, "valid_json": 0}
        for name in ("front_full_reprefill", "tail_full_reprefill", "tail_kv_reuse")
    }
    local_rows = rows[global_rank::world_size]
    with shard_path.open("w") as output:
        for step, row in enumerate(local_rows, 1):
            result = _evaluate_one(
                model,
                tokenizer,
                row,
                max_new_tokens=args.max_new_tokens,
            )
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            for name in counts:
                counts[name]["target"] += int(result[name]["target"]["compatible_em"])
                counts[name]["source"] += int(result[name]["source"]["compatible_em"])
                counts[name]["valid_json"] += int(result[name]["call"]["valid_json"])
            if step == 1 or step % 10 == 0:
                print(
                    {
                        "rank": global_rank,
                        "step": step,
                        "local_rows": len(local_rows),
                        **{
                            name: f"{value['target']}/{step}"
                            for name, value in counts.items()
                        },
                    },
                    flush=True,
                )
            gc.collect()
            torch.cuda.empty_cache()

    count_names = tuple(counts)
    packed = torch.tensor(
        [
            value
            for name in count_names
            for value in (
                counts[name]["target"],
                counts[name]["source"],
                counts[name]["valid_json"],
            )
        ]
        + [len(local_rows)],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        dist.barrier()

    if is_main:
        all_rows = []
        for rank in range(world_size):
            all_rows.extend(_jsonl(output_dir / f"rows.rank{rank}.jsonl"))
        all_rows.sort(key=lambda row: row["id"])
        total = int(packed[-1].item())
        summary = {
            "split": args.split,
            "rows": total,
            "layout_definitions": {
                "front_full_reprefill": "target_private_prefix + plaintext_history",
                "tail_full_reprefill": (
                    "neutral_target_tool_prefix + plaintext_history + native_target_system_tail"
                ),
                "tail_kv_reuse": (
                    "neutral_target_tool_prefix + source_imprinted_history_KV + "
                    "native_target_system_tail"
                ),
            },
            "selection": selection_audit,
            "metrics": {},
            "paired": _paired_counts(all_rows),
        }
        offset = 0
        for name in count_names:
            target, source, valid = (int(value) for value in packed[offset : offset + 3])
            offset += 3
            summary["metrics"][name] = {
                "target_correct": target,
                "target_accuracy": target / total,
                "source_correct": source,
                "source_accuracy": source / total,
                "valid_json": valid,
                "valid_json_rate": valid / total,
            }
        (output_dir / "rows.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows)
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    if distributed:
        dist.destroy_process_group()


@torch.inference_mode()
def _evaluate_one(model, tokenizer, row, *, max_new_tokens: int) -> dict:
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
    neutral = build_chat_segments(
        tokenizer,
        system_prompt=NEUTRAL_RECEIVER_PROMPT,
        tools=row["target_tools"],
        history=row["history"],
    )
    validate_shared_handoff(source, target)
    validate_shared_handoff(source, neutral)

    tail_message = {"role": "system", "content": row["target_prompt"]}
    neutral_tail = build_chat_segments(
        tokenizer,
        system_prompt=NEUTRAL_RECEIVER_PROMPT,
        tools=row["target_tools"],
        history=[*row["history"], tail_message],
    )
    tail_ids = _appended_history_ids(neutral, neutral_tail)

    # Control 1: the receiving agent recomputes every plaintext history token.
    target_context = prefill_legacy_cache(
        model, target.prefix_ids + target.history_ids
    )
    _, front_text = greedy_action(
        model,
        _factory(
            model,
            target_context,
            len(target.prefix_ids) + len(target.history_ids),
            target.readout_ids,
        ),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )

    # Layout-matched native diagnostic: B is late, but H is still recomputed.
    tail_native_context = prefill_legacy_cache(
        model, neutral_tail.prefix_ids + neutral_tail.history_ids
    )
    _, tail_native_text = greedy_action(
        model,
        _factory(
            model,
            tail_native_context,
            len(neutral_tail.prefix_ids) + len(neutral_tail.history_ids),
            neutral_tail.readout_ids,
        ),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )

    # Control 2: retain A-imprinted H-KV, drop A's private prefix, append B natively.
    source_context = prefill_legacy_cache(
        model, source.prefix_ids + source.history_ids
    )
    neutral_prefix = prefill_legacy_cache(model, neutral.prefix_ids)
    stitched = stitch_history_cache(
        model,
        source_context_cache=source_context,
        target_prefix_cache=neutral_prefix,
        source_prefix_length=len(source.prefix_ids),
        target_prefix_length=len(neutral.prefix_ids),
        transfer_history_length=len(source.history_ids),
    )
    _, tail_reuse_text = greedy_action(
        model,
        _factory(
            model,
            stitched,
            len(neutral.prefix_ids) + len(neutral.history_ids),
            tail_ids + neutral_tail.readout_ids,
        ),
        tokenizer,
        max_new_tokens=max_new_tokens,
    )

    del target_context, tail_native_context, source_context, neutral_prefix, stitched
    return {
        "id": row["id"],
        "history_tokens": row["history_tokens"],
        "tail_policy_tokens": len(tail_ids),
        "front_full_reprefill": _score_text(front_text, row),
        "tail_full_reprefill": _score_text(tail_native_text, row),
        "tail_kv_reuse": _score_text(tail_reuse_text, row),
    }


def _score_text(text: str, row: dict) -> dict:
    call = parse_tool_call(text)
    return {
        "text": text,
        "call": asdict(call),
        "target": score_bfcl_call(
            call,
            expected=row["target_expected"],
            argument_options=row["target_argument_options"],
        ),
        "source": score_bfcl_call(
            call,
            expected=row["source_expected"],
            argument_options=row["source_argument_options"],
        ),
    }


def _factory(model, cache, cached_tokens: int, input_ids):
    return lambda: start_readout(
        model,
        legacy_cache=cache,
        cached_tokens=cached_tokens,
        input_ids=input_ids,
    )


def _appended_history_ids(base, appended):
    base_ids = tuple(base.history_ids)
    full_ids = tuple(appended.history_ids)
    if full_ids[: len(base_ids)] != base_ids:
        raise ValueError("appending the target policy rewrote earlier history tokens")
    suffix = full_ids[len(base_ids) :]
    if not suffix:
        raise ValueError("target policy tail contains no tokens")
    return suffix


def _paired_counts(rows: list[dict]) -> dict:
    pairs = (
        ("front_full_reprefill", "tail_full_reprefill"),
        ("tail_full_reprefill", "tail_kv_reuse"),
        ("front_full_reprefill", "tail_kv_reuse"),
    )
    result = {}
    for left, right in pairs:
        left_only = right_only = both = neither = 0
        for row in rows:
            left_correct = row[left]["target"]["compatible_em"]
            right_correct = row[right]["target"]["compatible_em"]
            both += int(left_correct and right_correct)
            left_only += int(left_correct and not right_correct)
            right_only += int(right_correct and not left_correct)
            neither += int(not left_correct and not right_correct)
        result[f"{left}_vs_{right}"] = {
            "both_correct": both,
            "left_only_correct": left_only,
            "right_only_correct": right_only,
            "neither_correct": neither,
        }
    return result


def _jsonl(path: Path) -> list[dict]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


if __name__ == "__main__":
    main()
