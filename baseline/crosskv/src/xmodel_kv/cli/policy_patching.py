from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
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
    patch_history_cache,
    prefill_legacy_cache,
    start_readout,
    validate_shared_handoff,
)
from .bfcl_memory_policy_imprinting import MODES, POLICY_STYLES, memory_case
from .policy_imprinting import DIRECTIONS, SCENARIOS, _controlled_case


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causally localize policy imprinting across K/V, layers, and history tokens"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="history_selection")
    parser.add_argument("--direction", choices=DIRECTIONS, default="forward")
    parser.add_argument("--history-turn-pairs", type=int, default=32)
    parser.add_argument(
        "--bfcl-memory-root",
        help="BFCL data root; when set, localize --bfcl-memory-id instead of a synthetic case",
    )
    parser.add_argument("--bfcl-memory-id")
    parser.add_argument("--memory-mode", choices=MODES, default="tagged")
    parser.add_argument(
        "--memory-policy-style", choices=POLICY_STYLES, default="strong"
    )
    parser.add_argument(
        "--triad-family",
        choices=("position", "content"),
        help="localize a decisive matched/third triad case",
    )
    parser.add_argument(
        "--triad-direction",
        choices=("target_first", "target_last", "target_urgent", "target_cheapest"),
    )
    parser.add_argument(
        "--triad-arm", choices=("matched", "third"), default="third"
    )
    parser.add_argument(
        "--triad-layout", choices=("clean", "shifted"), default="clean"
    )
    parser.add_argument("--triad-replicate", type=int, default=0)
    parser.add_argument("--triad-seed", type=int, default=2027)
    parser.add_argument("--triad-record-count", type=int, default=9)
    parser.add_argument("--triad-shift-padding-tokens", type=int, default=32)
    parser.add_argument("--layer-block-size", type=int, default=6)
    parser.add_argument(
        "--extra-layer-ranges",
        default="",
        help="extra inclusive layer ranges, e.g. 18-29,30-35",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.history_turn_pairs < 0:
        parser.error("history-turn-pairs must be non-negative")
    if args.layer_block_size < 1:
        parser.error("layer-block-size must be positive")
    try:
        extra_layer_ranges = _parse_layer_ranges(args.extra_layer_ranges)
    except ValueError as error:
        parser.error(str(error))
    if bool(args.bfcl_memory_root) != bool(args.bfcl_memory_id):
        parser.error("--bfcl-memory-root and --bfcl-memory-id must be provided together")
    if args.triad_family and args.bfcl_memory_root:
        parser.error("triad and BFCL modes are mutually exclusive")
    if bool(args.triad_family) != bool(args.triad_direction):
        parser.error("--triad-family and --triad-direction must be provided together")
    if args.triad_family == "position" and args.triad_direction not in {
        "target_first",
        "target_last",
    }:
        parser.error("position triad requires target_first or target_last")
    if args.triad_family == "content" and args.triad_direction not in {
        "target_urgent",
        "target_cheapest",
    }:
        parser.error("content triad requires target_urgent or target_cheapest")
    if args.triad_replicate < 0:
        parser.error("--triad-replicate must be non-negative")
    if args.triad_record_count < 8:
        parser.error("--triad-record-count must be at least 8")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        attn_implementation="sdpa",
    ).eval()
    metadata = None
    if args.triad_family:
        # Import lazily because policy_triad reuses _run_case from the sibling
        # policy-imprinting CLI; keeping this edge local avoids a module cycle.
        from .policy_triad import _length_match_group, _shift_prompt, _triad_group

        group = _triad_group(
            args.triad_family,
            args.triad_direction,
            replicate=args.triad_replicate,
            seed=args.triad_seed,
            record_count=args.triad_record_count,
        )
        clean_cases, clean_prefix_length = _length_match_group(tokenizer, group)
        case = clean_cases[args.triad_arm]
        if args.triad_layout == "shifted":
            case = replace(
                case,
                source_prompt=_shift_prompt(
                    case.source_prompt, args.triad_shift_padding_tokens
                ),
            )
        metadata = {
            "family": args.triad_family,
            "direction": args.triad_direction,
            "arm": args.triad_arm,
            "layout": args.triad_layout,
            "replicate": args.triad_replicate,
            "seed": args.triad_seed,
            "record_count": args.triad_record_count,
            "chance_level": 1.0 / args.triad_record_count,
            "target_record_id": group.target_record_id,
            "third_record_id": group.third_record_id,
            "clean_prefix_length": clean_prefix_length,
            **group.metadata,
        }
    elif args.bfcl_memory_id:
        case, domain, metadata = memory_case(
            Path(args.bfcl_memory_root),
            bfcl_id=args.bfcl_memory_id,
            mode=args.memory_mode,
            direction=args.direction,
            policy_style=args.memory_policy_style,
        )
        metadata = {
            "bfcl_id": args.bfcl_memory_id,
            "bfcl_domain": domain,
            "bfcl_mode": args.memory_mode,
            "bfcl_policy_style": args.memory_policy_style,
            **metadata,
        }
    else:
        case = _controlled_case(args.scenario, args.direction, args.history_turn_pairs)
    row = _run_patching(
        model,
        tokenizer,
        case,
        layer_block_size=args.layer_block_size,
        max_new_tokens=args.max_new_tokens,
        extra_layer_ranges=extra_layer_ranges,
    )
    if metadata is not None:
        key = "triad_metadata" if args.triad_family else "bfcl_memory_metadata"
        row[key] = metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        json.dump(row, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(
        {
            "case_id": case.case_id,
            "target": row["target_native"]["tool_call"],
            "source": row["source_native"]["tool_call"],
            "all_source": row["interventions"][0]["tool_call"],
            "output": str(output_path),
        }
    )


@torch.inference_mode()
def _run_patching(
    model,
    tokenizer,
    case,
    *,
    layer_block_size: int,
    max_new_tokens: int,
    extra_layer_ranges: tuple[tuple[int, int], ...] = (),
) -> dict:
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

    source_factory = _factory(
        model,
        source_context,
        len(source.prefix_ids) + len(source.history_ids),
        source.readout_ids,
    )
    target_factory = _factory(
        model,
        target_context,
        len(target.prefix_ids) + len(target.history_ids),
        target.readout_ids,
    )
    source_action_ids, source_text = greedy_action(
        model, source_factory, tokenizer, max_new_tokens=max_new_tokens
    )
    target_action_ids, target_text = greedy_action(
        model, target_factory, tokenizer, max_new_tokens=max_new_tokens
    )
    target_logits = action_logits(model, target_factory, target_action_ids)
    target_source_logits = action_logits(model, target_factory, source_action_ids)
    target_margin = (
        action_mean_logprob(target_source_logits, source_action_ids)
        - action_mean_logprob(target_logits, target_action_ids)
    )

    layer_count = model.config.num_hidden_layers
    all_layers = set(range(layer_count))
    specifications: list[dict] = [
        {"name": "all_source", "layers": all_layers, "keys": True, "values": True},
        {"name": "all_source_K_only", "layers": all_layers, "keys": True, "values": False},
        {"name": "all_source_V_only", "layers": all_layers, "keys": False, "values": True},
    ]
    for start in range(0, layer_count, layer_block_size):
        block = set(range(start, min(start + layer_block_size, layer_count)))
        label = f"layers_{start:02d}_{max(block):02d}"
        specifications.append(
            {
                "name": f"source_only_{label}",
                "layers": block,
                "keys": True,
                "values": True,
            }
        )
        specifications.append(
            {
                "name": f"target_repair_{label}",
                "layers": all_layers - block,
                "keys": True,
                "values": True,
            }
        )

    for start, end in extra_layer_ranges:
        if end >= layer_count:
            raise ValueError(
                f"extra layer range {start}-{end} exceeds model layers 0-{layer_count - 1}"
            )
        block = set(range(start, end + 1))
        label = f"layers_{start:02d}_{end:02d}"
        specifications.append(
            {
                "name": f"source_only_{label}",
                "layers": block,
                "keys": True,
                "values": True,
            }
        )
        specifications.append(
            {
                "name": f"target_repair_{label}",
                "layers": all_layers - block,
                "keys": True,
                "values": True,
            }
        )

    spans = history_message_spans(
        tokenizer,
        system_prompt=case.target_prompt,
        tools=case.target_tools,
        history=case.history,
    )
    token_regions = _token_regions(len(target.history_ids), spans, case.history)
    for name, mask in token_regions.items():
        specifications.append(
            {
                "name": f"source_tokens_{name}",
                "layers": all_layers,
                "keys": True,
                "values": True,
                "mask": mask,
            }
        )

    interventions = []
    for specification in specifications:
        patched = patch_history_cache(
            model,
            source_context_cache=source_context,
            target_context_cache=target_context,
            target_prefix_cache=target_prefix,
            source_prefix_length=len(source.prefix_ids),
            target_prefix_length=len(target.prefix_ids),
            history_length=len(target.history_ids),
            source_layers=specification["layers"],
            source_keys=specification["keys"],
            source_values=specification["values"],
            source_token_mask=specification.get("mask"),
        )
        factory = _factory(
            model,
            patched,
            len(target.prefix_ids) + len(target.history_ids),
            target.readout_ids,
        )
        _, text = greedy_action(
            model, factory, tokenizer, max_new_tokens=max_new_tokens
        )
        candidate_logits = action_logits(model, factory, target_action_ids)
        candidate_source_logits = action_logits(model, factory, source_action_ids)
        margin = (
            action_mean_logprob(candidate_source_logits, source_action_ids)
            - action_mean_logprob(candidate_logits, target_action_ids)
        )
        call = parse_tool_call(text)
        interventions.append(
            {
                "name": specification["name"],
                "source_layer_count": len(specification["layers"]),
                "source_keys": specification["keys"],
                "source_values": specification["values"],
                "source_token_count": int(
                    sum(specification.get("mask", [True] * len(target.history_ids)))
                ),
                "generation": text,
                "tool_call": asdict(call),
                "target_full_call_em": (
                    call.name == case.target_expected["name"]
                    and call.arguments == case.target_expected["arguments"]
                ),
                "source_full_call_em": (
                    call.name == case.source_expected["name"]
                    and call.arguments == case.source_expected["arguments"]
                ),
                "distribution": asdict(
                    compare_action_distributions(
                        target_logits, candidate_logits, target_action_ids
                    )
                ),
                "source_action_margin": margin,
                "source_leakage_shift": margin - target_margin,
            }
        )

    return {
        "case_id": case.case_id,
        "scenario": case.scenario,
        "direction": case.direction,
        "token_lengths": {
            "source_prefix": len(source.prefix_ids),
            "target_prefix": len(target.prefix_ids),
            "history": len(target.history_ids),
            "readout": len(target.readout_ids),
        },
        "source_native": {
            "generation": source_text,
            "tool_call": asdict(parse_tool_call(source_text)),
        },
        "target_native": {
            "generation": target_text,
            "tool_call": asdict(parse_tool_call(target_text)),
        },
        "target_native_source_action_margin": target_margin,
        "message_spans": spans,
        "token_regions": {
            name: int(sum(mask)) for name, mask in token_regions.items()
        },
        "interventions": interventions,
    }


def _token_regions(history_length: int, spans, history) -> dict[str, list[bool]]:
    def mask_for(indices) -> list[bool]:
        mask = [False] * history_length
        for index in indices:
            start, end = spans[index]
            mask[start:end] = [True] * (end - start)
        return mask

    final_query = [len(history) - 1]
    prior_messages = list(range(len(history) - 1))
    user_messages = [
        index for index, message in enumerate(history[:-1]) if message["role"] == "user"
    ]
    assistant_messages = [
        index for index, message in enumerate(history[:-1]) if message["role"] == "assistant"
    ]
    regions = {
        "final_query": mask_for(final_query),
        "prior_history": mask_for(prior_messages),
        "prior_user_messages": mask_for(user_messages),
        "prior_assistant_messages": mask_for(assistant_messages),
    }
    if len(history) >= 5:
        regions["first_exchange"] = mask_for((0, 1))
        regions["last_exchange"] = mask_for((len(history) - 3, len(history) - 2))
    return regions


def _factory(model, cache, cached_tokens: int, input_ids):
    return lambda: start_readout(
        model,
        legacy_cache=cache,
        cached_tokens=cached_tokens,
        input_ids=input_ids,
    )


def _parse_layer_ranges(value: str) -> tuple[tuple[int, int], ...]:
    if not value.strip():
        return ()
    ranges = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        pieces = part.split("-")
        if len(pieces) != 2:
            raise ValueError(f"invalid layer range {part!r}; expected START-END")
        try:
            start, end = (int(piece) for piece in pieces)
        except ValueError as error:
            raise ValueError(
                f"invalid layer range {part!r}; bounds must be integers"
            ) from error
        if start < 0 or end < start:
            raise ValueError(f"invalid layer range {part!r}; require 0 <= START <= END")
        ranges.append((start, end))
    if len(set(ranges)) != len(ranges):
        raise ValueError("duplicate extra layer ranges are not allowed")
    return tuple(ranges)


if __name__ == "__main__":
    main()
