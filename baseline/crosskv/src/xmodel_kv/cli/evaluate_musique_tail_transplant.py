from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from xmodel_kv.coqa import coqa_f1
from xmodel_kv.policy_imprinting import build_chat_segments, stitch_history_cache

from .train_musique_semantic_handoff import (
    DEPENDENT_B_SYSTEM,
    INTERMEDIATE_B_SYSTEM,
    INTERMEDIATE_QUERY,
    _clean_generation,
    _format_target,
    _generate,
    _load_model,
    _load_slice,
    _normalized,
    _source_cache,
    _system_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a no-training, same-model oracle that transplants only Agent A's "
            "answer-tail KV into Agent B after exact RoPE repositioning."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--source-state",
        choices=("generated_answer", "gold_answer"),
        default="gold_answer",
    )
    parser.add_argument("--source-answer-max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--tail-mode", choices=("answer", "pre_answer"), default="answer"
    )
    parser.add_argument("--pre-answer-tail-tokens", type=int, default=4)
    parser.add_argument("--eval-offset", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--source-case-shift", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--quant-bits", default="16,8,4")
    parser.add_argument("--layer-patterns", default="all")
    parser.add_argument("--head-patterns", default="all")
    parser.add_argument("--head-ranks", default="full")
    args = parser.parse_args()
    if min(
        args.source_answer_max_new_tokens,
        args.pre_answer_tail_tokens,
        args.eval_count,
        args.max_new_tokens,
    ) <= 0 or min(args.eval_offset, args.source_case_shift) < 0:
        parser.error("counts and token limits must be positive; offset must be non-negative")
    quant_bits_values = tuple(int(value) for value in args.quant_bits.split(","))
    if (
        not quant_bits_values
        or any(value not in (4, 8, 16) for value in quant_bits_values)
        or len(set(quant_bits_values)) != len(quant_bits_values)
    ):
        parser.error("--quant-bits must be a unique comma-separated subset of 16,8,4")
    layer_patterns = tuple(value.strip() for value in args.layer_patterns.split(","))
    fixed_layer_patterns = {
        "all",
        "even",
        "odd",
        "first_half",
        "last_half",
        "mod4_0",
        "mod4_1",
        "mod4_2",
        "mod4_3",
    }
    if (
        not layer_patterns
        or any(
            value not in fixed_layer_patterns
            and not (
                (value.startswith("first_") or value.startswith("last_"))
                and value.split("_", 1)[1].isdigit()
                and int(value.split("_", 1)[1]) > 0
            )
            for value in layer_patterns
        )
        or len(set(layer_patterns)) != len(layer_patterns)
    ):
        parser.error("--layer-patterns contains an unknown or duplicate pattern")
    head_patterns = tuple(value.strip() for value in args.head_patterns.split(","))
    fixed_head_patterns = {"all", "even", "odd", "first_half", "last_half"}
    if (
        not head_patterns
        or any(
            value not in fixed_head_patterns
            and not (
                (value.startswith("first_") or value.startswith("last_"))
                and value.split("_", 1)[1].isdigit()
                and int(value.split("_", 1)[1]) > 0
            )
            for value in head_patterns
        )
        or len(set(head_patterns)) != len(head_patterns)
    ):
        parser.error("--head-patterns contains an unknown or duplicate pattern")
    head_ranks = tuple(value.strip() for value in args.head_ranks.split(","))
    if (
        not head_ranks
        or any(
            value != "full" and not (value.isdigit() and int(value) > 0)
            for value in head_ranks
        )
        or len(set(head_ranks)) != len(head_ranks)
    ):
        parser.error("--head-ranks must contain unique positive ranks and/or full")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "quant_bits": list(quant_bits_values),
        "layer_patterns": list(layer_patterns),
        "head_patterns": list(head_patterns),
        "head_ranks": list(head_ranks),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    stage_system_ids, stage_system_cache = _system_cache(
        model, tokenizer, DEPENDENT_B_SYSTEM, "placeholder"
    )
    bridge_system_ids, bridge_system_cache = _system_cache(
        model, tokenizer, INTERMEDIATE_B_SYSTEM, INTERMEDIATE_QUERY
    )

    rows = []
    eval_cases = _load_slice(
        args.dataset, offset=args.eval_offset, count=args.eval_count
    )
    for position, (index, case) in enumerate(eval_cases, 1):
        source_index, source_case = eval_cases[
            (position - 1 + args.source_case_shift) % len(eval_cases)
        ]
        case_rows = evaluate_case(
            model,
            tokenizer,
            index=index,
            case=case,
            source_index=source_index,
            source_case=source_case,
            source_state=args.source_state,
            source_answer_max_new_tokens=args.source_answer_max_new_tokens,
            tail_mode=args.tail_mode,
            pre_answer_tail_tokens=args.pre_answer_tail_tokens,
            stage_system_ids=stage_system_ids,
            stage_system_cache=stage_system_cache,
            bridge_system_ids=bridge_system_ids,
            bridge_system_cache=bridge_system_cache,
            max_new_tokens=args.max_new_tokens,
            quant_bits_values=quant_bits_values,
            layer_patterns=layer_patterns,
            head_patterns=head_patterns,
            head_ranks=head_ranks,
        )
        rows.extend(case_rows)
        for row in case_rows:
            print(json.dumps({"event": "evaluate", **row}), flush=True)
        print(json.dumps({"event": "evaluation_progress", "position": position}), flush=True)

    results_path = output_dir / "results.jsonl"
    results_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = summarize(rows, config=config, model=model)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


@torch.inference_mode()
def evaluate_case(
    model,
    tokenizer,
    *,
    index: int,
    case: dict,
    source_index: int,
    source_case: dict,
    source_state: str,
    source_answer_max_new_tokens: int,
    tail_mode: str,
    pre_answer_tail_tokens: int,
    stage_system_ids,
    stage_system_cache,
    bridge_system_ids,
    bridge_system_cache,
    max_new_tokens: int,
    quant_bits_values: tuple[int, ...],
    layer_patterns: tuple[str, ...],
    head_patterns: tuple[str, ...],
    head_ranks: tuple[str, ...],
) -> list[dict]:
    source_cache, source_tokens, source_answer, answer_tokens, source_metadata = _source_cache(
        model,
        tokenizer,
        source_case,
        source_state=source_state,
        max_new_tokens=source_answer_max_new_tokens,
        return_metadata=True,
    )
    if answer_tokens <= 0 or answer_tokens > source_tokens:
        raise ValueError("source answer tail must be non-empty and inside the source cache")
    answer_start = source_tokens - answer_tokens
    if tail_mode == "answer":
        source_segment_start = answer_start
        transfer_tokens = answer_tokens
    elif tail_mode == "pre_answer":
        transfer_tokens = min(pre_answer_tail_tokens, answer_start)
        source_segment_start = answer_start - transfer_tokens
    else:
        raise ValueError(f"unknown tail mode: {tail_mode}")

    stage_segments = build_chat_segments(
        tokenizer,
        system_prompt=DEPENDENT_B_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": _format_target(case)}],
        enable_thinking=False,
    )
    bridge_segments = build_chat_segments(
        tokenizer,
        system_prompt=INTERMEDIATE_B_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": INTERMEDIATE_QUERY}],
        enable_thinking=False,
    )
    if tuple(stage_segments.prefix_ids) != tuple(stage_system_ids):
        raise ValueError("stage prefix changed during evaluation")
    if tuple(bridge_segments.prefix_ids) != tuple(bridge_system_ids):
        raise ValueError("bridge prefix changed during evaluation")

    stage_cache_bf16 = stitch_history_cache(
        model,
        source_context_cache=source_cache,
        target_prefix_cache=stage_system_cache,
        source_prefix_length=source_segment_start,
        target_prefix_length=len(stage_system_ids),
        transfer_history_length=transfer_tokens,
    )
    bridge_cache_bf16 = stitch_history_cache(
        model,
        source_context_cache=source_cache,
        target_prefix_cache=bridge_system_cache,
        source_prefix_length=source_segment_start,
        target_prefix_length=len(bridge_system_ids),
        transfer_history_length=transfer_tokens,
    )
    no_summary = _clean_generation(
        _generate(
            model,
            tokenizer,
            legacy_cache=stage_system_cache,
            cached_tokens=len(stage_system_ids),
            query_ids=(*stage_segments.history_ids, *stage_segments.readout_ids),
            max_new_tokens=max_new_tokens,
        )
    )
    final_gold = case["agent_b"]["gold_answer"]
    bridge_gold = case["agent_a"]["gold_answer"]
    rows = []
    for quant_bits in quant_bits_values:
        for layer_pattern in layer_patterns:
            active_layers = active_layer_indices(
                layer_pattern, total_layers=len(stage_cache_bf16)
            )
            total_heads = stage_cache_bf16[0][0].shape[1]
            for head_pattern in head_patterns:
                active_heads = active_head_indices(
                    head_pattern, total_heads=total_heads
                )
                for head_rank_value in head_ranks:
                    head_rank = (
                        None if head_rank_value == "full" else int(head_rank_value)
                    )
                    if head_rank is not None and head_rank > len(active_heads):
                        raise ValueError("head rank exceeds the number of active KV heads")
                    stage_cache = mask_and_fake_quantize_suffix(
                        stage_cache_bf16,
                        suffix_tokens=transfer_tokens,
                        bits=quant_bits,
                        active_layers=active_layers,
                        active_heads=active_heads,
                        head_rank=head_rank,
                    )
                    bridge_cache = mask_and_fake_quantize_suffix(
                        bridge_cache_bf16,
                        suffix_tokens=transfer_tokens,
                        bits=quant_bits,
                        active_layers=active_layers,
                        active_heads=active_heads,
                        head_rank=head_rank,
                    )
                    student = _clean_generation(
                        _generate(
                            model,
                            tokenizer,
                            legacy_cache=stage_cache,
                            cached_tokens=len(stage_system_ids) + transfer_tokens,
                            query_ids=(*stage_segments.history_ids, *stage_segments.readout_ids),
                            max_new_tokens=max_new_tokens,
                        )
                    )
                    bridge_student = _clean_generation(
                        _generate(
                            model,
                            tokenizer,
                            legacy_cache=bridge_cache,
                            cached_tokens=len(bridge_system_ids) + transfer_tokens,
                            query_ids=(*bridge_segments.history_ids, *bridge_segments.readout_ids),
                            max_new_tokens=max_new_tokens,
                        )
                    )
                    rows.append({
                        "index": index,
                        "id": case["id"],
                        "source_index": source_index,
                        "source_id": source_case["id"],
                        "relation_key": case["relation_key"],
                        "quant_bits": quant_bits,
                        "layer_pattern": layer_pattern,
                        "head_pattern": head_pattern,
                        "head_rank": "full" if head_rank is None else head_rank,
                        "tail_mode": tail_mode,
                        "active_layers": len(active_layers),
                        "active_kv_heads": len(active_heads),
                        "source_tokens": source_tokens,
                        "tail_tokens": transfer_tokens,
                        "source_answer_tokens": answer_tokens,
                        "source_answer": source_answer,
                        **source_metadata,
                        "student": student,
                        "bridge_student": bridge_student,
                        "no_summary": no_summary,
                        "final_gold": final_gold,
                        "bridge_gold": bridge_gold,
                        "student_f1": coqa_f1(student, [final_gold]),
                        "bridge_f1": coqa_f1(bridge_student, [bridge_gold]),
                        "no_summary_f1": coqa_f1(no_summary, [final_gold]),
                        "student_exact": float(
                            _normalized(student) == _normalized(final_gold)
                        ),
                        "bridge_exact": float(
                            _normalized(bridge_student) == _normalized(bridge_gold)
                        ),
                        "no_summary_exact": float(
                            _normalized(no_summary) == _normalized(final_gold)
                        ),
                        "source_answer_exact": float(
                            _normalized(source_answer) == _normalized(bridge_gold)
                        ),
                        "source_answer_contains": float(
                            _normalized(bridge_gold) in _normalized(source_answer)
                        ),
                    })
    return rows


def active_layer_indices(pattern: str, *, total_layers: int) -> frozenset[int]:
    if total_layers < 1:
        raise ValueError("total_layers must be positive")
    if pattern == "all":
        return frozenset(range(total_layers))
    if pattern == "even":
        return frozenset(range(0, total_layers, 2))
    if pattern == "odd":
        return frozenset(range(1, total_layers, 2))
    if pattern == "first_half":
        return frozenset(range((total_layers + 1) // 2))
    if pattern == "last_half":
        return frozenset(range(total_layers // 2, total_layers))
    if pattern.startswith("first_") and pattern.removeprefix("first_").isdigit():
        count = int(pattern.removeprefix("first_"))
        if count > total_layers:
            raise ValueError("first-N layer pattern exceeds total layers")
        return frozenset(range(count))
    if pattern.startswith("last_") and pattern.removeprefix("last_").isdigit():
        count = int(pattern.removeprefix("last_"))
        if count > total_layers:
            raise ValueError("last-N layer pattern exceeds total layers")
        return frozenset(range(total_layers - count, total_layers))
    if pattern.startswith("mod4_"):
        residue = int(pattern[-1])
        return frozenset(range(residue, total_layers, 4))
    raise ValueError(f"unknown layer pattern: {pattern}")


def active_head_indices(pattern: str, *, total_heads: int) -> frozenset[int]:
    if total_heads < 1:
        raise ValueError("total_heads must be positive")
    if pattern == "all":
        return frozenset(range(total_heads))
    if pattern == "even":
        return frozenset(range(0, total_heads, 2))
    if pattern == "odd":
        return frozenset(range(1, total_heads, 2))
    if pattern == "first_half":
        return frozenset(range((total_heads + 1) // 2))
    if pattern == "last_half":
        return frozenset(range(total_heads // 2, total_heads))
    if pattern.startswith("first_") and pattern.split("_", 1)[1].isdigit():
        count = int(pattern.split("_", 1)[1])
        if count > total_heads:
            raise ValueError("first-N head pattern exceeds total heads")
        return frozenset(range(count))
    if pattern.startswith("last_") and pattern.split("_", 1)[1].isdigit():
        count = int(pattern.split("_", 1)[1])
        if count > total_heads:
            raise ValueError("last-N head pattern exceeds total heads")
        return frozenset(range(total_heads - count, total_heads))
    raise ValueError(f"unknown head pattern: {pattern}")


def mask_and_fake_quantize_suffix(
    cache,
    *,
    suffix_tokens: int,
    bits: int,
    active_layers: frozenset[int],
    active_heads: frozenset[int] | None = None,
    head_rank: int | None = None,
    straight_through: bool = False,
):
    """Mask unsent layers and quantize transmitted token/head vectors.

    ``straight_through`` keeps the same quantized forward values while routing
    gradients through the unquantized selected vectors.  It is intended only
    for packet-aware writer training; evaluation uses exact fake quantization.
    """

    if bits not in (4, 8, 16):
        raise ValueError("bits must be 4, 8, or 16")
    qmax = float((1 << (bits - 1)) - 1)
    result = []
    for layer_index, (key, value) in enumerate(cache):
        selected_heads = (
            frozenset(range(key.shape[1])) if active_heads is None else active_heads
        )
        tensors = []
        for tensor in (key, value):
            prefix = tensor[:, :, :-suffix_tokens]
            suffix = tensor[:, :, -suffix_tokens:].float()
            if layer_index not in active_layers:
                suffix = torch.zeros_like(suffix)
            else:
                head_indices = sorted(selected_heads)
                selected = suffix[:, head_indices]
                if head_rank is None:
                    if bits < 16:
                        scale = selected.abs().amax(
                            dim=-1, keepdim=True
                        ).clamp_min(1e-8) / qmax
                        quantized = (
                            (selected / scale).round().clamp(-qmax, qmax) * scale
                        )
                        selected = (
                            selected + (quantized - selected).detach()
                            if straight_through
                            else quantized
                        )
                else:
                    selected = low_rank_head_reconstruction(
                        selected, rank=head_rank, bits=bits
                    )
                head_mask = torch.zeros_like(suffix)
                head_mask[:, head_indices] = selected
                suffix = head_mask
            tensors.append(
                torch.cat((prefix, suffix.to(dtype=tensor.dtype)), dim=-2)
            )
        result.append(tuple(tensors))
    return tuple(result)


def low_rank_head_reconstruction(
    suffix: torch.Tensor, *, rank: int, bits: int
) -> torch.Tensor:
    """Factor each token's [KV-head, channel] matrix and fake-quantize factors."""

    if suffix.ndim != 4 or suffix.shape[0] != 1:
        raise ValueError("suffix must have shape [1, heads, tokens, channels]")
    if not 1 <= rank <= min(suffix.shape[1], suffix.shape[3]):
        raise ValueError("rank exceeds the suffix matrix dimensions")
    matrices = suffix.squeeze(0).permute(1, 0, 2).float()
    u, singular, vh = torch.linalg.svd(matrices, full_matrices=False)
    left = u[:, :, :rank] * singular[:, None, :rank]
    right = vh[:, :rank]
    if bits < 16:
        qmax = float((1 << (bits - 1)) - 1)
        left_scale = left.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        right_scale = right.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        left = (left / left_scale).round().clamp(-qmax, qmax) * left_scale
        right = (right / right_scale).round().clamp(-qmax, qmax) * right_scale
    reconstructed = torch.bmm(left, right)
    return reconstructed.permute(1, 0, 2).unsqueeze(0)


def summarize(rows: list[dict], *, config: dict, model) -> dict:
    config_model = model.config
    kv_heads = getattr(config_model, "num_key_value_heads", config_model.num_attention_heads)
    head_dim = getattr(
        config_model, "head_dim", config_model.hidden_size // config_model.num_attention_heads
    )
    elements_per_tail_token = config_model.num_hidden_layers * kv_heads * head_dim * 2
    by_layer_pattern = {}
    pattern_pairs = dict.fromkeys(
        (row["layer_pattern"], row["head_pattern"], row["head_rank"])
        for row in rows
    )
    for layer_pattern, head_pattern, head_rank in pattern_pairs:
        pattern_rows = [
            row
            for row in rows
            if row["layer_pattern"] == layer_pattern
            and row["head_pattern"] == head_pattern
            and row["head_rank"] == head_rank
        ]
        by_quant = {}
        for quant_bits in sorted(
            {row["quant_bits"] for row in pattern_rows}, reverse=True
        ):
            selected = [row for row in pattern_rows if row["quant_bits"] == quant_bits]
            mean = lambda key: sum(row[key] for row in selected) / len(selected)
            active_fraction = mean("active_layers") / config_model.num_hidden_layers
            active_heads = mean("active_kv_heads")
            active_layers = mean("active_layers")
            if head_rank == "full":
                factor_elements = active_heads * head_dim
                scale_vectors = active_heads
            else:
                factor_elements = float(head_rank) * (active_heads + head_dim)
                scale_vectors = active_heads + float(head_rank)
            payload = (
                mean("tail_tokens")
                * active_layers
                * 2
                * factor_elements
                * quant_bits
                / 8
            )
            if quant_bits < 16:
                # One fp16 scale per row of each transmitted K/V matrix or factor.
                payload += (
                    mean("tail_tokens")
                    * active_layers
                    * scale_vectors
                    * 2
                    * 2
                )
            by_quant[f"int{quant_bits}"] = {
                "cases": len(selected),
                "active_layers": mean("active_layers"),
                "active_kv_heads": mean("active_kv_heads"),
                "head_rank": head_rank,
                "student_accuracy": mean("student_exact"),
                "bridge_accuracy": mean("bridge_exact"),
                "no_summary_accuracy": mean("no_summary_exact"),
                "student_mean_f1": mean("student_f1"),
                "bridge_mean_f1": mean("bridge_f1"),
                "no_summary_mean_f1": mean("no_summary_f1"),
                "wins_vs_no_summary": sum(
                    row["student_f1"] > row["no_summary_f1"] for row in selected
                ),
                "mean_source_tokens": mean("source_tokens"),
                "mean_tail_tokens": mean("tail_tokens"),
                "mean_wire_payload_bytes": payload,
                "source_answer_exact": mean("source_answer_exact"),
                "source_answer_contains": mean("source_answer_contains"),
            }
        output_key = (
            layer_pattern
            if head_pattern == "all"
            else f"{layer_pattern}__heads_{head_pattern}"
        )
        if head_rank != "full":
            output_key += f"__rank_{head_rank}"
        by_layer_pattern[output_key] = by_quant
    return {
        "config": config,
        "by_layer_pattern": by_layer_pattern,
    }


if __name__ == "__main__":
    main()
