from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from xmodel_kv.soft_tail_capsule import SoftTailCapsule, load_soft_tail_capsule

from .train_musique_semantic_handoff import (
    DEPENDENT_B_SYSTEM,
    INTERMEDIATE_B_SYSTEM,
    INTERMEDIATE_QUERY,
    _load_model,
    _load_slice,
    _system_cache,
)
from .evaluate_musique_tail_transplant import active_layer_indices
from .train_musique_soft_tail_capsule import (
    _evaluate_case,
    _initial_embeddings,
    _summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen soft-tail capsule on a disjoint MuSiQue slice."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    capsule_source = parser.add_mutually_exclusive_group(required=True)
    capsule_source.add_argument("--checkpoint")
    capsule_source.add_argument(
        "--untrained-slots",
        type=int,
        help="Evaluate the deterministic embedding initialization without training.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--offset", type=int, default=96)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--eval-quant-bits", default="16,4")
    parser.add_argument("--layer-patterns", default="all")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--source-read-layers",
        type=int,
        default=None,
        help=(
            "Optional diagnostic override for the checkpoint's sender source-read "
            "bottleneck; zero disables it."
        ),
    )
    parser.add_argument(
        "--real-int4-codec",
        action="store_true",
        help="Actually pack/unpack INT4 bytes instead of using fake quantization.",
    )
    parser.add_argument(
        "--delta-base-checkpoint",
        default=None,
        help=(
            "Optional frozen base capsule encoded independently from the task "
            "checkpoint. Requires the real INT4 codec and --delta-layers."
        ),
    )
    parser.add_argument(
        "--delta-layers",
        default=None,
        help="Comma-separated task-delta layers composed onto the base packet.",
    )
    parser.add_argument(
        "--packet-position-mode",
        choices=("target", "canonical"),
        default="target",
        help=(
            "Quantize Keys after target repositioning or in receiver-independent "
            "canonical positions followed by receiver-side RoPE."
        ),
    )
    parser.add_argument(
        "--key-codec",
        choices=("cartesian", "rope_polar", "cartesian_k8"),
        default="cartesian",
        help=(
            "Cartesian signed INT4, Cartesian signed INT8, or equal-byte "
            "3-bit-radius/5-bit-phase Keys. Values remain INT4."
        ),
    )
    parser.add_argument(
        "--source-case-shift",
        type=int,
        default=0,
        help="Causal control: circularly shift the source dossier relative to each target.",
    )
    parser.add_argument(
        "--allow-training-overlap",
        action="store_true",
        help="Permit an explicitly labeled training-set diagnostic evaluation.",
    )
    args = parser.parse_args()
    if min(args.count, args.max_new_tokens) <= 0 or args.offset < 0:
        parser.error("count and token limit must be positive; offset non-negative")
    if args.untrained_slots is not None and args.untrained_slots < 1:
        parser.error("--untrained-slots must be positive")
    if args.source_read_layers is not None and args.source_read_layers < 0:
        parser.error("--source-read-layers must be non-negative")
    quant_bits_values = tuple(int(value) for value in args.eval_quant_bits.split(","))
    layer_patterns = tuple(value.strip() for value in args.layer_patterns.split(","))
    if (
        not quant_bits_values
        or any(value not in (4, 8, 16) for value in quant_bits_values)
        or len(set(quant_bits_values)) != len(quant_bits_values)
    ):
        parser.error("--eval-quant-bits must be a unique subset of 16,8,4")
    if not layer_patterns or len(set(layer_patterns)) != len(layer_patterns):
        parser.error("--layer-patterns must contain unique non-empty patterns")
    if args.real_int4_codec and quant_bits_values != (4,):
        parser.error("--real-int4-codec requires --eval-quant-bits 4")
    if args.packet_position_mode == "canonical" and not args.real_int4_codec:
        parser.error("canonical packet positions require --real-int4-codec")
    if args.key_codec == "rope_polar" and args.packet_position_mode != "canonical":
        parser.error("--key-codec rope_polar requires canonical packet positions")
    if args.key_codec == "cartesian_k8" and args.packet_position_mode != "canonical":
        parser.error("--key-codec cartesian_k8 requires canonical packet positions")
    if bool(args.delta_base_checkpoint) != bool(args.delta_layers):
        parser.error("--delta-base-checkpoint and --delta-layers require each other")
    if args.delta_base_checkpoint and not args.checkpoint:
        parser.error("base-plus-delta evaluation requires --checkpoint")
    if args.delta_base_checkpoint and not args.real_int4_codec:
        parser.error("base-plus-delta evaluation requires --real-int4-codec")
    delta_layers = None
    if args.delta_layers:
        try:
            delta_layers = frozenset(int(value) for value in args.delta_layers.split(","))
        except ValueError:
            parser.error("--delta-layers must contain comma-separated integers")
        if (
            not delta_layers
            or min(delta_layers) < 0
            or len(delta_layers) != len(args.delta_layers.split(","))
        ):
            parser.error("--delta-layers must contain unique non-negative integers")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "eval_quant_bits": list(quant_bits_values),
        "layer_patterns": list(layer_patterns),
        "delta_layers": None if delta_layers is None else sorted(delta_layers),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    for layer_pattern in layer_patterns:
        active_layers = active_layer_indices(
            layer_pattern, total_layers=model.config.num_hidden_layers
        )
        if delta_layers is not None and not delta_layers.issubset(active_layers):
            parser.error("every layer pattern must include all requested delta layers")
    if args.checkpoint:
        capsule, training_config = load_soft_tail_capsule(args.checkpoint)
    else:
        capsule = SoftTailCapsule(
            _initial_embeddings(model, tokenizer, slots=args.untrained_slots)
        )
        training_config = {
            "method": "deterministic_untrained_initialization",
            "slots": args.untrained_slots,
        }
    if args.source_read_layers is not None:
        capsule.source_read_layers = args.source_read_layers or None
    if capsule.source_read_layers is not None and not (
        1 <= capsule.source_read_layers < model.config.num_hidden_layers
    ):
        parser.error(
            "--source-read-layers must leave at least one source-reading and "
            "one source-masked model layer"
        )
    config["effective_source_read_layers"] = capsule.source_read_layers
    if capsule.embeddings.shape[1] != model.config.hidden_size:
        raise ValueError(
            "capsule hidden size does not match the evaluation model: "
            f"{capsule.embeddings.shape[1]} != {model.config.hidden_size}"
        )
    capsule = capsule.to(args.device).eval()
    delta_base_capsule = None
    if args.delta_base_checkpoint:
        delta_base_capsule, _ = load_soft_tail_capsule(
            args.delta_base_checkpoint
        )
        if delta_base_capsule.slots != capsule.slots:
            raise ValueError("delta base and task capsules have different slot counts")
        if delta_base_capsule.embeddings.shape[1] != model.config.hidden_size:
            raise ValueError("delta base capsule hidden size does not match the model")
        delta_base_capsule = delta_base_capsule.to(args.device).eval()
        config["delta_base_source_read_layers"] = (
            delta_base_capsule.source_read_layers
        )

    stage_system_ids, stage_system_cache = _system_cache(
        model, tokenizer, DEPENDENT_B_SYSTEM, "placeholder"
    )
    bridge_system_ids, bridge_system_cache = _system_cache(
        model, tokenizer, INTERMEDIATE_B_SYSTEM, INTERMEDIATE_QUERY
    )
    cases = _load_slice(args.dataset, offset=args.offset, count=args.count)
    source_cases = shifted_source_cases(cases, args.source_case_shift)
    training_ids = set(training_config.get("train_case_ids", []))
    training_keys = ("train_dataset", "train_offset", "train_count")
    if not training_ids and all(key in training_config for key in training_keys):
        training_ids = {
            case["id"]
            for _, case in _load_slice(
                training_config["train_dataset"],
                offset=int(training_config["train_offset"]),
                count=int(training_config["train_count"]),
            )
        }
    overlap = training_ids & {case["id"] for _, case in cases}
    if overlap and not args.allow_training_overlap:
        raise ValueError(f"evaluation overlaps checkpoint training IDs: {len(overlap)}")
    config["training_overlap_cases"] = len(overlap)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    rows = []
    for position, ((index, case), (source_index, source_case)) in enumerate(
        zip(cases, source_cases, strict=True), 1
    ):
        rows.extend(
            _evaluate_case(
                model,
                tokenizer,
                capsule,
                index=index,
                case=case,
                stage_system_ids=stage_system_ids,
                stage_system_cache=stage_system_cache,
                bridge_system_ids=bridge_system_ids,
                bridge_system_cache=bridge_system_cache,
                quant_bits_values=quant_bits_values,
                max_new_tokens=args.max_new_tokens,
                source_index=source_index,
                source_case=source_case,
                layer_patterns=layer_patterns,
                real_int4_codec=args.real_int4_codec,
                delta_base_capsule=delta_base_capsule,
                delta_layers=delta_layers,
                packet_position_mode=args.packet_position_mode,
                key_codec=args.key_codec,
            )
        )
        print(json.dumps({"event": "evaluation_progress", "position": position}), flush=True)

    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    summary = _summarize(
        rows,
        config={**config, "slots": capsule.slots, "training_config": training_config},
        model=model,
        slots=capsule.slots,
        training_seconds=0.0,
        training_means={},
        resident_source_cache_bytes=0,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def shifted_source_cases(cases, shift: int):
    if not cases:
        raise ValueError("evaluation cases must be non-empty")
    normalized = shift % len(cases)
    if shift and normalized == 0:
        raise ValueError("non-zero source shift must change every source assignment")
    return cases[normalized:] + cases[:normalized]


if __name__ == "__main__":
    main()
