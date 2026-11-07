from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from xmodel_kv.capsule_codec import pack_int4_capsule, unpack_int4_capsule
from xmodel_kv.policy_imprinting import build_chat_segments, prefill_legacy_cache, start_readout
from xmodel_kv.soft_tail_capsule import (
    BoundaryKVWriteAdapter,
    ProgressiveSoftTailCapsule,
    concatenate_tail_caches,
    load_soft_tail_capsule,
    reposition_tail_cache,
)

from .evaluate_hotpot_summary_transfer import format_source_user, load_slice
from .evaluate_musique_agent_a import SOURCE_ANSWER_SYSTEM
from .evaluate_musique_tail_transplant import active_layer_indices
from .train_musique_semantic_handoff import _load_model
from .train_musique_soft_tail_capsule import _initial_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-model wire invariance probe for a nested 4+4 capsule."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--core-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--refinement-slots", type=int, default=4)
    parser.add_argument("--refinement-rank", type=int, default=16)
    parser.add_argument("--boundary-layer", type=int, default=14)
    parser.add_argument("--layer-pattern", default="first_16")
    args = parser.parse_args()
    if min(args.refinement_slots, args.refinement_rank) < 1 or args.index < 0:
        parser.error("index must be non-negative and refinement sizes positive")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    core, core_config = load_soft_tail_capsule(args.core_checkpoint)
    core = core.to(args.device).eval()
    refinement_initial = _initial_embeddings(
        model, tokenizer, slots=args.refinement_slots
    )
    progressive = ProgressiveSoftTailCapsule(
        core,
        refinement_initial,
        refinement_boundary_write_adapter=BoundaryKVWriteAdapter(
            hidden_size=model.config.hidden_size,
            layer_index=args.boundary_layer,
            rank=args.refinement_rank,
        ),
    ).to(args.device).eval()
    _, document = load_slice(args.dataset, offset=args.index, count=1)[0]
    segments = build_chat_segments(
        tokenizer,
        system_prompt=SOURCE_ANSWER_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": format_source_user(document)}],
        enable_thinking=False,
    )
    system_ids = tuple(segments.prefix_ids)
    system_cache = prefill_legacy_cache(model, system_ids)
    query_ids = (*segments.history_ids, *segments.readout_ids)
    source = start_readout(
        model,
        legacy_cache=system_cache,
        cached_tokens=len(system_ids),
        input_ids=query_ids,
    )
    source_tokens = source.cached_tokens
    with torch.inference_mode():
        core_tail = progressive.materialize_core(
            model,
            prefix_cache=source.past_key_values,
            prefix_tokens=source_tokens,
        )
        reused_core, refinement_tail = progressive.materialize_parts(
            model,
            prefix_cache=source.past_key_values,
            prefix_tokens=source_tokens,
            core_tail=core_tail,
        )
        full_tail = concatenate_tail_caches(reused_core, refinement_tail)

        core_storage_reused = all(
            reused.data_ptr() == original.data_ptr()
            for original_layer, reused_layer in zip(core_tail, reused_core, strict=True)
            for original, reused in zip(original_layer, reused_layer, strict=True)
        )
        core_values_exact = all(
            torch.equal(original, reused)
            for original_layer, reused_layer in zip(core_tail, reused_core, strict=True)
            for original, reused in zip(original_layer, reused_layer, strict=True)
        )
        full_prefix_exact = all(
            torch.equal(full[:, :, : progressive.core_slots], core_tensor)
            for core_layer, full_layer in zip(core_tail, full_tail, strict=True)
            for core_tensor, full in zip(core_layer, full_layer, strict=True)
        )

        active_layers = active_layer_indices(
            args.layer_pattern, total_layers=model.config.num_hidden_layers
        )
        canonical_core = reposition_tail_cache(
            model, core_tail, source_start=source_tokens, target_start=0
        )
        canonical_refinement = reposition_tail_cache(
            model,
            refinement_tail,
            source_start=source_tokens + progressive.core_slots,
            target_start=progressive.core_slots,
        )
        canonical_full = reposition_tail_cache(
            model, full_tail, source_start=source_tokens, target_start=0
        )
        full_core_prefix = tuple(
            (
                key[:, :, : progressive.core_slots],
                value[:, :, : progressive.core_slots],
            )
            for key, value in canonical_full
        )
        core_packet = pack_int4_capsule(
            canonical_core,
            suffix_tokens=progressive.core_slots,
            active_layers=active_layers,
        )
        full_prefix_packet = pack_int4_capsule(
            full_core_prefix,
            suffix_tokens=progressive.core_slots,
            active_layers=active_layers,
        )
        refinement_packet = pack_int4_capsule(
            canonical_refinement,
            suffix_tokens=progressive.refinement_slots,
            active_layers=active_layers,
        )
        full_packet = pack_int4_capsule(
            canonical_full,
            suffix_tokens=progressive.slots,
            active_layers=active_layers,
        )
        decoded_core = unpack_int4_capsule(
            core_packet, dtype=canonical_full[0][0].dtype, device=args.device
        )
        decoded_refinement = unpack_int4_capsule(
            refinement_packet,
            dtype=canonical_full[0][0].dtype,
            device=args.device,
        )
        decoded_full = unpack_int4_capsule(
            full_packet, dtype=canonical_full[0][0].dtype, device=args.device
        )
        decoded_parts = concatenate_tail_caches(decoded_core, decoded_refinement)
        separate_decode_matches_monolithic = all(
            torch.equal(separate, monolithic)
            for separate_layer, monolithic_layer in zip(
                decoded_parts, decoded_full, strict=True
            )
            for separate, monolithic in zip(
                separate_layer, monolithic_layer, strict=True
            )
        )

    result = {
        "dataset": args.dataset,
        "index": args.index,
        "id": document.get("_id", str(args.index)),
        "model": args.model,
        "core_checkpoint": args.core_checkpoint,
        "core_checkpoint_slots": core_config.get("slots"),
        "source_tokens": source_tokens,
        "core_slots": progressive.core_slots,
        "refinement_slots": progressive.refinement_slots,
        "layer_pattern": args.layer_pattern,
        "core_storage_reused": core_storage_reused,
        "core_values_bitwise_exact": core_values_exact,
        "full_prefix_bitwise_exact": full_prefix_exact,
        "core_packet_bytes": len(core_packet),
        "refinement_packet_bytes": len(refinement_packet),
        "separate_cold_bytes": len(core_packet) + len(refinement_packet),
        "monolithic_full_packet_bytes": len(full_packet),
        "incremental_refinement_bytes": len(refinement_packet),
        "core_packet_sha256": hashlib.sha256(core_packet).hexdigest(),
        "full_prefix_packet_sha256": hashlib.sha256(full_prefix_packet).hexdigest(),
        "core_packet_bitwise_exact": core_packet == full_prefix_packet,
        "separate_decode_matches_monolithic": separate_decode_matches_monolithic,
    }
    if not all(
        result[field]
        for field in (
            "core_storage_reused",
            "core_values_bitwise_exact",
            "full_prefix_bitwise_exact",
            "core_packet_bitwise_exact",
            "separate_decode_matches_monolithic",
        )
    ):
        raise RuntimeError("progressive capsule invariance probe failed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
