from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from xmodel_kv.composable_delta import compose_int4_base_delta
from xmodel_kv.capsule_codec import pack_int4_capsule, unpack_int4_capsule
from xmodel_kv.hotpotqa import answer_em, answer_f1, clean_short_answer
from xmodel_kv.policy_imprinting import (
    build_chat_segments,
    prefill_legacy_cache,
    start_readout,
)
from xmodel_kv.semantic_kv_summary import concatenate_caches
from xmodel_kv.soft_tail_capsule import (
    load_soft_tail_capsule,
    move_legacy_cache,
    reposition_tail_cache,
)

from .evaluate_musique_agent_a import SOURCE_ANSWER_SYSTEM
from .evaluate_musique_tail_transplant import (
    active_layer_indices,
    mask_and_fake_quantize_suffix,
)
from .probe_semantic_kv_summary import _generate
from .train_musique_semantic_handoff import (
    INTERMEDIATE_B_SYSTEM,
    INTERMEDIATE_QUERY,
    _advance_source_one,
    _load_model,
)
from .train_musique_soft_tail_capsule import (
    _cache_nrms,
    _real_int4_round_trip,
    _require_composition_invariants,
    _require_delta_topology,
)


QUESTION_B_SYSTEM = (
    "You are Agent B. Answer the user's question using the handed-off compact state. "
    "Return only the exact short answer without explanation."
)
PROTOCOLS = ("state_readout", "question_conditioned")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot HotpotQA transfer test for a frozen MuSiQue soft-KV capsule, "
            "with no-state, shifted-state, and generated-answer Tail-KV controls."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--source-case-shift", type=int, default=1)
    parser.add_argument("--source-answer-max-new-tokens", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--quant-bits", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--layer-pattern", default="first_30")
    parser.add_argument(
        "--real-int4-codec",
        action="store_true",
        help="Actually frame, pack, and unpack INT4 caches.",
    )
    parser.add_argument(
        "--delta-base-checkpoint",
        default=None,
        help="Optional reusable base capsule for independently encoded task deltas.",
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
        help="Encode in receiver target positions or canonical positions 0..slots-1.",
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
        "--generated-tail-layer-pattern",
        default="all",
        help="Tail-KV uses its own frozen layer operating point (default: all layers)",
    )
    parser.add_argument(
        "--protocols",
        default=",".join(PROTOCOLS),
        help="comma-separated subset of state_readout,question_conditioned",
    )
    args = parser.parse_args()
    if min(
        args.offset,
        args.source_case_shift,
    ) < 0 or min(
        args.count,
        args.source_answer_max_new_tokens,
        args.max_new_tokens,
    ) <= 0:
        parser.error("offset/shift must be non-negative and counts positive")
    protocols = tuple(value.strip() for value in args.protocols.split(","))
    if (
        not protocols
        or len(set(protocols)) != len(protocols)
        or any(value not in PROTOCOLS for value in protocols)
    ):
        parser.error("--protocols must contain a unique subset of known protocols")
    if args.source_case_shift % args.count == 0:
        parser.error("source-case-shift must not be zero modulo count")
    if bool(args.delta_base_checkpoint) != bool(args.delta_layers):
        parser.error("--delta-base-checkpoint and --delta-layers require each other")
    if args.delta_base_checkpoint and not args.real_int4_codec:
        parser.error("base-plus-delta evaluation requires --real-int4-codec")
    if args.real_int4_codec and args.quant_bits != 4:
        parser.error("--real-int4-codec requires --quant-bits 4")
    if args.packet_position_mode == "canonical" and not args.real_int4_codec:
        parser.error("canonical packet positions require --real-int4-codec")
    if args.key_codec == "rope_polar" and args.packet_position_mode != "canonical":
        parser.error("--key-codec rope_polar requires canonical packet positions")
    if args.key_codec == "cartesian_k8" and args.packet_position_mode != "canonical":
        parser.error("--key-codec cartesian_k8 requires canonical packet positions")
    delta_layers = None
    if args.delta_layers:
        values = args.delta_layers.split(",")
        try:
            delta_layers = frozenset(int(value) for value in values)
        except ValueError:
            parser.error("--delta-layers must contain comma-separated integers")
        if not delta_layers or min(delta_layers) < 0 or len(delta_layers) != len(values):
            parser.error("--delta-layers must contain unique non-negative integers")
    if args.device.startswith("cuda"):
        # timed_cuda synchronizes the current CUDA device.  Explicitly set it
        # because device_map does not guarantee changing torch's process-local
        # current-device state for cuda:1/cuda:2 workers.
        torch.cuda.set_device(torch.device(args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "protocols": list(protocols),
        "delta_layers": None if delta_layers is None else sorted(delta_layers),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    cases = load_slice(args.dataset, offset=args.offset, count=args.count)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = _load_model(args.model, args.device, getattr(torch, args.dtype))
    capsule, training_config = load_soft_tail_capsule(args.checkpoint)
    if capsule.embeddings.shape[1] != model.config.hidden_size:
        raise ValueError("capsule hidden size does not match evaluation model")
    capsule = capsule.to(args.device).eval()
    delta_base_capsule = None
    if args.delta_base_checkpoint:
        delta_base_capsule, _ = load_soft_tail_capsule(args.delta_base_checkpoint)
        if delta_base_capsule.slots != capsule.slots:
            raise ValueError("delta base and task capsules have different slot counts")
        if delta_base_capsule.embeddings.shape[1] != model.config.hidden_size:
            raise ValueError("delta base capsule hidden size does not match the model")
        delta_base_capsule = delta_base_capsule.to(args.device).eval()
        config["delta_base_source_read_layers"] = (
            delta_base_capsule.source_read_layers
        )
    active_layers = active_layer_indices(
        args.layer_pattern, total_layers=model.config.num_hidden_layers
    )
    if delta_layers is not None and not delta_layers.issubset(active_layers):
        parser.error("the layer pattern must include every requested delta layer")
    generated_tail_active_layers = active_layer_indices(
        args.generated_tail_layer_pattern,
        total_layers=model.config.num_hidden_layers,
    )
    active_heads = frozenset(range(model.config.num_key_value_heads))

    target_specs = {
        protocol: prepare_target_spec(model, tokenizer, protocol)
        for protocol in protocols
    }
    source_system_segments = build_chat_segments(
        tokenizer,
        system_prompt=SOURCE_ANSWER_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": "placeholder"}],
        enable_thinking=False,
    )
    source_system_ids = tuple(source_system_segments.prefix_ids)
    source_system_cache = prefill_legacy_cache(model, source_system_ids)

    source_states = []
    for position, (index, document) in enumerate(cases, 1):
        state = prepare_source_state(
            model,
            tokenizer,
            capsule,
            base_capsule=delta_base_capsule,
            index=index,
            document=document,
            source_system_ids=source_system_ids,
            source_system_cache=source_system_cache,
            max_new_tokens=args.source_answer_max_new_tokens,
        )
        source_states.append(state)
        print(
            json.dumps(
                {
                    "event": "source",
                    "position": position,
                    "index": index,
                    "id": state["id"],
                    "source_tokens": state["source_tokens"],
                    "source_answer": state["source_answer"],
                    "source_answer_em": state["source_answer_em"],
                    "capsule_emission_ms": state["capsule_emission_ms"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = []
    for ordinal, ((index, document), correct_state) in enumerate(
        zip(cases, source_states, strict=True)
    ):
        shifted_state = source_states[
            (ordinal + args.source_case_shift) % len(source_states)
        ]
        for protocol in protocols:
            row = evaluate_target(
                model,
                tokenizer,
                index=index,
                document=document,
                protocol=protocol,
                target_spec=target_specs[protocol],
                correct_state=correct_state,
                shifted_state=shifted_state,
                quant_bits=args.quant_bits,
                capsule_active_layers=active_layers,
                generated_tail_active_layers=generated_tail_active_layers,
                active_heads=active_heads,
                max_new_tokens=args.max_new_tokens,
                real_int4_codec=args.real_int4_codec,
                delta_layers=delta_layers,
                packet_position_mode=args.packet_position_mode,
                key_codec=args.key_codec,
            )
            rows.append(row)
            print(json.dumps({"event": "evaluate", **row}, ensure_ascii=False), flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.packet_position_mode == "canonical":
        config["canonical_packet_target_invariance_audited"] = (
            require_canonical_packet_invariance(rows, protocols=protocols)
        )
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results_path = output_dir / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    summary = summarize(
        rows,
        config=config,
        model_config=model.config,
        slots=capsule.slots,
        active_layers=len(active_layers),
        generated_tail_active_layers=len(generated_tail_active_layers),
        training_config=training_config,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"event": "complete", "summary": summary}), flush=True)


def load_slice(path: str | Path, *, offset: int, count: int) -> list[tuple[int, dict]]:
    selected = []
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            if offset <= index < offset + count and line.strip():
                selected.append((index, json.loads(line)))
            if index >= offset + count:
                break
    if len(selected) != count:
        raise ValueError("dataset does not contain the requested slice")
    required = {"input", "context", "answers"}
    if any(not required.issubset(document) for _, document in selected):
        raise ValueError("Hotpot rows require input, context, and answers")
    if any(not document["answers"] for _, document in selected):
        raise ValueError("Hotpot rows require at least one gold answer")
    return selected


def format_source_user(document: dict) -> str:
    return (
        f"<DOSSIER>\n{document['context'].strip()}\n</DOSSIER>\n\n"
        f"Question: {document['input'].strip()}\n"
        "Output only the exact short answer."
    )


def prepare_target_spec(model, tokenizer, protocol: str) -> dict:
    if protocol == "state_readout":
        system_prompt = INTERMEDIATE_B_SYSTEM
    elif protocol == "question_conditioned":
        system_prompt = QUESTION_B_SYSTEM
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    placeholder_query = (
        INTERMEDIATE_QUERY if protocol == "state_readout" else "placeholder"
    )
    segments = build_chat_segments(
        tokenizer,
        system_prompt=system_prompt,
        tools=[],
        history=[{"role": "user", "content": placeholder_query}],
        enable_thinking=False,
    )
    system_ids = tuple(segments.prefix_ids)
    return {
        "system_prompt": system_prompt,
        "system_ids": system_ids,
        "system_cache": prefill_legacy_cache(model, system_ids),
    }


@torch.inference_mode()
def prepare_source_state(
    model,
    tokenizer,
    capsule,
    *,
    base_capsule=None,
    index: int,
    document: dict,
    source_system_ids,
    source_system_cache,
    max_new_tokens: int,
) -> dict:
    segments = build_chat_segments(
        tokenizer,
        system_prompt=SOURCE_ANSWER_SYSTEM,
        tools=[],
        history=[{"role": "user", "content": format_source_user(document)}],
        enable_thinking=False,
    )
    if tuple(segments.prefix_ids) != tuple(source_system_ids):
        raise ValueError("source system prefix changed across documents")
    query_ids = (*segments.history_ids, *segments.readout_ids)
    source_state, source_prefill_ms = timed_cuda(
        lambda: start_readout(
            model,
            legacy_cache=source_system_cache,
            cached_tokens=len(source_system_ids),
            input_ids=query_ids,
        )
    )
    source_tokens = source_state.cached_tokens
    capsule_tail, capsule_emission_ms = timed_cuda(
        lambda: capsule.materialize(
            model,
            prefix_cache=source_state.past_key_values,
            prefix_tokens=source_tokens,
        )
    )
    if base_capsule is None:
        base_capsule_tail = None
        base_capsule_emission_ms = None
    else:
        base_capsule_tail, base_capsule_emission_ms = timed_cuda(
            lambda: base_capsule.materialize(
                model,
                prefix_cache=source_state.past_key_values,
                prefix_tokens=source_tokens,
            )
        )
    generated_ids, generated_state, source_decode_ms = generate_from_state(
        model, tokenizer, source_state, max_new_tokens=max_new_tokens
    )
    source_answer = clean_short_answer(
        tokenizer.decode(generated_ids, skip_special_tokens=False)
    )
    if not generated_ids or not source_answer:
        raise ValueError(f"source generated an empty answer for document {index}")
    generated_tail = slice_cache_tail(generated_state.past_key_values, len(generated_ids))
    golds = list(document["answers"])
    return {
        "index": index,
        "id": document.get("_id", str(index)),
        "source_tokens": source_tokens,
        "source_answer": source_answer,
        "source_answer_tokens": len(generated_ids),
        "source_answer_em": answer_em(source_answer, golds),
        "source_answer_f1": answer_f1(source_answer, golds),
        "source_prefill_ms": source_prefill_ms,
        "capsule_emission_ms": capsule_emission_ms,
        "base_capsule_emission_ms": base_capsule_emission_ms,
        "source_decode_ms": source_decode_ms,
        "capsule_tail": move_legacy_cache(capsule_tail, "cpu"),
        "base_capsule_tail": (
            None
            if base_capsule_tail is None
            else move_legacy_cache(base_capsule_tail, "cpu")
        ),
        "capsule_source_start": source_tokens,
        "generated_tail": move_legacy_cache(generated_tail, "cpu"),
        "generated_source_start": source_tokens,
    }


@torch.inference_mode()
def evaluate_target(
    model,
    tokenizer,
    *,
    index: int,
    document: dict,
    protocol: str,
    target_spec: dict,
    correct_state: dict,
    shifted_state: dict,
    quant_bits: int,
    capsule_active_layers: frozenset[int],
    generated_tail_active_layers: frozenset[int],
    active_heads: frozenset[int],
    max_new_tokens: int,
    real_int4_codec: bool = False,
    delta_layers: frozenset[int] | None = None,
    packet_position_mode: str = "target",
    key_codec: str = "cartesian",
) -> dict:
    if delta_layers is not None and not real_int4_codec:
        raise ValueError("base-plus-delta composition requires the real INT4 codec")
    if packet_position_mode not in ("target", "canonical"):
        raise ValueError("packet position mode must be target or canonical")
    if packet_position_mode == "canonical" and not real_int4_codec:
        raise ValueError("canonical packet positions require the real INT4 codec")
    if key_codec not in ("cartesian", "rope_polar", "cartesian_k8"):
        raise ValueError(
            "key codec must be cartesian, rope_polar, or cartesian_k8"
        )
    if key_codec == "rope_polar" and packet_position_mode != "canonical":
        raise ValueError("rope_polar Keys require canonical packet positions")
    if key_codec == "cartesian_k8" and packet_position_mode != "canonical":
        raise ValueError("cartesian_k8 Keys require canonical packet positions")
    query = INTERMEDIATE_QUERY if protocol == "state_readout" else document["input"]
    segments = build_chat_segments(
        tokenizer,
        system_prompt=target_spec["system_prompt"],
        tools=[],
        history=[{"role": "user", "content": query}],
        enable_thinking=False,
    )
    if tuple(segments.prefix_ids) != tuple(target_spec["system_ids"]):
        raise ValueError("target system prefix changed across documents")
    query_ids = (*segments.history_ids, *segments.readout_ids)
    target_start = len(target_spec["system_ids"])
    system_cache = target_spec["system_cache"]
    device = model.model.embed_tokens.weight.device

    no_summary, no_summary_ms = generate_target(
        model,
        tokenizer,
        cache=system_cache,
        cached_tokens=target_start,
        query_ids=query_ids,
        max_new_tokens=max_new_tokens,
    )
    arm_outputs = {}
    for arm, state, tail_name, start_name in (
        ("capsule", correct_state, "capsule_tail", "capsule_source_start"),
        ("shifted_capsule", shifted_state, "capsule_tail", "capsule_source_start"),
        ("generated_tail", correct_state, "generated_tail", "generated_source_start"),
    ):
        compact = move_legacy_cache(state[tail_name], device)
        slots = int(compact[0][0].shape[-2])
        arm_active_layers = (
            generated_tail_active_layers
            if arm == "generated_tail"
            else capsule_active_layers
        )

        def prepare_compact():
            packet_target_start = (
                0 if packet_position_mode == "canonical" else target_start
            )
            repositioned = reposition_tail_cache(
                model,
                compact,
                source_start=state[start_name],
                target_start=packet_target_start,
            )
            if delta_layers is not None and arm != "generated_tail":
                base_compact = move_legacy_cache(state["base_capsule_tail"], device)
                base_repositioned = reposition_tail_cache(
                    model,
                    base_compact,
                    source_start=state[start_name],
                    target_start=packet_target_start,
                )
                _require_delta_topology(
                    base_repositioned,
                    repositioned,
                    base_layers=arm_active_layers,
                    delta_layers=delta_layers,
                )
                composition = compose_int4_base_delta(
                    base_repositioned,
                    repositioned,
                    suffix_tokens=slots,
                    base_layers=arm_active_layers,
                    delta_layers=delta_layers,
                    key_codec=key_codec,
                )
                _require_composition_invariants(
                    composition,
                    active_layers=arm_active_layers,
                    delta_layers=delta_layers,
                )
                return composition.cache, composition, repositioned
            if real_int4_codec:
                if packet_position_mode == "canonical":
                    packet = pack_int4_capsule(
                        repositioned,
                        suffix_tokens=slots,
                        active_layers=arm_active_layers,
                        key_codec=key_codec,
                    )
                    prepared = unpack_int4_capsule(
                        packet,
                        dtype=repositioned[0][0].dtype,
                        device=repositioned[0][0].device,
                    )
                    packet_metadata = {
                        "packet_bytes": len(packet),
                        "packet_sha256": hashlib.sha256(packet).hexdigest(),
                    }
                else:
                    prepared, packet_bytes = _real_int4_round_trip(
                        repositioned,
                        slots=slots,
                        active_layers=arm_active_layers,
                        key_codec=key_codec,
                    )
                    packet_metadata = {"packet_bytes": packet_bytes}
                return prepared, packet_metadata, None
            prepared = mask_and_fake_quantize_suffix(
                repositioned,
                suffix_tokens=slots,
                bits=quant_bits,
                active_layers=arm_active_layers,
                active_heads=active_heads,
            )
            return prepared, None, None

        prepared_bundle, prepare_ms = timed_cuda(prepare_compact)
        prepared, packet_metadata, unquantized_task = prepared_bundle
        wire_metadata = {}
        if delta_layers is not None and arm != "generated_tail":
            composition = packet_metadata
            direct_cache, direct_packet_bytes = _real_int4_round_trip(
                unquantized_task,
                slots=slots,
                active_layers=arm_active_layers,
                key_codec=key_codec,
            )
            wire_metadata = {
                "base_packet_bytes": composition.base_packet_bytes,
                "delta_packet_bytes": composition.delta_packet_bytes,
                "cold_packet_bytes": composition.cold_packet_bytes,
                "incremental_packet_bytes": composition.incremental_packet_bytes,
                "direct_packet_bytes": direct_packet_bytes,
                "delta_vs_direct_nrms": _cache_nrms(
                    composition.cache, direct_cache, layers=delta_layers
                ),
                "delta_vs_task_nrms": _cache_nrms(
                    composition.cache, unquantized_task, layers=delta_layers
                ),
            }
            if packet_position_mode == "canonical":
                wire_metadata.update(
                    {
                        "base_packet_sha256": hashlib.sha256(
                            composition.base_packet
                        ).hexdigest(),
                        "delta_packet_sha256": hashlib.sha256(
                            composition.delta_packet
                        ).hexdigest(),
                    }
                )
        elif real_int4_codec:
            wire_metadata = packet_metadata
        if packet_position_mode == "canonical":
            prepared = reposition_tail_cache(
                model,
                prepared,
                source_start=0,
                target_start=target_start,
            )
        generation, generation_ms = generate_target(
            model,
            tokenizer,
            cache=concatenate_caches(system_cache, prepared),
            cached_tokens=target_start + slots,
            query_ids=query_ids,
            max_new_tokens=max_new_tokens,
        )
        arm_outputs[arm] = {
            "generation": generation,
            "tokens": slots,
            "source_id": state["id"],
            "prepare_ms": prepare_ms,
            "generation_ms": generation_ms,
            **wire_metadata,
        }
        del compact, prepared, prepared_bundle

    golds = list(document["answers"])
    row = {
        "index": index,
        "id": document.get("_id", str(index)),
        "dataset": document.get("dataset", "hotpotqa"),
        "protocol": protocol,
        "packet_position_mode": packet_position_mode,
        "key_codec": key_codec,
        "question": document["input"],
        "gold_answers": golds,
        "source_tokens": correct_state["source_tokens"],
        "source_answer": correct_state["source_answer"],
        "source_answer_tokens": correct_state["source_answer_tokens"],
        "source_answer_em": correct_state["source_answer_em"],
        "source_answer_f1": correct_state["source_answer_f1"],
        "source_prefill_ms": correct_state["source_prefill_ms"],
        "capsule_emission_ms": correct_state["capsule_emission_ms"],
        "base_capsule_emission_ms": correct_state["base_capsule_emission_ms"],
        "source_decode_ms": correct_state["source_decode_ms"],
        "no_summary": no_summary,
        "no_summary_em": answer_em(no_summary, golds),
        "no_summary_f1": answer_f1(no_summary, golds),
        "no_summary_generation_ms": no_summary_ms,
    }
    for arm, output in arm_outputs.items():
        generation = output.pop("generation")
        row[arm] = generation
        row[f"{arm}_em"] = answer_em(generation, golds)
        row[f"{arm}_f1"] = answer_f1(generation, golds)
        for key, value in output.items():
            row[f"{arm}_{key}"] = value
    return row


def require_canonical_packet_invariance(
    rows: list[dict], *, protocols: tuple[str, ...]
) -> bool:
    """Assert receiver-prefix-independent packet identity across protocols."""

    if len(protocols) < 2:
        return False
    by_id: dict[str, list[dict]] = {}
    for row in rows:
        by_id.setdefault(row["id"], []).append(row)
    for case_id, case_rows in by_id.items():
        if {row["protocol"] for row in case_rows} != set(protocols):
            raise RuntimeError(f"canonical packet audit is incomplete for {case_id}")
        first = case_rows[0]
        for arm in ("capsule", "shifted_capsule"):
            hash_fields = (
                (f"{arm}_base_packet_sha256", f"{arm}_delta_packet_sha256")
                if f"{arm}_base_packet_sha256" in first
                else (f"{arm}_packet_sha256",)
            )
            for field in hash_fields:
                hashes = {row[field] for row in case_rows}
                if len(hashes) != 1:
                    raise RuntimeError(
                        f"canonical packet {field} depends on receiver protocol"
                    )
    return True


def generate_target(
    model,
    tokenizer,
    *,
    cache,
    cached_tokens: int,
    query_ids,
    max_new_tokens: int,
) -> tuple[str, float]:
    generation, elapsed_ms = timed_cuda(
        lambda: _generate(
            model,
            tokenizer,
            legacy_cache=cache,
            cached_tokens=cached_tokens,
            query_ids=query_ids,
            max_new_tokens=max_new_tokens,
        )
    )
    return clean_short_answer(generation), elapsed_ms


@torch.inference_mode()
def generate_from_state(model, tokenizer, state, *, max_new_tokens: int):
    generated = []

    def run():
        nonlocal state
        for _ in range(max_new_tokens):
            next_id = int(state.logits.argmax(dim=-1).item())
            candidate = tokenizer.decode(
                [*generated, next_id], skip_special_tokens=False
            )
            if (
                next_id == tokenizer.eos_token_id
                or "<|im_end|>" in candidate
                or "<|eot_id|>" in candidate
                or "<|end_of_text|>" in candidate
            ):
                break
            generated.append(next_id)
            state = _advance_source_one(model, state, next_id)
        return tuple(generated), state

    result, elapsed_ms = timed_cuda(run)
    return (*result, elapsed_ms)


def slice_cache_tail(cache: Any, tokens: int):
    layers = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    if tokens < 1 or any(key.shape[-2] < tokens for key, _ in layers):
        raise ValueError("tail length is invalid for generated cache")
    return tuple(
        (key[:, :, -tokens:].detach(), value[:, :, -tokens:].detach())
        for key, value in layers
    )


def timed_cuda(function: Callable[[], Any]) -> tuple[Any, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000


def wire_payload_bytes(
    model_config,
    *,
    tokens: float,
    active_layers: int,
    bits: int,
) -> float:
    kv_heads = model_config.num_key_value_heads
    head_dim = getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )
    payload = tokens * active_layers * kv_heads * head_dim * 2 * bits / 8
    if bits < 16:
        payload += tokens * active_layers * kv_heads * 2 * 2
    return float(payload)


def summarize(
    rows: list[dict],
    *,
    config: dict,
    model_config,
    slots: int,
    active_layers: int,
    generated_tail_active_layers: int,
    training_config: dict | None,
) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty results")
    by_protocol = {}
    for protocol in dict.fromkeys(row["protocol"] for row in rows):
        selected = [row for row in rows if row["protocol"] == protocol]
        mean = lambda key: sum(float(row[key]) for row in selected) / len(selected)
        arms = ("no_summary", "capsule", "shifted_capsule", "generated_tail")
        metrics = {}
        for arm in arms:
            metrics[arm] = {
                "em": mean(f"{arm}_em"),
                "f1": mean(f"{arm}_f1"),
                "generation_ms": mean(
                    "no_summary_generation_ms"
                    if arm == "no_summary"
                    else f"{arm}_generation_ms"
                ),
            }
            if arm != "no_summary":
                metrics[arm]["prepare_ms"] = mean(f"{arm}_prepare_ms")
                if f"{arm}_packet_bytes" in selected[0]:
                    metrics[arm]["actual_packet_bytes"] = mean(
                        f"{arm}_packet_bytes"
                    )
                if f"{arm}_cold_packet_bytes" in selected[0]:
                    for key in (
                        "base_packet_bytes",
                        "delta_packet_bytes",
                        "cold_packet_bytes",
                        "incremental_packet_bytes",
                        "direct_packet_bytes",
                        "delta_vs_direct_nrms",
                        "delta_vs_task_nrms",
                    ):
                        metrics[arm][key] = mean(f"{arm}_{key}")
        metrics["capsule"]["wins_vs_no_summary"] = sum(
            row["capsule_f1"] > row["no_summary_f1"] for row in selected
        )
        metrics["capsule"]["losses_vs_no_summary"] = sum(
            row["capsule_f1"] < row["no_summary_f1"] for row in selected
        )
        metrics["capsule"]["wins_vs_shift"] = sum(
            row["capsule_f1"] > row["shifted_capsule_f1"] for row in selected
        )
        metrics["capsule"]["losses_vs_shift"] = sum(
            row["capsule_f1"] < row["shifted_capsule_f1"] for row in selected
        )
        if "cold_packet_bytes" in metrics["capsule"]:
            metrics["capsule"]["wire_payload_bytes"] = metrics["capsule"][
                "cold_packet_bytes"
            ]
        elif "actual_packet_bytes" in metrics["capsule"]:
            metrics["capsule"]["wire_payload_bytes"] = metrics["capsule"][
                "actual_packet_bytes"
            ]
        else:
            metrics["capsule"]["wire_payload_bytes"] = wire_payload_bytes(
                model_config,
                tokens=slots,
                active_layers=active_layers,
                bits=config["quant_bits"],
            )
        mean_tail_tokens = mean("generated_tail_tokens")
        metrics["generated_tail"]["mean_tokens"] = mean_tail_tokens
        metrics["generated_tail"]["mean_wire_payload_bytes"] = (
            metrics["generated_tail"].get("actual_packet_bytes")
            or wire_payload_bytes(
                model_config,
                tokens=mean_tail_tokens,
                active_layers=generated_tail_active_layers,
                bits=config["quant_bits"],
            )
        )
        by_protocol[protocol] = {
            "cases": len(selected),
            "source_answer_em": mean("source_answer_em"),
            "source_answer_f1": mean("source_answer_f1"),
            "mean_source_tokens": mean("source_tokens"),
            "mean_source_prefill_ms": mean("source_prefill_ms"),
            "mean_capsule_emission_ms": mean("capsule_emission_ms"),
            "mean_base_capsule_emission_ms": (
                None
                if selected[0].get("base_capsule_emission_ms") is None
                else mean("base_capsule_emission_ms")
            ),
            "mean_source_decode_ms": mean("source_decode_ms"),
            "mean_capsule_postprefill_handoff_ms": (
                mean("capsule_emission_ms")
                + mean("capsule_prepare_ms")
                + mean("capsule_generation_ms")
            ),
            "mean_generated_tail_postprefill_handoff_ms": (
                mean("source_decode_ms")
                + mean("generated_tail_prepare_ms")
                + mean("generated_tail_generation_ms")
            ),
            "arms": metrics,
        }
    return {
        "config": config,
        "checkpoint_training_config": training_config,
        "by_protocol": by_protocol,
    }


if __name__ == "__main__":
    main()
