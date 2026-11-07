from __future__ import annotations

from dataclasses import dataclass

import torch

from .capsule_codec import pack_int4_capsule, unpack_int4_capsule
from .policy_imprinting import build_chat_segments, prefill_legacy_cache
from .receiver_lens import extend_cache_with_tokens
from .semantic_kv_summary import concatenate_caches
from .soft_tail_capsule import reposition_tail_cache


LENS_INITIALIZATION_TEXT = " internal memory answer state"


@dataclass(frozen=True)
class LensRoute:
    history_ids: tuple[int, ...]
    readout_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]


@dataclass
class LensDocument:
    index: int
    case_id: str
    source_ids: tuple[int, ...]
    stage: LensRoute
    bridge: LensRoute
    packet: bytes | None = None


def lens_token_ids(tokenizer, *, slots: int = 4) -> tuple[int, ...]:
    ids = tuple(
        int(token_id)
        for token_id in tokenizer.encode(
            LENS_INITIALIZATION_TEXT, add_special_tokens=False
        )
    )
    if len(ids) != slots:
        raise ValueError(
            f"lens initialization must tokenize to exactly {slots} tokens; got {len(ids)}"
        )
    return ids


def prepare_lens_document(
    tokenizer,
    *,
    index: int,
    case: dict,
    source_system_prompt: str,
    stage_system_prompt: str,
    stage_system_ids,
    stage_query: str,
    bridge_system_prompt: str,
    bridge_system_ids,
    bridge_query: str,
) -> LensDocument:
    documents = "\n\n".join(
        f"[DOC-{number}] {document['title']}\n{document['text']}"
        for number, document in enumerate(case["agent_a"]["documents"], 1)
    )
    source_user = (
        f"<DOSSIER>\n{documents}\n</DOSSIER>\n\n"
        f"Question: {case['agent_a']['question']}\n"
        "Output only the exact short answer."
    )
    source = build_chat_segments(
        tokenizer,
        system_prompt=source_system_prompt,
        tools=[],
        history=[{"role": "user", "content": source_user}],
        enable_thinking=False,
    )
    stage = prepare_route(
        tokenizer,
        system_prompt=stage_system_prompt,
        expected_system_ids=stage_system_ids,
        query=stage_query,
        answer=case["agent_b"]["gold_answer"],
    )
    bridge = prepare_route(
        tokenizer,
        system_prompt=bridge_system_prompt,
        expected_system_ids=bridge_system_ids,
        query=bridge_query,
        answer=case["agent_a"]["gold_answer"],
    )
    return LensDocument(
        index=index,
        case_id=case["id"],
        source_ids=tuple(
            (*source.prefix_ids, *source.history_ids, *source.readout_ids)
        ),
        stage=stage,
        bridge=bridge,
    )


def prepare_route(
    tokenizer,
    *,
    system_prompt: str,
    expected_system_ids,
    query: str,
    answer: str,
) -> LensRoute:
    segments = build_chat_segments(
        tokenizer,
        system_prompt=system_prompt,
        tools=[],
        history=[{"role": "user", "content": query}],
        enable_thinking=False,
    )
    if tuple(segments.prefix_ids) != tuple(expected_system_ids):
        raise ValueError("target system prefix changed across tasks")
    answer_ids = tuple(tokenizer.encode(answer, add_special_tokens=False))
    if not answer_ids:
        raise ValueError("answer tokenization is empty")
    return LensRoute(
        tuple(segments.history_ids),
        tuple(segments.readout_ids),
        answer_ids,
    )


@torch.inference_mode()
def materialize_canonical_packet(
    model,
    capsule,
    document: LensDocument,
) -> bytes:
    source_cache = prefill_legacy_cache(model, document.source_ids)
    source_tail = capsule.materialize(
        model,
        prefix_cache=source_cache,
        prefix_tokens=len(document.source_ids),
    )
    canonical = reposition_tail_cache(
        model,
        source_tail,
        source_start=len(document.source_ids),
        target_start=0,
    )
    return pack_int4_capsule(
        canonical,
        suffix_tokens=capsule.slots,
        active_layers=frozenset(range(model.config.num_hidden_layers)),
    )


def decode_packet_for_system(
    model,
    packet: bytes,
    *,
    system_tokens: int,
):
    device = model.model.embed_tokens.weight.device
    dtype = model.model.embed_tokens.weight.dtype
    canonical = unpack_int4_capsule(packet, dtype=dtype, device=device)
    return reposition_tail_cache(
        model,
        canonical,
        source_start=0,
        target_start=system_tokens,
    )


def build_route_prefix(
    model,
    *,
    system_cache,
    system_tokens: int,
    semantic_tail,
    semantic_slots: int,
    history_ids,
):
    with_memory = concatenate_caches(system_cache, semantic_tail)
    cached_tokens = system_tokens + semantic_slots
    full = extend_cache_with_tokens(
        model,
        prefix_cache=with_memory,
        prefix_tokens=cached_tokens,
        token_ids=history_ids,
    )
    return full, cached_tokens + len(history_ids)
