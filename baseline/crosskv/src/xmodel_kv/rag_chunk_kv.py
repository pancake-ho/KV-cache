from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .rope import apply_rope, model_rope_cos_sin, remove_rope

LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]
ProducerMode = Literal["sequential", "independent"]


@dataclass(frozen=True)
class DocumentChunkKV:
    """CPU-resident KV produced for one tokenized RAG document chunk."""

    name: str
    token_ids: tuple[int, ...]
    source_start: int
    layers: LegacyCache

    @property
    def tokens(self) -> int:
        return len(self.token_ids)

    @property
    def kv_bytes(self) -> int:
        return _cache_bytes(self.layers)


@dataclass(frozen=True)
class ChunkKVBundle:
    """System-conditioned document chunks emitted by a producer."""

    mode: ProducerMode
    system_ids: tuple[int, ...]
    system_layers: LegacyCache
    chunks: tuple[DocumentChunkKV, ...]
    forward_tokens: int

    @property
    def document_tokens(self) -> int:
        return sum(chunk.tokens for chunk in self.chunks)

    @property
    def kv_bytes(self) -> int:
        return _cache_bytes(self.system_layers) + sum(
            chunk.kv_bytes for chunk in self.chunks
        )


@dataclass(frozen=True)
class ComposedChunkKV:
    """CPU cache and metadata consumed without document-token prefill."""

    layers: LegacyCache
    system_tokens: int
    document_tokens: int
    chunk_ranges: tuple[tuple[str, int, int], ...]
    relocated_tokens: int

    @property
    def cold_start(self) -> int:
        return self.system_tokens

    @property
    def cold_end(self) -> int:
        return self.system_tokens + self.document_tokens

    @property
    def kv_bytes(self) -> int:
        return _cache_bytes(self.layers)

    def page_ranges(self, page_size: int) -> tuple[tuple[int, int], ...]:
        """Return document-relative pages that never cross chunk boundaries."""
        if page_size < 1:
            raise ValueError("page_size must be positive")
        ranges: list[tuple[int, int]] = []
        for _, absolute_start, absolute_end in self.chunk_ranges:
            chunk_start = absolute_start - self.cold_start
            chunk_end = absolute_end - self.cold_start
            for start in range(chunk_start, chunk_end, page_size):
                ranges.append((start, min(start + page_size, chunk_end)))
        return tuple(ranges)

    def select_seed_pages(
        self,
        *,
        document_token_ids: Sequence[int],
        query_token_ids: Sequence[int],
        page_size: int,
        budget_ratio: float,
        min_pages: int,
        seed_budget_ratio: float = 0.75,
    ) -> tuple[int, ...]:
        """Select chunk landmarks and lexical evidence pages for GPU seeding."""
        document_token_ids = tuple(int(token) for token in document_token_ids)
        if len(document_token_ids) != self.document_tokens:
            raise ValueError("document token IDs do not match the composed cache")
        if not 0 < budget_ratio <= 1 or not 0 <= seed_budget_ratio <= 1:
            raise ValueError("page and seed budget ratios must lie in [0, 1]")
        ranges = self.page_ranges(page_size)
        budget = min(
            len(ranges),
            max(min_pages, math.ceil(len(ranges) * budget_ratio)),
        )
        if budget == 0:
            return ()

        page_tokens = [
            set(document_token_ids[start:end]) for start, end in ranges
        ]
        frequencies = Counter(
            token for tokens in page_tokens for token in tokens
        )
        query_tokens = {int(token) for token in query_token_ids}
        lexical_scores = [
            sum(
                math.log((1 + len(ranges)) / (1 + frequencies[token])) + 1
                for token in tokens & query_tokens
            )
            for tokens in page_tokens
        ]

        chunk_starts = {
            start - self.cold_start for _, start, _ in self.chunk_ranges
        }
        title_pages = [
            index for index, (start, _) in enumerate(ranges) if start in chunk_starts
        ]
        seed_budget = max(
            len(title_pages),
            math.ceil(budget * seed_budget_ratio),
        )
        seed_budget = min(seed_budget, budget)
        ranked_titles = sorted(
            title_pages,
            key=lambda index: (lexical_scores[index], -index),
            reverse=True,
        )
        selected = ranked_titles[:seed_budget]
        if len(selected) < seed_budget:
            selected_set = set(selected)
            ranked_pages = sorted(
                (
                    index
                    for index in range(len(ranges))
                    if index not in selected_set
                ),
                key=lambda index: (lexical_scores[index], -index),
                reverse=True,
            )
            selected.extend(ranked_pages[: seed_budget - len(selected)])
        return tuple(sorted(selected))


@torch.inference_mode()
def generate_from_chunk_controller(
    model,
    controller,
    *,
    suffix_ids: Sequence[int],
    virtual_prefix_tokens: int,
    max_new_tokens: int,
    eos_token_id: int | Sequence[int] | None,
) -> list[int]:
    """Consume question tokens and generate without prefilling documents."""
    suffix_ids = tuple(int(token) for token in suffix_ids)
    if not suffix_ids:
        raise ValueError("suffix_ids must not be empty")
    if virtual_prefix_tokens < 1 or max_new_tokens < 1:
        raise ValueError("prefix and generation lengths must be positive")

    controller.attach(model)
    if eos_token_id is None:
        stop_token_ids: set[int] = set()
    elif isinstance(eos_token_id, int):
        stop_token_ids = {eos_token_id}
    else:
        stop_token_ids = {int(token) for token in eos_token_id}
    position = virtual_prefix_tokens
    logits = None
    try:
        for token in suffix_ids:
            logits = _forward_one(model, token=token, position=position)
            controller.next_step()
            position += 1

        generated: list[int] = []
        assert logits is not None
        for index in range(max_new_tokens):
            next_token = int(logits.argmax(dim=-1).item())
            generated.append(next_token)
            if next_token in stop_token_ids:
                break
            if index + 1 < max_new_tokens:
                logits = _forward_one(
                    model,
                    token=next_token,
                    position=position,
                )
                controller.next_step()
                position += 1
        return generated
    finally:
        controller.detach(model)


@torch.inference_mode()
def precompute_chunk_bundle(
    model,
    *,
    system_ids: Sequence[int],
    chunks: Sequence[tuple[str, Sequence[int]]],
    mode: ProducerMode,
    pin_memory: bool = False,
) -> ChunkKVBundle:
    """Precompute system-conditioned document KV on the producer.

    Sequential mode is order-specific and slices one ordinary causal prefill.
    Independent mode evaluates ``system + chunk`` separately for every chunk,
    making chunks reorderable but removing cross-document conditioning.

    Args:
        model: A Transformers causal language model.
        system_ids: Tokenized system prefix, including any BOS token.
        chunks: Pairs of stable chunk names and token IDs.
        mode: ``sequential`` or ``independent``.
        pin_memory: Whether CPU cache tensors should use pinned memory.

    Returns:
        A CPU-resident chunk bundle.
    """
    if mode not in ("sequential", "independent"):
        raise ValueError("mode must be 'sequential' or 'independent'")
    system_ids = tuple(int(token) for token in system_ids)
    normalized_chunks = tuple(
        (str(name), tuple(int(token) for token in token_ids))
        for name, token_ids in chunks
    )
    if not system_ids:
        raise ValueError("system_ids must not be empty")
    if not normalized_chunks or any(not token_ids for _, token_ids in normalized_chunks):
        raise ValueError("chunks must contain at least one non-empty chunk")

    if mode == "sequential":
        all_ids = system_ids + tuple(
            token for _, chunk_ids in normalized_chunks for token in chunk_ids
        )
        full_cache = _prefill(model, all_ids)
        system_layers = _slice_to_cpu(
            full_cache,
            start=0,
            end=len(system_ids),
            pin_memory=pin_memory,
        )
        document_chunks = []
        cursor = len(system_ids)
        for name, token_ids in normalized_chunks:
            end = cursor + len(token_ids)
            document_chunks.append(
                DocumentChunkKV(
                    name=name,
                    token_ids=token_ids,
                    source_start=cursor,
                    layers=_slice_to_cpu(
                        full_cache,
                        start=cursor,
                        end=end,
                        pin_memory=pin_memory,
                    ),
                )
            )
            cursor = end
        forward_tokens = len(all_ids)
    else:
        system_cache = _prefill(model, system_ids)
        system_layers = _slice_to_cpu(
            system_cache,
            start=0,
            end=len(system_ids),
            pin_memory=pin_memory,
        )
        del system_cache
        document_chunks = []
        forward_tokens = len(system_ids)
        for name, token_ids in normalized_chunks:
            chunk_cache = _prefill(model, system_ids + token_ids)
            document_chunks.append(
                DocumentChunkKV(
                    name=name,
                    token_ids=token_ids,
                    source_start=len(system_ids),
                    layers=_slice_to_cpu(
                        chunk_cache,
                        start=len(system_ids),
                        end=len(system_ids) + len(token_ids),
                        pin_memory=pin_memory,
                    ),
                )
            )
            forward_tokens += len(system_ids) + len(token_ids)
            del chunk_cache

    return ChunkKVBundle(
        mode=mode,
        system_ids=system_ids,
        system_layers=system_layers,
        chunks=tuple(document_chunks),
        forward_tokens=forward_tokens,
    )


@torch.inference_mode()
def compose_chunk_bundle(
    model,
    bundle: ChunkKVBundle,
    *,
    chunk_order: Sequence[int] | None = None,
) -> ComposedChunkKV:
    """Compose stored chunks in a target order and relocate their RoPE keys."""
    if chunk_order is None:
        chunk_order = tuple(range(len(bundle.chunks)))
    else:
        chunk_order = tuple(int(index) for index in chunk_order)
    if sorted(chunk_order) != list(range(len(bundle.chunks))):
        raise ValueError("chunk_order must be a permutation of all chunk indices")

    ordered = tuple(bundle.chunks[index] for index in chunk_order)
    per_chunk_layers: list[LegacyCache] = []
    chunk_ranges: list[tuple[str, int, int]] = []
    cursor = len(bundle.system_ids)
    relocated_tokens = 0
    for chunk in ordered:
        target_start = cursor
        target_end = target_start + chunk.tokens
        if target_start == chunk.source_start:
            positioned_layers = chunk.layers
        else:
            positioned_layers = _relocate_cache(
                model,
                chunk.layers,
                source_start=chunk.source_start,
                target_start=target_start,
            )
            relocated_tokens += chunk.tokens
        per_chunk_layers.append(positioned_layers)
        chunk_ranges.append((chunk.name, target_start, target_end))
        cursor = target_end

    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, (system_key, system_value) in enumerate(bundle.system_layers):
        keys = [system_key]
        values = [system_value]
        for chunk_layers in per_chunk_layers:
            chunk_key, chunk_value = chunk_layers[layer_index]
            keys.append(chunk_key)
            values.append(chunk_value)
        layers.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
    return ComposedChunkKV(
        layers=tuple(layers),
        system_tokens=len(bundle.system_ids),
        document_tokens=sum(chunk.tokens for chunk in ordered),
        chunk_ranges=tuple(chunk_ranges),
        relocated_tokens=relocated_tokens,
    )


def _prefill(model, token_ids: Sequence[int]) -> LegacyCache:
    device = _model_device(model)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        return_dict=True,
    )
    return _legacy_cache(output.past_key_values)


def _forward_one(model, *, token: int, position: int) -> torch.Tensor:
    device = _model_device(model)
    output = model(
        input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
        attention_mask=torch.ones(
            (1, position + 1), dtype=torch.long, device=device
        ),
        position_ids=torch.tensor([[position]], dtype=torch.long, device=device),
        cache_position=torch.tensor([position], dtype=torch.long, device=device),
        use_cache=False,
        return_dict=True,
    )
    return output.logits[:, -1]


def _slice_to_cpu(
    cache: Any,
    *,
    start: int,
    end: int,
    pin_memory: bool,
) -> LegacyCache:
    result = []
    for key, value in _legacy_cache(cache):
        result.append(
            (
                _to_cpu(key[:, :, start:end], pin_memory=pin_memory),
                _to_cpu(value[:, :, start:end], pin_memory=pin_memory),
            )
        )
    return tuple(result)


def _relocate_cache(
    model,
    layers: LegacyCache,
    *,
    source_start: int,
    target_start: int,
) -> LegacyCache:
    tokens = layers[0][0].shape[2]
    source_positions = torch.arange(source_start, source_start + tokens)
    target_positions = torch.arange(target_start, target_start + tokens)
    source_cos, source_sin = model_rope_cos_sin(
        model,
        source_positions,
        device="cpu",
        dtype=torch.float32,
    )
    target_cos, target_sin = model_rope_cos_sin(
        model,
        target_positions,
        device="cpu",
        dtype=torch.float32,
    )
    relocated = []
    for key, value in layers:
        content_key = remove_rope(
            key.float(),
            source_cos,
            source_sin,
            sequence_dim=2,
        )
        target_key = apply_rope(
            content_key,
            target_cos,
            target_sin,
            sequence_dim=2,
        ).to(key.dtype)
        relocated.append((target_key, value))
    return tuple(relocated)


def _legacy_cache(cache: Any) -> LegacyCache:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise TypeError(f"unsupported cache type: {type(cache)!r}")


def _to_cpu(tensor: torch.Tensor, *, pin_memory: bool) -> torch.Tensor:
    result = tensor.detach().to("cpu").contiguous()
    if pin_memory and torch.cuda.is_available():
        result = result.pin_memory()
    return result


def _cache_bytes(cache: LegacyCache) -> int:
    return sum(
        (key.numel() + value.numel()) * key.element_size()
        for key, value in cache
    )


def _model_device(model) -> torch.device:
    backbone = getattr(model, "model", model)
    return backbone.embed_tokens.weight.device
