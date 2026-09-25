from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from transformers import DynamicCache

from .semantic_kv_summary import concatenate_caches
from .soft_tail_capsule import LayerwiseKVWriteAdapter, reposition_tail_cache


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


class ReceiverLens(nn.Module):
    """Materialize task-local readout slots after an immutable receiver cache.

    The prefix is expected to contain the ordinary receiver prompt followed by
    a transported semantic memory.  Prefix tensors are detached before the
    lens pass, making the semantic packet an exact gradient boundary.  The
    returned cache contains only the newly materialized lens positions.
    """

    def __init__(
        self,
        initial_embeddings: torch.Tensor,
        *,
        write_adapter: LayerwiseKVWriteAdapter | None = None,
    ) -> None:
        super().__init__()
        if initial_embeddings.ndim != 2 or min(initial_embeddings.shape) < 1:
            raise ValueError("initial_embeddings must have shape [slots, hidden]")
        self.embeddings = nn.Parameter(initial_embeddings.detach().float().clone())
        self.write_adapter = write_adapter

    @property
    def slots(self) -> int:
        return int(self.embeddings.shape[0])

    def materialize(
        self,
        model,
        *,
        prefix_cache: Any,
        prefix_tokens: int,
    ) -> LegacyCache:
        layers = validate_cache(prefix_cache, expected_tokens=prefix_tokens)
        if len(layers) != model.config.num_hidden_layers:
            raise ValueError("prefix cache depth disagrees with receiver model")
        if self.embeddings.shape[1] != model.config.hidden_size:
            raise ValueError("lens hidden size disagrees with receiver model")
        detached_layers = tuple(
            (key.detach(), value.detach()) for key, value in layers
        )
        device = model.model.embed_tokens.weight.device
        dtype = model.model.embed_tokens.weight.dtype
        positions = torch.arange(
            prefix_tokens,
            prefix_tokens + self.slots,
            dtype=torch.long,
            device=device,
        )
        cache = DynamicCache(ddp_cache_data=detached_layers, config=model.config)
        active_writer = (
            nullcontext()
            if self.write_adapter is None
            else self.write_adapter.activate(model)
        )
        with active_writer:
            output = model(
                inputs_embeds=self.embeddings.to(
                    device=device, dtype=dtype
                ).unsqueeze(0),
                attention_mask=torch.ones(
                    (1, prefix_tokens + self.slots),
                    dtype=torch.long,
                    device=device,
                ),
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        full = _legacy_cache(output.past_key_values)
        return tuple(
            (key[:, :, -self.slots :], value[:, :, -self.slots :])
            for key, value in full
        )


def cache_with_lens(prefix_cache: Any, lens_cache: Any) -> LegacyCache:
    """Append a materialized lens without changing either input cache."""

    return concatenate_caches(prefix_cache, lens_cache)


@torch.no_grad()
def extend_cache_with_tokens(
    model,
    *,
    prefix_cache: Any,
    prefix_tokens: int,
    token_ids: Sequence[int],
) -> LegacyCache:
    """Append ordinary hard tokens and return the complete detached cache."""

    layers = validate_cache(prefix_cache, expected_tokens=prefix_tokens)
    ids = tuple(int(token_id) for token_id in token_ids)
    if not ids:
        raise ValueError("token_ids must be non-empty")
    if len(layers) != model.config.num_hidden_layers:
        raise ValueError("prefix cache depth disagrees with receiver model")
    device = model.model.embed_tokens.weight.device
    positions = torch.arange(
        prefix_tokens,
        prefix_tokens + len(ids),
        dtype=torch.long,
        device=device,
    )
    cache = DynamicCache(
        ddp_cache_data=tuple(
            (key.detach(), value.detach()) for key, value in layers
        ),
        config=model.config,
    )
    output = model(
        input_ids=torch.tensor([ids], dtype=torch.long, device=device),
        attention_mask=torch.ones(
            (1, prefix_tokens + len(ids)), dtype=torch.long, device=device
        ),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    return tuple(
        (key.detach(), value.detach())
        for key, value in _legacy_cache(output.past_key_values)
    )


def cache_after_memory_eviction(
    model,
    *,
    receiver_prompt_cache: Any,
    receiver_prompt_tokens: int,
    lens_cache: Any,
    lens_source_start: int,
) -> LegacyCache:
    """Relocate a lens after removing the semantic-memory positions before it.

    This helper only establishes valid cache and RoPE coordinates.  It does not
    claim that the lens has consolidated enough information for equivalent
    continuation behavior; that requires a separate task gate.
    """

    prompt = validate_cache(
        receiver_prompt_cache, expected_tokens=receiver_prompt_tokens
    )
    lens = validate_cache(lens_cache)
    if len(prompt) != len(lens):
        raise ValueError("receiver prompt and lens caches have different layer counts")
    if lens_source_start < receiver_prompt_tokens:
        raise ValueError("lens source start must not precede the retained prompt")
    relocated = reposition_tail_cache(
        model,
        lens,
        source_start=lens_source_start,
        target_start=receiver_prompt_tokens,
    )
    return concatenate_caches(prompt, relocated)


def validate_cache(
    cache: Any, *, expected_tokens: int | None = None
) -> LegacyCache:
    layers = _legacy_cache(cache)
    if not layers:
        raise ValueError("cache must contain at least one layer")
    tokens = None
    batch = None
    for key, value in layers:
        if key.ndim != 4 or value.shape != key.shape:
            raise ValueError("cache layers must contain matching rank-four K/V")
        if key.shape[-2] < 1:
            raise ValueError("cache token length must be positive")
        if tokens is None:
            tokens = int(key.shape[-2])
            batch = int(key.shape[0])
        elif key.shape[-2] != tokens or key.shape[0] != batch:
            raise ValueError("cache layers must share batch and token dimensions")
    if batch != 1:
        raise ValueError("receiver lens currently requires batch size one")
    if expected_tokens is not None:
        if expected_tokens < 1 or tokens != expected_tokens:
            raise ValueError("cache token length disagrees with prefix_tokens")
    return layers


def _legacy_cache(cache: Any) -> LegacyCache:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise TypeError(f"unsupported cache type: {type(cache)!r}")
