from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from transformers import DynamicCache

from .rope import apply_rope, model_rope_cos_sin, remove_rope


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


class TrainableKVSummary(nn.Module):
    """A per-instance oracle summary represented directly as receiver-native KV.

    Parameters are kept in fp32 for optimization and cast to the frozen model's
    cache dtype when materialized.  There is intentionally no token embedding or
    Transformer prefill between these parameters and the receiver attention.
    """

    def __init__(self, initial_cache: Sequence[tuple[torch.Tensor, torch.Tensor]]):
        super().__init__()
        if not initial_cache:
            raise ValueError("initial_cache must contain at least one layer")
        shapes = []
        self.keys = nn.ParameterList()
        self.values = nn.ParameterList()
        for key, value in initial_cache:
            if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
                raise ValueError("each KV layer must contain matching rank-four tensors")
            if key.shape[0] != 1 or key.shape[-2] < 1:
                raise ValueError("oracle summaries currently require batch one and positive length")
            shapes.append(tuple(key.shape))
            self.keys.append(nn.Parameter(key.detach().float().clone()))
            self.values.append(nn.Parameter(value.detach().float().clone()))
        if len({shape[-2] for shape in shapes}) != 1:
            raise ValueError("all summary layers must use the same token budget")

    @property
    def tokens(self) -> int:
        return int(self.keys[0].shape[-2])

    def cache(self, *, dtype: torch.dtype, device: torch.device | str) -> LegacyCache:
        return tuple(
            (
                key.to(device=device, dtype=dtype),
                value.to(device=device, dtype=dtype),
            )
            for key, value in zip(self.keys, self.values, strict=True)
        )

    def detached_cache(
        self, *, dtype: torch.dtype, device: torch.device | str
    ) -> LegacyCache:
        return tuple(
            (
                key.detach().to(device=device, dtype=dtype),
                value.detach().to(device=device, dtype=dtype),
            )
            for key, value in zip(self.keys, self.values, strict=True)
        )


@torch.no_grad()
def mean_pool_segment_cache(
    model,
    legacy_cache: Any,
    *,
    segment_start: int,
    segment_end: int,
    target_start: int,
    tokens: int,
) -> LegacyCache:
    """Pool a contextual KV segment into fewer position-correct initial slots.

    Keys are first moved out of RoPE space, pooled in content space, and then
    rotated into the compact receiver positions. Values are pooled directly.
    The result is only an initialization; oracle optimization is expected to
    synthesize new K/V rather than merely retain these averages.
    """

    layers = _legacy_cache(legacy_cache)
    if not 0 <= segment_start < segment_end:
        raise ValueError("segment bounds must be positive and non-empty")
    segment_tokens = segment_end - segment_start
    if not 1 <= tokens <= segment_tokens:
        raise ValueError("tokens must be in [1, segment length]")
    if target_start < 0:
        raise ValueError("target_start must be non-negative")

    first_key = layers[0][0]
    source_positions = torch.arange(
        segment_start, segment_end, device=first_key.device, dtype=torch.long
    )
    target_positions = torch.arange(
        target_start, target_start + tokens, device=first_key.device, dtype=torch.long
    )
    source_cos, source_sin = model_rope_cos_sin(
        model, source_positions, device=first_key.device, dtype=torch.float32
    )
    target_cos, target_sin = model_rope_cos_sin(
        model, target_positions, device=first_key.device, dtype=torch.float32
    )

    compacted = []
    for key, value in layers:
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("legacy cache layers must contain matching rank-four KV")
        if segment_end > key.shape[-2]:
            raise ValueError("segment exceeds cache length")
        content_key = remove_rope(
            key[:, :, segment_start:segment_end].float(),
            source_cos,
            source_sin,
            sequence_dim=2,
        )
        pooled_key = even_bin_mean(content_key, tokens, sequence_dim=2)
        pooled_value = even_bin_mean(
            value[:, :, segment_start:segment_end].float(), tokens, sequence_dim=2
        )
        positioned_key = apply_rope(
            pooled_key, target_cos, target_sin, sequence_dim=2
        )
        compacted.append(
            (
                positioned_key.to(dtype=key.dtype),
                pooled_value.to(dtype=value.dtype),
            )
        )
    return tuple(compacted)


def even_bin_mean(
    tensor: torch.Tensor, bins: int, *, sequence_dim: int = -2
) -> torch.Tensor:
    if tensor.ndim < 1:
        raise ValueError("tensor must have at least one dimension")
    sequence_dim %= tensor.ndim
    length = tensor.shape[sequence_dim]
    if not 1 <= bins <= length:
        raise ValueError("bins must be in [1, sequence length]")
    boundaries = torch.linspace(0, length, bins + 1, device=tensor.device).round().long()
    pooled = []
    for index in range(bins):
        start = min(int(boundaries[index]), length - 1)
        end = max(start + 1, int(boundaries[index + 1]))
        slices = [slice(None)] * tensor.ndim
        slices[sequence_dim] = slice(start, end)
        pooled.append(tensor[tuple(slices)].mean(dim=sequence_dim))
    return torch.stack(pooled, dim=sequence_dim)


def concatenate_caches(prefix: Any, suffix: Any) -> LegacyCache:
    prefix_layers = _legacy_cache(prefix)
    suffix_layers = _legacy_cache(suffix)
    if len(prefix_layers) != len(suffix_layers):
        raise ValueError("prefix and suffix caches have different layer counts")
    result = []
    for (prefix_key, prefix_value), (suffix_key, suffix_value) in zip(
        prefix_layers, suffix_layers, strict=True
    ):
        if prefix_key.shape[:-2] + prefix_key.shape[-1:] != suffix_key.shape[:-2] + suffix_key.shape[-1:]:
            raise ValueError("prefix and suffix key shapes are incompatible")
        if prefix_value.shape[:-2] + prefix_value.shape[-1:] != suffix_value.shape[:-2] + suffix_value.shape[-1:]:
            raise ValueError("prefix and suffix value shapes are incompatible")
        result.append(
            (
                torch.cat((prefix_key, suffix_key), dim=-2),
                torch.cat((prefix_value, suffix_value), dim=-2),
            )
        )
    return tuple(result)


def teacher_forced_answer_logits(
    model,
    *,
    legacy_cache: Any,
    cached_tokens: int,
    query_ids: Sequence[int],
    answer_ids: Sequence[int],
) -> torch.Tensor:
    """Return one next-token logit vector for every answer token."""

    if not query_ids or not answer_ids:
        raise ValueError("query_ids and answer_ids must be non-empty")
    device = _model_device(model)
    input_ids = tuple(int(token) for token in query_ids) + tuple(
        int(token) for token in answer_ids[:-1]
    )
    current = torch.tensor([input_ids], dtype=torch.long, device=device)
    final_length = cached_tokens + current.shape[1]
    positions = torch.arange(cached_tokens, final_length, device=device)
    cache = DynamicCache(ddp_cache_data=_legacy_cache(legacy_cache), config=model.config)
    output = model(
        input_ids=current,
        attention_mask=torch.ones((1, final_length), dtype=torch.long, device=device),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=cache,
        use_cache=False,
        return_dict=True,
    )
    start = len(query_ids) - 1
    return output.logits[:, start : start + len(answer_ids)]


def _legacy_cache(cache: Any) -> LegacyCache:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise TypeError(f"unsupported cache type: {type(cache)!r}")


def _model_device(model) -> torch.device:
    backbone = getattr(model, "model", model)
    return backbone.embed_tokens.weight.device
