"""Selective Qwen3 replay over mapped KV with original absolute positions."""
from __future__ import annotations

import torch
from transformers import DynamicCache


class _ReplayCache(DynamicCache):
    """Overwrite selected existing entries before each layer attends to the prefix."""

    def __init__(self, mapped: DynamicCache, positions: torch.Tensor):
        super().__init__()
        self.layers = mapped.layers
        self.positions = positions

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_layer = self.layers[layer_idx]
        key, value = cache_layer.keys, cache_layer.values
        key.index_copy_(2, self.positions, key_states)
        value.index_copy_(2, self.positions, value_states)
        return key, value


@torch.inference_mode()
def recompute(target, mapped: DynamicCache, prefix: torch.Tensor,
              positions: torch.Tensor) -> DynamicCache:
    """Replay selected tokens in one call, preserving normal append behavior afterward."""
    length = prefix.shape[1]
    if prefix.shape[0] != 1 or mapped.get_seq_length() != length:
        raise ValueError("Prefix and mapped cache must have matching, unpadded token lengths")
    if not positions.numel():
        return mapped
    if positions.ndim != 1 or positions.dtype != torch.long or (
            positions[0] < 0 or positions[-1] >= length
            or (positions[1:] <= positions[:-1]).any()):
        raise ValueError("Replay positions must be sorted, unique, and in bounds")
    device = prefix.device
    dtype = target.model.embed_tokens.weight.dtype
    blocked = torch.arange(length, device=device)[None, :] > positions[:, None]
    mask = torch.zeros((1, 1, len(positions), length), device=device, dtype=dtype)
    mask.masked_fill_(blocked, torch.finfo(dtype).min)
    target.model(
        input_ids=prefix.index_select(1, positions),
        past_key_values=_ReplayCache(mapped, positions),
        position_ids=positions.unsqueeze(0), cache_position=positions,
        attention_mask={"full_attention": mask}, use_cache=True, return_dict=True,
    )
    return mapped
