"""Qwen3 cache boundaries shared by both mapper experiments."""
from __future__ import annotations

import torch
from transformers import DynamicCache
from xmodel_kv.rope import apply_rope, model_rope_cos_sin, remove_rope


def legacy_cache(cache: DynamicCache) -> tuple:
    return cache.to_legacy_cache()


def prefill(model, input_ids: torch.Tensor) -> DynamicCache:
    """Prefill the decoder body while skipping the vocabulary projection."""
    return model.model(input_ids=input_ids, use_cache=True,
                       return_dict=True).past_key_values


def unrotate_rows(model, cache: DynamicCache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Convert cache layers from [1,H,T,D] to [T,H,D] in content coordinates."""
    blocks = legacy_cache(cache)
    first = blocks[0][0]
    if first.ndim != 4 or first.shape[0] != 1:
        raise ValueError("Expected unpadded cache [1,heads,tokens,width]")
    pos = torch.arange(first.shape[2], device=first.device)
    cos, sin = model_rope_cos_sin(model, pos, device=first.device, dtype=torch.float32)
    return [(remove_rope(k.float(), cos, sin)[0].transpose(0, 1),
             v[0].transpose(0, 1)) for k, v in blocks]


def prepare_key(key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                dtype: torch.dtype) -> torch.Tensor:
    """Apply target RoPE and return the Hugging Face cache layout."""
    return apply_rope(key.transpose(0, 1).unsqueeze(0), cos, sin).to(dtype).contiguous()


def prepare_value(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return value.transpose(0, 1).unsqueeze(0).to(dtype).contiguous()
