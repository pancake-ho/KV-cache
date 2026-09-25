from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """The rotate-half convention used by Qwen3/Llama in Transformers."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sequence_dim: int = -2,
) -> torch.Tensor:
    cos, sin = _broadcast_rope(x, cos, sin, sequence_dim)
    return x * cos + rotate_half(x) * sin


def remove_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sequence_dim: int = -2,
) -> torch.Tensor:
    """Apply the exact inverse of an orthogonal RoPE rotation."""
    cos, sin = _broadcast_rope(x, cos, sin, sequence_dim)
    return x * cos - rotate_half(x) * sin


def _broadcast_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sequence_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_dim %= x.ndim
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shape mismatch: {cos.shape} vs {sin.shape}")
    if cos.ndim == 1:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.ndim not in (2, 3):
        raise ValueError(
            "cos and sin must have shape [tokens, head_dim] or "
            "[batch, tokens, head_dim]"
        )
    if cos.shape[-2] != x.shape[sequence_dim] or cos.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"RoPE shape {tuple(cos.shape)} is incompatible with tensor {tuple(x.shape)} "
            f"at sequence_dim={sequence_dim}"
        )
    if cos.ndim == 3:
        if x.ndim < 3 or cos.shape[0] != x.shape[0]:
            raise ValueError(
                f"batched RoPE shape {tuple(cos.shape)} is incompatible with "
                f"tensor {tuple(x.shape)}"
            )
        shape = [1] * x.ndim
        shape[0] = cos.shape[0]
        shape[sequence_dim] = cos.shape[1]
        shape[-1] = cos.shape[2]
        return cos.reshape(shape), sin.reshape(shape)
    shape = [1] * x.ndim
    shape[sequence_dim] = cos.shape[0]
    shape[-1] = cos.shape[1]
    return cos.reshape(shape), sin.reshape(shape)


def model_rope_cos_sin(
    model,
    positions: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ask the HF model's own rotary module for its exact RoPE schedule."""
    backbone = getattr(model, "model", model)
    rotary = backbone.rotary_emb
    positions = positions.to(device=device, dtype=torch.long)
    if positions.ndim == 1:
        position_ids = positions.unsqueeze(0)
    elif positions.ndim == 2:
        position_ids = positions
    else:
        raise ValueError("positions must have shape [tokens] or [batch, tokens]")
    # Rotary implementations only inspect shape/device/dtype of x.
    hidden_size = int(backbone.config.hidden_size)
    dummy = torch.empty((*position_ids.shape, hidden_size), device=device, dtype=dtype)
    cos, sin = rotary(dummy, position_ids)
    cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
    if positions.ndim == 1:
        return cos[0], sin[0]
    return cos, sin
