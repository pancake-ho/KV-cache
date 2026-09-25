from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import DynamicCache

from .rope import apply_rope, model_rope_cos_sin


@torch.inference_mode()
def capture_teacher_forced_query_probes(
    model,
    *,
    legacy_cache: Any,
    cached_tokens: int,
    query_ids: Sequence[int],
    answer_ids: Sequence[int],
    position_shift: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Capture RoPE-applied receiver queries at answer-prediction positions.

    Query content is obtained from a common no-summary receiver trajectory.
    ``position_shift`` then places those probes after a hypothetical compact
    memory without making the probe content depend on the candidate memory.
    This produces a fixed, task-relevant query distribution for comparing two
    K/V operators.
    """

    if min(cached_tokens, len(query_ids), len(answer_ids)) < 1 or position_shift < 0:
        raise ValueError("cache, query, answer, and position shift are invalid")
    layers = model.model.layers
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles = []
    for layer_index, layer in enumerate(layers):
        def hook(_module, _inputs, output, *, index=layer_index):
            captured[index] = output.detach()

        handles.append(layer.self_attn.q_proj.register_forward_hook(hook))
    input_ids = tuple(int(token) for token in query_ids) + tuple(
        int(token) for token in answer_ids[:-1]
    )
    device = model.model.embed_tokens.weight.device
    current = torch.tensor([input_ids], dtype=torch.long, device=device)
    positions = torch.arange(
        cached_tokens,
        cached_tokens + current.shape[1],
        dtype=torch.long,
        device=device,
    )
    cache = DynamicCache(ddp_cache_data=_legacy_cache(legacy_cache), config=model.config)
    try:
        model(
            input_ids=current,
            attention_mask=torch.ones(
                (1, cached_tokens + current.shape[1]),
                dtype=torch.long,
                device=device,
            ),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("failed to capture a query projection from every layer")

    start = len(query_ids) - 1
    selected_positions = positions[start : start + len(answer_ids)] + position_shift
    cos, sin = model_rope_cos_sin(
        model, selected_positions, device=device, dtype=torch.float32
    )
    probes = []
    for projection in captured:
        projection = projection[:, start : start + len(answer_ids)]
        projection = projection.view(
            1,
            len(answer_ids),
            model.config.num_attention_heads,
            model.config.head_dim,
        ).transpose(1, 2)
        probes.append(
            apply_rope(projection.float(), cos, sin, sequence_dim=2).to(
                dtype=projection.dtype
            )
        )
    return tuple(probes)


def _legacy_cache(cache: Any):
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise TypeError(f"unsupported cache type: {type(cache)!r}")
