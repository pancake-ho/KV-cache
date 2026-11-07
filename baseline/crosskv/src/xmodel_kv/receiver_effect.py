from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import DynamicCache


@dataclass(frozen=True)
class ReceiverEffectLoss:
    """Layer-balanced distance between two counterfactual receiver effects."""

    total: torch.Tensor
    direction: torch.Tensor
    log_norm: torch.Tensor


def predictor_hidden_trajectory(
    model,
    *,
    legacy_cache: Any,
    cached_tokens: int,
    query_ids: Sequence[int],
    answer_ids: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    """Return block-output states at every teacher-forced answer predictor.

    Unlike a fixed-query operator probe, this executes the complete receiver
    trajectory under ``legacy_cache``.  Consequently every returned layer is
    on-policy with respect to the supplied memory.  The first selected state is
    the final query state that predicts the first answer token; subsequent
    states include the teacher-forced answer prefix.
    """

    if cached_tokens < 1 or not query_ids or not answer_ids:
        raise ValueError("cache, query, and answer must be non-empty")
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
    output = model.model(
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
        output_hidden_states=True,
        return_dict=True,
    )
    start = len(query_ids) - 1
    # hidden_states[0] is the common token embedding and precedes all memory
    # interaction.  Each remaining tensor is the output of one decoder block.
    return tuple(
        hidden[:, start : start + len(answer_ids)]
        for hidden in output.hidden_states[1:]
    )


def counterfactual_receiver_effect_loss(
    student_trajectory: Sequence[torch.Tensor],
    student_null_trajectory: Sequence[torch.Tensor],
    teacher_trajectory: Sequence[torch.Tensor],
    teacher_null_trajectory: Sequence[torch.Tensor],
    *,
    active_layers: frozenset[int] | None = None,
    log_norm_weight: float = 0.1,
    epsilon: float = 1e-8,
) -> ReceiverEffectLoss:
    """Match the marginal trajectory effect of compact and plaintext memory.

    Each memory is compared with a zero-K/V counterfactual of its own length,
    so common system/query computation and absolute query-position changes are
    removed before matching.  Direction and magnitude are treated separately
    to prevent a few high-norm layers from dominating the objective.
    """

    trajectories = tuple(
        tuple(value)
        for value in (
            student_trajectory,
            student_null_trajectory,
            teacher_trajectory,
            teacher_null_trajectory,
        )
    )
    layer_count = len(trajectories[0])
    if layer_count < 1 or any(len(value) != layer_count for value in trajectories):
        raise ValueError("all trajectories must have the same non-zero layer count")
    if log_norm_weight < 0 or epsilon <= 0:
        raise ValueError("loss weight must be non-negative and epsilon positive")
    layers = (
        tuple(range(layer_count))
        if active_layers is None
        else tuple(sorted(active_layers))
    )
    if not layers or layers[0] < 0 or layers[-1] >= layer_count:
        raise ValueError("active layers are empty or outside the trajectory")

    direction_terms = []
    log_norm_terms = []
    for layer_index in layers:
        student, student_null, teacher, teacher_null = (
            values[layer_index] for values in trajectories
        )
        if not (
            student.shape
            == student_null.shape
            == teacher.shape
            == teacher_null.shape
        ):
            raise ValueError("paired trajectory tensors must have identical shapes")
        student_effect = (student - student_null).float().reshape(-1)
        teacher_effect = (teacher - teacher_null).float().reshape(-1)
        student_norm = torch.linalg.vector_norm(student_effect).clamp_min(epsilon)
        teacher_norm = torch.linalg.vector_norm(teacher_effect).clamp_min(epsilon)
        cosine = torch.dot(student_effect, teacher_effect) / (
            student_norm * teacher_norm
        )
        direction_terms.append(1.0 - cosine)
        log_norm_terms.append(torch.log(student_norm / teacher_norm).square())

    direction = torch.stack(direction_terms).mean()
    log_norm = torch.stack(log_norm_terms).mean()
    return ReceiverEffectLoss(
        total=direction + log_norm_weight * log_norm,
        direction=direction,
        log_norm=log_norm,
    )


def zero_cache_like(cache: Any):
    """Construct a differentiable-safe zero-K/V counterfactual of equal shape."""

    return tuple(
        (torch.zeros_like(key), torch.zeros_like(value))
        for key, value in _legacy_cache(cache)
    )


def slice_cache_tokens(cache: Any, start: int, end: int | None = None):
    """Slice the sequence dimension of every layer in a legacy cache."""

    if start < 0 or (end is not None and end < start):
        raise ValueError("invalid cache slice")
    return tuple(
        (key[..., start:end, :], value[..., start:end, :])
        for key, value in _legacy_cache(cache)
    )


def _legacy_cache(cache: Any):
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise TypeError(f"unsupported cache type: {type(cache)!r}")
