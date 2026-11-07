from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class AttentionMemoryStatistics:
    """Sufficient tail-memory statistics for competition with a common prefix."""

    log_mass: torch.Tensor
    conditional_value: torch.Tensor


@dataclass(frozen=True)
class AttentionOperatorLoss:
    total: torch.Tensor
    value: torch.Tensor
    log_mass: torch.Tensor


def attention_memory_statistics(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> AttentionMemoryStatistics:
    """Evaluate the operator represented by one K/V memory.

    Queries have shape ``[batch, query_heads, probes, head_dim]`` and K/V have
    shape ``[batch, kv_heads, memory_positions, head_dim]``.  Grouped-query
    attention is handled by repeating each KV head over its query-head group.
    ``log_mass`` must be retained alongside the conditional value: two memories
    can return the same value after tail-only softmax while competing differently
    with receiver-prefix tokens in the full attention normalization.
    """

    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("queries, keys, and values must be rank-four tensors")
    if keys.shape != values.shape:
        raise ValueError("keys and values must have identical shapes")
    if (
        queries.shape[0] != keys.shape[0]
        or queries.shape[-1] != keys.shape[-1]
        or min(queries.shape[1], queries.shape[2], keys.shape[1], keys.shape[2]) < 1
    ):
        raise ValueError("query and memory dimensions are incompatible")
    if queries.shape[1] % keys.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    groups = queries.shape[1] // keys.shape[1]
    expanded_keys = keys.repeat_interleave(groups, dim=1).float()
    expanded_values = values.repeat_interleave(groups, dim=1).float()
    scores = torch.matmul(queries.float(), expanded_keys.transpose(-1, -2))
    scores = scores / math.sqrt(queries.shape[-1])
    log_mass = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    conditional_value = torch.matmul(probabilities, expanded_values)
    return AttentionMemoryStatistics(
        log_mass=log_mass, conditional_value=conditional_value
    )


def attention_operator_distillation_loss(
    queries: torch.Tensor,
    student_keys: torch.Tensor,
    student_values: torch.Tensor,
    teacher_keys: torch.Tensor,
    teacher_values: torch.Tensor,
    *,
    value_weight: float = 1.0,
    log_mass_weight: float = 0.1,
    epsilon: float = 1e-6,
) -> AttentionOperatorLoss:
    """Match a teacher memory without requiring token-wise correspondence."""

    if min(value_weight, log_mass_weight) < 0 or value_weight + log_mass_weight <= 0:
        raise ValueError("operator-loss weights must be non-negative and non-zero")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    student = attention_memory_statistics(queries, student_keys, student_values)
    teacher = attention_memory_statistics(
        queries, teacher_keys.detach(), teacher_values.detach()
    )
    value_error = (
        student.conditional_value - teacher.conditional_value.detach()
    ).square().mean(dim=-1)
    teacher_power = (
        teacher.conditional_value.detach().square().mean(dim=-1).clamp_min(epsilon)
    )
    value_loss = (value_error / teacher_power).mean()
    log_mass_loss = (
        student.log_mass - teacher.log_mass.detach()
    ).square().mean()
    total = value_weight * value_loss + log_mass_weight * log_mass_loss
    return AttentionOperatorLoss(
        total=total, value=value_loss, log_mass=log_mass_loss
    )


def cache_attention_operator_distillation_loss(
    student_cache: Any,
    teacher_cache: Any,
    queries_by_layer: Sequence[torch.Tensor],
    *,
    active_layers: frozenset[int],
    value_weight: float = 1.0,
    log_mass_weight: float = 0.1,
) -> AttentionOperatorLoss:
    """Layer-balanced attention-operator loss over two variable-length caches."""

    student = tuple(student_cache)
    teacher = tuple(teacher_cache)
    if not student or len(student) != len(teacher) or len(student) != len(
        queries_by_layer
    ):
        raise ValueError("cache and query collections must have equal non-zero depth")
    if (
        not active_layers
        or min(active_layers) < 0
        or max(active_layers) >= len(student)
    ):
        raise ValueError("active_layers must be a non-empty in-range subset")
    layer_losses = []
    for layer_index in sorted(active_layers):
        student_key, student_value = student[layer_index]
        teacher_key, teacher_value = teacher[layer_index]
        layer_losses.append(
            attention_operator_distillation_loss(
                queries_by_layer[layer_index],
                student_key,
                student_value,
                teacher_key,
                teacher_value,
                value_weight=value_weight,
                log_mass_weight=log_mass_weight,
            )
        )
    value = torch.stack([loss.value for loss in layer_losses]).mean()
    log_mass = torch.stack([loss.log_mass for loss in layer_losses]).mean()
    total = value_weight * value + log_mass_weight * log_mass
    return AttentionOperatorLoss(total=total, value=value, log_mass=log_mass)
