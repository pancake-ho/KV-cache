from __future__ import annotations

from typing import Any

import torch

from .differentiable_delta import fake_int4_cache_round_trip
from .soft_tail_capsule import reposition_tail_cache


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def cache_prefix_relative_mse(
    student_cache: Any,
    teacher_cache: Any,
    *,
    active_layers: frozenset[int],
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Mean teacher-normalized MSE over an aligned cache prefix.

    The teacher may contain fewer positions than the student.  Its complete
    token axis is aligned with the first teacher-length student positions.
    Every selected K or V tensor contributes equally after averaging over
    batch, heads, positions, and channels.  Normalizing each vector by the
    teacher mean-square prevents high-scale layers from silently owning the
    objective while retaining both direction and magnitude information.
    """

    student = tuple(student_cache)
    teacher = tuple(teacher_cache)
    if not student or len(student) != len(teacher):
        raise ValueError("student and teacher caches must have equal non-zero depth")
    if (
        not active_layers
        or min(active_layers) < 0
        or max(active_layers) >= len(student)
    ):
        raise ValueError("active_layers must be a non-empty in-range subset")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    teacher_tokens = int(teacher[0][0].shape[-2])
    student_tokens = int(student[0][0].shape[-2])
    if teacher_tokens < 1 or student_tokens < teacher_tokens:
        raise ValueError("student must contain the non-empty teacher token prefix")

    terms = []
    for layer_index in sorted(active_layers):
        student_layer = student[layer_index]
        teacher_layer = teacher[layer_index]
        if len(student_layer) != 2 or len(teacher_layer) != 2:
            raise ValueError("every cache layer must contain one key and one value")
        for student_tensor, teacher_tensor in zip(
            student_layer, teacher_layer, strict=True
        ):
            if (
                student_tensor.ndim != 4
                or teacher_tensor.ndim != 4
                or student_tensor.shape[0] != teacher_tensor.shape[0]
                or student_tensor.shape[1] != teacher_tensor.shape[1]
                or student_tensor.shape[3] != teacher_tensor.shape[3]
                or student_tensor.shape[-2] < teacher_tokens
                or teacher_tensor.shape[-2] != teacher_tokens
            ):
                raise ValueError("student and teacher cache tensor shapes are incompatible")
            candidate = student_tensor[..., :teacher_tokens, :].float()
            target = teacher_tensor.detach().float()
            squared_error = (candidate - target).square().mean(dim=-1)
            teacher_power = target.square().mean(dim=-1).clamp_min(epsilon)
            terms.append((squared_error / teacher_power).mean())
    return torch.stack(terms).mean()


def canonical_int4_tail_state_loss(
    model,
    student_tail: Any,
    teacher_tail: Any,
    *,
    source_start: int,
    active_layers: frozenset[int],
    straight_through: bool,
) -> torch.Tensor:
    """Compare production-rate K4/V4 states in a receiver-independent frame."""

    student = tuple(student_tail)
    teacher = tuple(teacher_tail)
    if not student or not teacher:
        raise ValueError("student and teacher tails must be non-empty")
    student_tokens = int(student[0][0].shape[-2])
    teacher_tokens = int(teacher[0][0].shape[-2])
    canonical_student = reposition_tail_cache(
        model, student, source_start=source_start, target_start=0
    )
    canonical_teacher = reposition_tail_cache(
        model, teacher, source_start=source_start, target_start=0
    )
    quantized_student = fake_int4_cache_round_trip(
        canonical_student,
        suffix_tokens=student_tokens,
        active_layers=active_layers,
        straight_through=straight_through,
    )
    quantized_teacher = fake_int4_cache_round_trip(
        canonical_teacher,
        suffix_tokens=teacher_tokens,
        active_layers=active_layers,
        straight_through=False,
    )
    return cache_prefix_relative_mse(
        quantized_student,
        quantized_teacher,
        active_layers=active_layers,
    )
