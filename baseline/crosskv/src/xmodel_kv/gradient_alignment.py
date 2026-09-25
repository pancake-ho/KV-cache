from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GradientAlignment:
    dot: float
    cosine: float
    first_norm: float
    second_norm: float


def loss_gradient_tuple(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    """Return detached FP32 gradients with explicit zeros for unused parameters."""

    if loss.numel() != 1 or not loss.requires_grad:
        raise ValueError("loss must be a differentiable scalar")
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("parameters must be non-empty")
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(parameter, dtype=torch.float32)
        if gradient is None
        else gradient.detach().float()
        for parameter, gradient in zip(parameters, gradients, strict=True)
    )


def gradient_alignment(
    first: Sequence[torch.Tensor],
    second: Sequence[torch.Tensor],
    *,
    epsilon: float = 1e-30,
) -> GradientAlignment:
    """Compute a numerically stable global dot product and cosine."""

    first = tuple(first)
    second = tuple(second)
    if not first or len(first) != len(second):
        raise ValueError("gradient collections must have equal non-zero length")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    dot = torch.zeros((), dtype=torch.float64)
    first_square = torch.zeros((), dtype=torch.float64)
    second_square = torch.zeros((), dtype=torch.float64)
    for left, right in zip(first, second, strict=True):
        if left.shape != right.shape:
            raise ValueError("paired gradient tensors must have equal shape")
        left_cpu = left.detach().double().cpu()
        right_cpu = right.detach().double().cpu()
        dot += (left_cpu * right_cpu).sum()
        first_square += left_cpu.square().sum()
        second_square += right_cpu.square().sum()
    first_norm = torch.sqrt(first_square)
    second_norm = torch.sqrt(second_square)
    denominator = (first_norm * second_norm).clamp_min(epsilon)
    return GradientAlignment(
        dot=float(dot.item()),
        cosine=float((dot / denominator).item()),
        first_norm=float(first_norm.item()),
        second_norm=float(second_norm.item()),
    )


def project_away_conflicting_component(
    candidate: Sequence[torch.Tensor],
    protected: Sequence[torch.Tensor],
    *,
    epsilon: float = 1e-30,
) -> tuple[torch.Tensor, ...]:
    """PCGrad-style projection only when candidate opposes the protected loss."""

    candidate = tuple(candidate)
    protected = tuple(protected)
    alignment = gradient_alignment(candidate, protected, epsilon=epsilon)
    if alignment.dot >= 0:
        return tuple(tensor.clone() for tensor in candidate)
    protected_square = sum(
        tensor.detach().double().cpu().square().sum() for tensor in protected
    ).clamp_min(epsilon)
    coefficient = alignment.dot / float(protected_square.item())
    return tuple(
        left - coefficient * right
        for left, right in zip(candidate, protected, strict=True)
    )
