"""RaReCache rank-disagreement statistic on centered mapped calibration states."""
from __future__ import annotations

import math

import torch


def fit_basis(weight: torch.Tensor, gram: torch.Tensor, rank: int) -> torch.Tensor:
    if not 1 <= rank < weight.shape[1]:
        raise ValueError("Rank must be positive and smaller than the flattened target KV width")
    covariance = weight.T @ (gram @ weight)
    _, vectors = torch.linalg.eigh((covariance + covariance.T) * 0.5)
    return vectors[:, -rank:].contiguous()


def disagreement(mapped: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    centered = mapped - mean
    return (centered.square().sum(dim=-1) -
            (centered @ basis).square().sum(dim=-1)).clamp_min_(0)


def select_positions(scores: torch.Tensor, budget: float, *,
                     first_tokens: int = 0, last_tokens: int = 0) -> torch.Tensor:
    if scores.ndim != 1 or not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("Expected one finite recompute fraction in [0, 1]")
    if first_tokens < 0 or last_tokens < 0:
        raise ValueError("Reserved token counts must be nonnegative")
    count = math.ceil(len(scores) * budget)
    if not (first_tokens or last_tokens):
        return scores.argsort(descending=True, stable=True)[:count].sort().values
    # For short prefixes keep the start first and cap both ends to the budget.
    first = min(first_tokens, count)
    last = min(last_tokens, count - first)
    end = len(scores) - last
    interior = scores[first:end]
    ranked = interior.argsort(descending=True, stable=True)[:count - first - last]
    return torch.cat((
        torch.arange(first, device=scores.device),
        ranked.sort().values + first,
        torch.arange(end, len(scores), device=scores.device),
    ))
