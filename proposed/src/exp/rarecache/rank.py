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


def select_positions(scores: torch.Tensor, budget: float) -> torch.Tensor:
    if scores.ndim != 1 or not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("Expected one finite recompute fraction in [0, 1]")
    count = math.ceil(len(scores) * budget)
    return scores.argsort(descending=True, stable=True)[:count].sort().values
