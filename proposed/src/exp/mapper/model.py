"""NVIDIA Cross-Model KV paper (2026), Appendix E MLP architecture."""
from __future__ import annotations

import torch
from torch import nn


class HeadMLP(nn.Module):
    """Independent mapper for one (target layer, target head, K or V)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        if min(input_dim, output_dim, hidden_dim) <= 0:
            raise ValueError("input/output/hidden dimensions must be positive")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
