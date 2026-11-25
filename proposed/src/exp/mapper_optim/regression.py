"""Memory-bounded affine ridge and calibration-only layer selection."""
from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy as np
import torch

BatchFactory = Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]


def fit_ridge(batches: BatchFactory, first: tuple[int, int], *,
              ridge: float, device: torch.device, dtype: torch.dtype):
    """Return W, bias, means and centered Gram for sum-loss ridge.

    The callable is replayed for weighted means and for centered statistics.
    A head or layer is fitted at a time using fixed-size observation panels.
    """
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    width, outputs = first
    mx = torch.zeros(width, device=device, dtype=dtype)
    my = torch.zeros(outputs, device=device, dtype=dtype)
    mass = torch.zeros((), device=device, dtype=dtype)
    for x, y, w in batches():
        mx.add_((x * w).sum(dim=0))
        my.add_((y * w).sum(dim=0))
        mass.add_(w.sum())
    if not torch.isfinite(mass) or mass <= 0:
        raise ValueError("Invalid calibration weight mass")
    mx /= mass
    my /= mass
    gram = torch.zeros((width, width), device=device, dtype=dtype)
    cross = torch.zeros((width, outputs), device=device, dtype=dtype)
    for x, y, w in batches():
        xc = x - mx
        wx = xc * w.sqrt()
        wy = (y - my) * w.sqrt()
        gram.addmm_(wx.T, wx)
        cross.addmm_(wx.T, wy)
    gram.diagonal().add_(ridge)
    factor, info = torch.linalg.cholesky_ex(gram)
    gram.diagonal().sub_(ridge)
    if info.item() != 0:
        raise RuntimeError("Ridge system is not positive definite. Use float64 or increase ridge")
    weight = torch.cholesky_solve(cross, factor)
    bias = my - mx @ weight
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise FloatingPointError("Non-finite ridge solution")
    return weight, bias, mx, my, gram


def _probe_tensor(array: np.ndarray, rows: np.ndarray, device: torch.device,
                  head_local: bool) -> torch.Tensor:
    data = torch.tensor(array[rows], device=device, dtype=torch.float32)
    data -= data.mean(dim=0)
    if head_local:
        return data.transpose(0, 1).contiguous()
    return data.flatten(start_dim=1).unsqueeze(0)


def select_layers(source, target, *, head_local: bool, top_k: int,
                  num_rows: int, sampled: int, device: torch.device) -> dict:
    """Rank source layers by held-in OLS R² on deterministic calibration rows."""
    rows = np.linspace(0, num_rows - 1, min(sampled, num_rows), dtype=np.int64)
    source_width = source.width if head_local else source.width * source.heads
    if len(rows) <= source_width:
        raise ValueError(f"OLS selection needs >{source_width} probe rows, got {len(rows)}")
    if not 1 <= top_k <= source.layers:
        raise ValueError("top_k must be within the source layer count")

    selections = {}
    for kind in ("k", "v"):
        scores = torch.empty((target.layers, source.layers), device=device)
        # Build one source design at a time. Keep only sampled target rows on device.
        targets = [_probe_tensor(target.array(kind, l), rows, device, head_local)
                   for l in range(target.layers)]
        for layer in range(source.layers):
            x = _probe_tensor(source.array(kind, layer), rows, device, head_local)
            u, singular, _ = torch.linalg.svd(x, full_matrices=False)
            valid = singular > singular[:, :1] * (max(x.shape[-2:]) * torch.finfo(x.dtype).eps)
            basis = u * valid.unsqueeze(-2)
            for out, y in enumerate(targets):
                explained = (basis.transpose(-1, -2) @ y).square().sum(dim=(-2, -1))
                total = y.square().sum(dim=(-2, -1))
                per_head = torch.where(total > 0, explained / total.clamp_min(1e-20), 0)
                scores[out, layer] = per_head.mean()
        selections[kind] = scores.cpu()
    if head_local:
        # CacheBridge v1 shares one ordered support across K, V, and all heads.
        shared = (selections["k"] + selections["v"]) * 0.5
        selections = {"k": shared, "v": shared}
    return {kind: score.argsort(descending=True, stable=True)[:, :top_k].tolist()
            for kind, score in selections.items()}
