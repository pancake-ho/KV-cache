from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RidgeResult:
    weight: torch.Tensor
    bias: torch.Tensor
    r2: torch.Tensor
    solver: str


def fit_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    ridge: float = 0.01,
    force_solver: str | None = None,
) -> RidgeResult:
    """Centered multi-output ridge with an unregularized recovered bias.

    Rows are observations. The returned weight follows ``y_hat = x @ weight + bias``.
    The optimized objective is ``mean_squared_error + ridge * ||weight||^2``;
    consequently the primal covariance is normalized by N. This normalization is
    required for the paper's lambda sweep (where lambda=1 has a material effect).
    The dual solve is mathematically equivalent and makes tiny smoke runs possible when
    the production feature dimension is larger than the observation count.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"expected x=[N,D], y=[N,O], got {x.shape}, {y.shape}")
    if x.shape[0] < 2:
        raise ValueError("ridge needs at least two observations")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    calculation_dtype = torch.float64 if x.device.type == "cpu" else torch.float32
    x = x.to(calculation_dtype)
    y = y.to(calculation_dtype)
    mean_x = x.mean(dim=0)
    mean_y = y.mean(dim=0)
    xc = x - mean_x
    yc = y - mean_y
    n, features = xc.shape
    solver = force_solver or ("primal" if features <= n else "dual")

    if solver == "primal":
        gram = (xc.T @ xc) / n
        gram.diagonal().add_(ridge)
        cross = (xc.T @ yc) / n
        weight = _symmetric_solve(gram, cross)
    elif solver == "dual":
        gram = xc @ xc.T
        # X^T (X X^T + N lambda I)^-1 Y is equivalent to the normalized primal.
        gram.diagonal().add_(n * ridge)
        alpha = _symmetric_solve(gram, yc)
        weight = xc.T @ alpha
    else:
        raise ValueError(f"unknown solver: {solver}")

    bias = mean_y - mean_x @ weight
    prediction_error = y - (x @ weight + bias)
    sse = prediction_error.square().sum(dim=0)
    sst = (y - mean_y).square().sum(dim=0)
    r2 = 1.0 - sse / sst.clamp_min(torch.finfo(sst.dtype).eps)
    return RidgeResult(weight=weight, bias=bias, r2=r2, solver=solver)


def single_source_head_r2(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 0.01,
) -> torch.Tensor:
    """R2 per matched head for the paper's single-source layer probe.

    Both inputs have shape [N, H, D]. Head h is fit independently to target head h.
    """
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError(f"matched [N,H,D] tensors required, got {source.shape}, {target.shape}")
    scores = []
    for head in range(source.shape[1]):
        scores.append(fit_ridge(source[:, head], target[:, head], ridge=ridge).r2.mean())
    return torch.stack(scores)


def head_covariance(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-head mean and centered covariance for [N,H,D] observations."""
    if tensor.ndim != 3:
        raise ValueError(f"expected [N,H,D], got {tensor.shape}")
    tensor = tensor.float()
    mean = tensor.mean(dim=0)
    raw_second_moment = torch.bmm(tensor.permute(1, 2, 0), tensor.permute(1, 0, 2)) / tensor.shape[0]
    covariance = raw_second_moment - mean.unsqueeze(-1) * mean.unsqueeze(-2)
    return mean, covariance


def single_source_head_r2_from_stats(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    source_mean: torch.Tensor,
    source_covariance: torch.Tensor,
    target_mean: torch.Tensor | None = None,
    target_variance: torch.Tensor | None = None,
    ridge: float = 0.0,
) -> torch.Tensor:
    """Vectorized matched-head R2 without materializing token-level predictions."""
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError(f"matched [N,H,D] tensors required, got {source.shape}, {target.shape}")
    source = source.float()
    target = target.float()
    n = source.shape[0]
    if target_mean is None:
        target_mean = target.mean(dim=0)
    if target_variance is None:
        target_variance = target.square().mean(dim=0) - target_mean.square()
    raw_cross = torch.bmm(source.permute(1, 2, 0), target.permute(1, 0, 2)) / n
    cross = raw_cross - source_mean.unsqueeze(-1) * target_mean.unsqueeze(-2)
    system = source_covariance.clone()
    system.diagonal(dim1=-2, dim2=-1).add_(ridge)
    weight = _batched_symmetric_solve(system, cross)
    linear = (weight * cross).sum(dim=1)
    quadratic = torch.einsum("hdo,hde,heo->ho", weight, source_covariance, weight)
    mse = (target_variance - 2.0 * linear + quadratic).clamp_min(0)
    eps = torch.finfo(target_variance.dtype).eps
    return 1.0 - mse / target_variance.clamp_min(eps)


def _symmetric_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info.max()) == 0:
        return torch.cholesky_solve(rhs, factor)
    # Useful for ridge=0 probes and numerically rank-deficient smoke data.
    return torch.linalg.lstsq(matrix, rhs).solution


def _batched_symmetric_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info.max()) == 0:
        return torch.cholesky_solve(rhs, factor)
    solutions = []
    for head in range(matrix.shape[0]):
        solutions.append(_symmetric_solve(matrix[head], rhs[head]))
    return torch.stack(solutions)
