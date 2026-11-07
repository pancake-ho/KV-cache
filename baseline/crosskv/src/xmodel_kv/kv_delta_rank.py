from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch


DEFAULT_RANKS = (1, 2, 4, 8, 16, 32)


def context_orders(
    pool_size: int,
    *,
    regime: str,
    num_contexts: int,
    chunks_per_context: int,
    generator: torch.Generator,
) -> list[tuple[int, ...]]:
    """Generate unique fixed-width context layouts.

    ``permutation`` holds the selected chunks fixed and varies only their order.
    ``composition`` varies both the selected chunks and their order.
    """

    if pool_size < chunks_per_context:
        raise ValueError("context pool is smaller than chunks_per_context")
    if num_contexts < 2:
        raise ValueError("num_contexts must be at least two")
    if regime not in {"permutation", "composition"}:
        raise ValueError(f"unknown context regime: {regime}")

    if regime == "permutation":
        maximum = math.factorial(chunks_per_context)
        if num_contexts > maximum:
            raise ValueError(
                f"requested {num_contexts} permutations, but only {maximum} are possible"
            )
        selected = torch.randperm(pool_size, generator=generator)[
            :chunks_per_context
        ].tolist()

    orders: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    maximum_attempts = max(10_000, num_contexts * 100)
    while len(orders) < num_contexts and attempts < maximum_attempts:
        attempts += 1
        if regime == "permutation":
            permutation = torch.randperm(chunks_per_context, generator=generator).tolist()
            order = tuple(selected[index] for index in permutation)
        else:
            order = tuple(
                torch.randperm(pool_size, generator=generator)[
                    :chunks_per_context
                ].tolist()
            )
        if order not in seen:
            seen.add(order)
            orders.append(order)
    if len(orders) != num_contexts:
        raise RuntimeError(
            f"could only generate {len(orders)} unique {regime} contexts"
        )
    return orders


def pca_rank_metrics(
    train: torch.Tensor,
    test: torch.Tensor,
    *,
    ranks: Sequence[int] = DEFAULT_RANKS,
) -> list[dict[str, Any]]:
    """Measure per-head train spectrum and held-out oracle PCA reconstruction.

    Args:
        train: Float tensor with shape ``[heads, train_contexts, features]``.
        test: Float tensor with shape ``[heads, test_contexts, features]``.

    The held-out coefficients are obtained by oracle projection onto a basis fit
    only on ``train``. Consequently this is an optimistic upper bound for a
    learned context-to-delta mapper using the same rank.
    """

    if train.ndim != 3 or test.ndim != 3:
        raise ValueError("train and test must have shape [heads, contexts, features]")
    if train.shape[0] != test.shape[0] or train.shape[2] != test.shape[2]:
        raise ValueError("train and test head/feature dimensions do not match")
    if train.shape[1] < 2 or test.shape[1] < 1:
        raise ValueError("need at least two train contexts and one test context")
    requested_ranks = tuple(sorted(set(int(rank) for rank in ranks)))
    if not requested_ranks or requested_ranks[0] < 1:
        raise ValueError("ranks must contain positive integers")

    train = train.float()
    test = test.float()
    center = train.mean(dim=1, keepdim=True)
    centered_train = train - center
    centered_test = test - center

    gram = torch.bmm(centered_train, centered_train.transpose(1, 2))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(-1).clamp_min_(0)
    eigenvectors = eigenvectors.flip(-1)
    train_energy = eigenvalues.sum(dim=-1)
    train_cumulative = eigenvalues.cumsum(dim=-1)
    safe_train_energy = train_energy.clamp_min(torch.finfo(train.dtype).tiny)
    train_fraction = train_cumulative / safe_train_energy.unsqueeze(-1)

    # Avoid materializing the very wide right-singular-vector matrix. If
    # Z = U S V^T, then Y V = (Y Z^T) U S^-1.
    cross_gram = torch.bmm(centered_test, centered_train.transpose(1, 2))
    max_eigenvalue = eigenvalues[:, :1]
    valid = eigenvalues > max_eigenvalue * 1e-10
    inverse_singular = torch.where(
        valid,
        eigenvalues.clamp_min(torch.finfo(train.dtype).tiny).rsqrt(),
        torch.zeros_like(eigenvalues),
    )
    coefficients = torch.bmm(cross_gram, eigenvectors) * inverse_singular.unsqueeze(1)
    test_component_energy = coefficients.square().sum(dim=1)
    test_cumulative = test_component_energy.cumsum(dim=-1)
    test_energy = centered_test.square().sum(dim=(1, 2))
    safe_test_energy = test_energy.clamp_min(torch.finfo(train.dtype).tiny)
    test_fraction = (test_cumulative / safe_test_energy.unsqueeze(-1)).clamp_(0, 1)

    probabilities = eigenvalues / safe_train_energy.unsqueeze(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
    entropy_rank = entropy.exp()
    stable_rank = train_energy / eigenvalues[:, 0].clamp_min(
        torch.finfo(train.dtype).tiny
    )

    results: list[dict[str, Any]] = []
    rank_ceiling = train.shape[1] - 1
    for head in range(train.shape[0]):
        if float(train_energy[head]) == 0:
            threshold_ranks = {threshold: 0 for threshold in (0.9, 0.95, 0.99)}
        else:
            threshold_ranks = {
                threshold: int(
                    (train_fraction[head] < threshold).sum().item() + 1
                )
                for threshold in (0.9, 0.95, 0.99)
            }
        train_at_rank: dict[str, float] = {}
        test_at_rank: dict[str, float] = {}
        captured_at_rank: dict[str, float] = {}
        for requested_rank in requested_ranks:
            effective_rank = min(requested_rank, rank_ceiling)
            key = f"r{requested_rank}"
            train_at_rank[key] = float(
                train_fraction[head, effective_rank - 1].item()
            )
            test_at_rank[key] = float(
                test_fraction[head, effective_rank - 1].item()
            )
            captured_at_rank[key] = float(
                test_cumulative[head, effective_rank - 1].item()
            )
        results.append(
            {
                "rank_ceiling": rank_ceiling,
                "r90": threshold_ranks[0.9],
                "r95": threshold_ranks[0.95],
                "r99": threshold_ranks[0.99],
                "stable_rank": float(stable_rank[head].item()),
                "entropy_rank": float(entropy_rank[head].item()),
                "train_energy": float(train_energy[head].item()),
                "test_energy": float(test_energy[head].item()),
                "train_explained": train_at_rank,
                "test_explained": test_at_rank,
                "test_captured_energy": captured_at_rank,
            }
        )
    return results


def matrix_delta_rank_metrics(
    deltas: torch.Tensor,
    *,
    ranks: Sequence[int] = DEFAULT_RANKS,
) -> list[dict[str, Any]]:
    """Measure the matrix rank of individual ``[tokens, head_dim]`` KV deltas.

    Args:
        deltas: Float tensor with shape ``[heads, deltas, tokens, head_dim]``.

    Unlike :func:`pca_rank_metrics`, every delta gets its own oracle SVD factors.
    This tests whether a dynamic mapper could emit a compact low-rank update even
    when the collection of updates does not share one fixed basis.
    """

    if deltas.ndim != 4:
        raise ValueError("deltas must have shape [heads, deltas, tokens, head_dim]")
    if deltas.shape[1] < 1:
        raise ValueError("need at least one delta matrix")
    requested_ranks = tuple(sorted(set(int(rank) for rank in ranks)))
    if not requested_ranks or requested_ranks[0] < 1:
        raise ValueError("ranks must contain positive integers")

    deltas = deltas.float()
    # Chunking avoids the less reliable very-large batched Jacobi solver path.
    # Symmetrization and a scale-relative diagonal shift make exactly-zero and
    # repeated-eigenvalue early-layer deltas well conditioned; the shift is
    # removed from the resulting eigenvalues.
    flat_deltas = deltas.flatten(0, 1)
    eigenvalue_chunks = []
    gram_size = min(deltas.shape[-2:])
    identity = torch.eye(gram_size, device=deltas.device, dtype=deltas.dtype)
    for begin in range(0, flat_deltas.shape[0], 32):
        chunk = flat_deltas[begin : begin + 32]
        if chunk.shape[-2] <= chunk.shape[-1]:
            gram = torch.matmul(chunk, chunk.transpose(-1, -2))
        else:
            gram = torch.matmul(chunk.transpose(-1, -2), chunk)
        gram = (gram + gram.transpose(-1, -2)) * 0.5
        shift = gram.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1) * 1e-6 + 1e-12
        shifted = gram + shift[:, None, None] * identity
        eigenvalue_chunks.append(
            torch.linalg.eigvalsh(shifted).flip(-1) - shift.unsqueeze(-1)
        )
    eigenvalues = torch.cat(eigenvalue_chunks).unflatten(
        0, (deltas.shape[0], deltas.shape[1])
    ).clamp_min_(0)
    energy = eigenvalues.sum(dim=-1)
    cumulative = eigenvalues.cumsum(dim=-1)
    safe_energy = energy.clamp_min(torch.finfo(deltas.dtype).tiny)
    fractions = cumulative / safe_energy.unsqueeze(-1)
    rank_ceiling = eigenvalues.shape[-1]

    results: list[dict[str, Any]] = []
    for head in range(deltas.shape[0]):
        threshold_ranks = {
            threshold: torch.where(
                energy[head] > 0,
                (fractions[head] < threshold).sum(dim=-1) + 1,
                torch.zeros_like(energy[head], dtype=torch.long),
            )
            for threshold in (0.9, 0.95, 0.99)
        }
        total_energy = float(energy[head].sum().item())
        explained: dict[str, float] = {}
        captured: dict[str, float] = {}
        for requested_rank in requested_ranks:
            effective_rank = min(requested_rank, rank_ceiling)
            key = f"r{requested_rank}"
            captured[key] = float(cumulative[head, :, effective_rank - 1].sum().item())
            explained[key] = captured[key] / total_energy if total_energy > 0 else 1.0
        results.append(
            {
                "num_deltas": deltas.shape[1],
                "rank_ceiling": rank_ceiling,
                "total_energy": total_energy,
                "captured_energy": captured,
                "energy_weighted_explained": explained,
                "r90": _tensor_distribution(threshold_ranks[0.9]),
                "r95": _tensor_distribution(threshold_ranks[0.95]),
                "r99": _tensor_distribution(threshold_ranks[0.99]),
            }
        )
    return results


def summarize_rank_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    cells = [cell for row in rows for cell in row["cells"]]
    if not cells:
        raise ValueError("no rank cells to summarize")

    regimes = sorted({str(row["regime"]) for row in rows})
    kinds = sorted({str(cell["kind"]) for cell in cells})
    by_regime = {
        regime: _aggregate_cells(
            [cell for row in rows if row["regime"] == regime for cell in row["cells"]]
        )
        for regime in regimes
    }
    summary = {
        "num_document_regimes": len(rows),
        "num_cells": len(cells),
        "overall": _aggregate_cells(cells),
        "by_regime": by_regime,
        "by_kind": {
            kind: _aggregate_cells([cell for cell in cells if cell["kind"] == kind])
            for kind in kinds
        },
        "by_regime_and_kind": {
            regime: {
                kind: _aggregate_cells(
                    [
                        cell
                        for row in rows
                        if row["regime"] == regime
                        for cell in row["cells"]
                        if cell["kind"] == kind
                    ]
                )
                for kind in kinds
            }
            for regime in regimes
        },
        "decision_by_regime": {
            regime: classify_low_rank(aggregate) for regime, aggregate in by_regime.items()
        },
        "decision_rule": {
            "strong_low_rank_go": "held-out energy-weighted r8 >= 0.90 and r16 >= 0.95",
            "moderate_low_rank": "held-out energy-weighted r16 >= 0.90 and r32 >= 0.95",
            "high_rank_reassess": "held-out energy-weighted r32 < 0.90",
            "inconclusive": "all other cases",
        },
    }
    return summary


def classify_low_rank(aggregate: dict[str, Any]) -> str:
    explained = aggregate["energy_weighted_test_explained"]
    if explained.get("r8", 0.0) >= 0.90 and explained.get("r16", 0.0) >= 0.95:
        return "strong_low_rank_go"
    if explained.get("r16", 0.0) >= 0.90 and explained.get("r32", 0.0) >= 0.95:
        return "moderate_low_rank"
    if explained.get("r32", 0.0) < 0.90:
        return "high_rank_reassess"
    return "inconclusive"


def _aggregate_cells(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not cells:
        return {}
    rank_keys = tuple(cells[0]["test_explained"])
    total_test_energy = sum(float(cell["test_energy"]) for cell in cells)
    weighted = {
        rank: (
            sum(float(cell["test_captured_energy"][rank]) for cell in cells)
            / total_test_energy
            if total_test_energy > 0
            else 1.0
        )
        for rank in rank_keys
    }
    return {
        "cells": len(cells),
        "energy_weighted_test_explained": weighted,
        "median_test_explained": {
            rank: _quantile([float(cell["test_explained"][rank]) for cell in cells], 0.5)
            for rank in rank_keys
        },
        "r95": _distribution([float(cell["r95"]) for cell in cells]),
        "stable_rank": _distribution(
            [float(cell["stable_rank"]) for cell in cells]
        ),
        "entropy_rank": _distribution(
            [float(cell["entropy_rank"]) for cell in cells]
        ),
        "centered_relative_l2": _distribution(
            [float(cell["centered_relative_l2"]) for cell in cells]
        ),
        "mean_isolated_delta_relative_l2": _distribution(
            [float(cell["mean_isolated_delta_relative_l2"]) for cell in cells]
        ),
        "paired_context_matrix_delta": _aggregate_matrix_deltas(
            cells, "paired_context_matrix_delta"
        ),
        "isolated_matrix_delta": _aggregate_matrix_deltas(
            cells, "isolated_matrix_delta"
        ),
        "total_test_energy": total_test_energy,
    }


def _aggregate_matrix_deltas(
    cells: Sequence[dict[str, Any]], field: str
) -> dict[str, Any]:
    rank_keys = tuple(cells[0][field]["energy_weighted_explained"])
    total_energy = sum(float(cell[field]["total_energy"]) for cell in cells)
    explained = {
        rank: (
            sum(float(cell[field]["captured_energy"][rank]) for cell in cells)
            / total_energy
            if total_energy > 0
            else 1.0
        )
        for rank in rank_keys
    }
    return {
        "energy_weighted_explained": explained,
        "cell_median_r95": _distribution(
            [float(cell[field]["r95"]["median"]) for cell in cells]
        ),
        "cell_p90_r95": _distribution(
            [float(cell[field]["r95"]["p90"]) for cell in cells]
        ),
        "total_energy": total_energy,
        "decision": classify_low_rank(
            {"energy_weighted_test_explained": explained}
        ),
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "p10": _quantile(values, 0.1),
        "median": _quantile(values, 0.5),
        "p90": _quantile(values, 0.9),
    }


def _tensor_distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.to(dtype=torch.float64)
    return {
        "p10": float(torch.quantile(values, 0.1).item()),
        "median": float(torch.quantile(values, 0.5).item()),
        "p90": float(torch.quantile(values, 0.9).item()),
    }


def _quantile(values: Sequence[float], quantile: float) -> float:
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    return float(torch.quantile(tensor, quantile).item())
