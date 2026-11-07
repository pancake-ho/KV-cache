from __future__ import annotations

import torch

from xmodel_kv.kv_delta_rank import (
    classify_low_rank,
    context_orders,
    matrix_delta_rank_metrics,
    pca_rank_metrics,
)


def test_permutation_contexts_hold_composition_fixed() -> None:
    generator = torch.Generator().manual_seed(7)
    orders = context_orders(
        30,
        regime="permutation",
        num_contexts=20,
        chunks_per_context=5,
        generator=generator,
    )
    assert len(orders) == len(set(orders)) == 20
    assert all(set(order) == set(orders[0]) for order in orders)


def test_composition_contexts_vary_selected_chunks() -> None:
    generator = torch.Generator().manual_seed(11)
    orders = context_orders(
        30,
        regime="composition",
        num_contexts=20,
        chunks_per_context=5,
        generator=generator,
    )
    assert len(orders) == len(set(orders)) == 20
    assert len({frozenset(order) for order in orders}) > 1


def test_heldout_pca_recovers_known_low_rank_subspace() -> None:
    generator = torch.Generator().manual_seed(13)
    heads, features, latent = 2, 40, 3
    basis = torch.randn(heads, latent, features, generator=generator)
    train_coefficients = torch.randn(heads, 24, latent, generator=generator)
    test_coefficients = torch.randn(heads, 12, latent, generator=generator)
    offset = torch.randn(heads, 1, features, generator=generator)
    train = torch.bmm(train_coefficients, basis) + offset
    test = torch.bmm(test_coefficients, basis) + offset

    metrics = pca_rank_metrics(train, test, ranks=(1, 2, 3, 4, 8))

    assert all(metric["r99"] <= latent for metric in metrics)
    assert all(metric["test_explained"]["r3"] > 0.999 for metric in metrics)


def test_heldout_pca_rejects_small_basis_for_isotropic_delta() -> None:
    generator = torch.Generator().manual_seed(17)
    train = torch.randn(1, 64, 128, generator=generator)
    test = torch.randn(1, 32, 128, generator=generator)

    metric = pca_rank_metrics(train, test, ranks=(4, 8, 16, 32))[0]

    assert metric["r95"] > 32
    assert metric["test_explained"]["r8"] < 0.2
    assert metric["test_explained"]["r32"] < 0.5


def test_individual_matrix_delta_rank_recovers_low_rank_factors() -> None:
    generator = torch.Generator().manual_seed(19)
    left = torch.randn(2, 7, 24, 3, generator=generator)
    right = torch.randn(2, 7, 3, 20, generator=generator)
    deltas = torch.matmul(left, right)

    metrics = matrix_delta_rank_metrics(deltas, ranks=(1, 2, 3, 4, 8))

    assert all(metric["r99"]["p90"] <= 3 for metric in metrics)
    assert all(metric["energy_weighted_explained"]["r3"] > 0.999 for metric in metrics)


def test_individual_random_matrix_delta_is_not_low_rank() -> None:
    generator = torch.Generator().manual_seed(23)
    deltas = torch.randn(1, 6, 48, 48, generator=generator)

    metric = matrix_delta_rank_metrics(deltas, ranks=(4, 8, 16, 32))[0]

    assert metric["r95"]["median"] > 24
    assert metric["energy_weighted_explained"]["r8"] < 0.6


def test_predeclared_decision_rule() -> None:
    assert (
        classify_low_rank(
            {"energy_weighted_test_explained": {"r8": 0.91, "r16": 0.96, "r32": 0.98}}
        )
        == "strong_low_rank_go"
    )
    assert (
        classify_low_rank(
            {"energy_weighted_test_explained": {"r8": 0.50, "r16": 0.80, "r32": 0.85}}
        )
        == "high_rank_reassess"
    )
