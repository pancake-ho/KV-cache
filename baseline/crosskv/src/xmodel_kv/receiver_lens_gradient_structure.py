from __future__ import annotations

import math

import numpy as np
import torch


ROUTES = ("final", "bridge", "joint")


def analyze_receiver_lens_gradients(
    dev: dict[str, torch.Tensor],
    musique: dict[str, torch.Tensor],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    """Analyze frozen per-case lens gradients without retaining model state."""

    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    dev = _validated_group(dev, name="dev")
    musique = _validated_group(musique, name="musique")
    if dev["joint"].shape[1] != musique["joint"].shape[1]:
        raise ValueError("gradient groups must share the parameter dimension")

    rng = np.random.default_rng(seed)
    comparisons = {
        "dev_final_vs_musique_bridge": aggregate_cosine_bootstrap(
            dev["final"],
            musique["bridge"],
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
            paired=False,
        ),
        "dev_joint_vs_musique_joint": aggregate_cosine_bootstrap(
            dev["joint"],
            musique["joint"],
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
            paired=False,
        ),
        "dev_final_vs_bridge": aggregate_cosine_bootstrap(
            dev["final"],
            dev["bridge"],
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
            paired=True,
        ),
        "musique_final_vs_bridge": aggregate_cosine_bootstrap(
            musique["final"],
            musique["bridge"],
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
            paired=True,
        ),
    }

    dev_unit = normalize_rows(dev["joint"])
    musique_unit = normalize_rows(musique["joint"])
    margins = domain_centroid_margins(dev_unit, musique_unit)
    margin_values = torch.cat((margins["dev"], margins["musique"])).numpy()
    domain_structure = {
        "mean_own_minus_other_centroid_cosine": float(margin_values.mean()),
        "bootstrap_95ci": bootstrap_mean_interval(
            margin_values,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        ),
        "positive_fraction": float(np.mean(margin_values > 0)),
        "dev_mean_margin": float(margins["dev"].mean()),
        "musique_mean_margin": float(margins["musique"].mean()),
        "mean_pairwise_cosines": {
            "dev_within": off_diagonal_mean(dev_unit @ dev_unit.T),
            "musique_within": off_diagonal_mean(musique_unit @ musique_unit.T),
            "cross": float((dev_unit @ musique_unit.T).mean()),
        },
    }

    spectrum = normalized_gradient_spectrum(torch.cat((dev_unit, musique_unit)))
    conflict = comparisons["dev_final_vs_musique_bridge"]
    conditional = domain_structure
    gates = {
        "cross_domain_conflict": conflict["bootstrap_95ci"][1] < 0.0,
        "domain_conditioned_directions": conditional["bootstrap_95ci"][0] > 0.0,
    }
    gates["conditional_writer_authorized"] = all(gates.values())
    return {
        "comparisons": comparisons,
        "domain_structure": domain_structure,
        "normalized_joint_gradient_spectrum": spectrum,
        "gates": gates,
    }


def aggregate_cosine_bootstrap(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    bootstrap_replicates: int,
    rng: np.random.Generator,
    paired: bool,
    batch_size: int = 512,
) -> dict:
    """Bootstrap the cosine between two aggregate gradients via Gram matrices."""

    first = _matrix(first, "first").double()
    second = _matrix(second, "second").double()
    if first.shape[1] != second.shape[1]:
        raise ValueError("gradient matrices must share the parameter dimension")
    if paired and first.shape[0] != second.shape[0]:
        raise ValueError("paired gradient matrices must share the case dimension")
    if min(bootstrap_replicates, batch_size) < 1:
        raise ValueError("bootstrap_replicates and batch_size must be positive")

    first_np = first.numpy()
    second_np = second.numpy()
    first_gram = first_np @ first_np.T
    second_gram = second_np @ second_np.T
    cross_gram = first_np @ second_np.T
    point = _cosine(first_np.mean(axis=0), second_np.mean(axis=0))
    samples = np.empty(bootstrap_replicates, dtype=np.float64)
    written = 0
    while written < bootstrap_replicates:
        batch = min(batch_size, bootstrap_replicates - written)
        first_weights = _bootstrap_weights(rng, batch=batch, cases=len(first_np))
        second_weights = (
            first_weights
            if paired
            else _bootstrap_weights(rng, batch=batch, cases=len(second_np))
        )
        dots = np.sum((first_weights @ cross_gram) * second_weights, axis=1)
        first_squares = np.sum(
            (first_weights @ first_gram) * first_weights, axis=1
        )
        second_squares = np.sum(
            (second_weights @ second_gram) * second_weights, axis=1
        )
        denominator = np.sqrt(
            np.maximum(first_squares, 0.0) * np.maximum(second_squares, 0.0)
        )
        values = np.divide(
            dots,
            denominator,
            out=np.zeros_like(dots),
            where=denominator > 1e-30,
        )
        samples[written : written + batch] = np.clip(values, -1.0, 1.0)
        written += batch
    return {
        "aggregate_cosine": point,
        "bootstrap_95ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "paired_case_resampling": paired,
    }


def domain_centroid_margins(
    dev_unit: torch.Tensor, musique_unit: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return own leave-one-out centroid cosine minus other-centroid cosine."""

    dev_unit = _matrix(dev_unit, "dev_unit").double()
    musique_unit = _matrix(musique_unit, "musique_unit").double()
    if dev_unit.shape[1] != musique_unit.shape[1]:
        raise ValueError("normalized groups must share the parameter dimension")
    if min(dev_unit.shape[0], musique_unit.shape[0]) < 2:
        raise ValueError("each domain needs at least two cases")
    dev_sum = dev_unit.sum(dim=0)
    musique_sum = musique_unit.sum(dim=0)
    dev_other = normalize_vector(musique_sum)
    musique_other = normalize_vector(dev_sum)
    dev_own = normalize_rows(dev_sum.unsqueeze(0) - dev_unit)
    musique_own = normalize_rows(musique_sum.unsqueeze(0) - musique_unit)
    return {
        "dev": (dev_unit * dev_own).sum(dim=1)
        - (dev_unit * dev_other).sum(dim=1),
        "musique": (musique_unit * musique_own).sum(dim=1)
        - (musique_unit * musique_other).sum(dim=1),
    }


def normalized_gradient_spectrum(unit_gradients: torch.Tensor) -> dict:
    unit_gradients = _matrix(unit_gradients, "unit_gradients").double()
    eigenvalues = torch.linalg.eigvalsh(unit_gradients @ unit_gradients.T)
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    total = float(eigenvalues.sum())
    if total <= 0:
        raise ValueError("gradient spectrum has zero energy")
    probabilities = eigenvalues / total
    nonzero = probabilities > 0
    entropy = float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    cumulative = probabilities.cumsum(0)

    def energy_rank(fraction: float) -> int:
        return int(torch.searchsorted(cumulative, fraction).item()) + 1

    return {
        "cases": int(unit_gradients.shape[0]),
        "parameter_dimension": int(unit_gradients.shape[1]),
        "stable_rank": float(total / max(float(eigenvalues[0]), 1e-30)),
        "entropy_effective_rank": float(math.exp(entropy)),
        "rank_50_energy": energy_rank(0.50),
        "rank_90_energy": energy_rank(0.90),
        "rank_95_energy": energy_rank(0.95),
        "leading_energy_fractions": [
            float(value) for value in probabilities[: min(10, len(probabilities))]
        ],
    }


def normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    matrix = _matrix(matrix, "matrix").double()
    norms = matrix.norm(dim=1, keepdim=True)
    if torch.any(norms <= 1e-30):
        raise ValueError("gradient rows must have non-zero norm")
    return matrix / norms


def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.double().flatten()
    norm = vector.norm()
    if norm <= 1e-30:
        raise ValueError("centroid vector must have non-zero norm")
    return vector / norm


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    bootstrap_replicates: int,
    rng: np.random.Generator,
    batch_size: int = 2048,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or bootstrap_replicates < 1:
        raise ValueError("values and bootstrap_replicates must be non-empty")
    samples = np.empty(bootstrap_replicates, dtype=np.float64)
    written = 0
    while written < bootstrap_replicates:
        batch = min(batch_size, bootstrap_replicates - written)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        samples[written : written + batch] = values[indices].mean(axis=1)
        written += batch
    return [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def off_diagonal_mean(gram: torch.Tensor) -> float:
    gram = _matrix(gram, "gram").double()
    if gram.shape[0] != gram.shape[1] or gram.shape[0] < 2:
        raise ValueError("gram must be square with at least two rows")
    return float((gram.sum() - gram.diagonal().sum()) / (gram.numel() - len(gram)))


def _validated_group(group: dict[str, torch.Tensor], *, name: str):
    if set(group) != set(ROUTES):
        raise ValueError(f"{name} must contain exactly {ROUTES}")
    validated = {route: _matrix(group[route], f"{name}.{route}") for route in ROUTES}
    shapes = {tuple(matrix.shape) for matrix in validated.values()}
    if len(shapes) != 1:
        raise ValueError(f"{name} gradient matrices must share shape")
    return validated


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).detach().cpu()
    if value.ndim != 2 or min(value.shape) < 1 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite non-empty matrix")
    return value


def _bootstrap_weights(
    rng: np.random.Generator, *, batch: int, cases: int
) -> np.ndarray:
    indices = rng.integers(0, cases, size=(batch, cases))
    weights = np.zeros((batch, cases), dtype=np.float64)
    rows = np.broadcast_to(np.arange(batch)[:, None], indices.shape)
    np.add.at(weights, (rows, indices), 1.0 / cases)
    return weights


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-30:
        raise ValueError("aggregate gradient norm is zero")
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
