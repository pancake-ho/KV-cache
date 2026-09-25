from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


EPS = 1e-12


def _absolute_stats(
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, float | int]:
    """
    a, b: [T, H, D]
    """
    if a.shape != b.shape:
        raise RuntimeError(
            "Feature shape mismatch: "
            f"{list(a.shape)} "
            f"vs {list(b.shape)}"
        )

    a = a.float()
    b = b.float()

    cos = F.cosine_similarity(
        a,
        b,
        dim=-1,
        eps=1e-8,
    ).reshape(-1)

    global_cos = (
        F.cosine_similarity(
            a.reshape(1, -1),
            b.reshape(1, -1),
            dim=-1,
            eps=1e-8,
        )
        .item()
    )

    norm_a = torch.linalg.vector_norm(
        a
    )

    norm_b = torch.linalg.vector_norm(
        b
    )

    diff_norm = (
        torch.linalg.vector_norm(
            a - b
        )
    )

    symmetric_rel_l2 = (
        2.0
        * diff_norm
        / (
            norm_a
            + norm_b
            + EPS
        )
    ).item()

    return {
        "num_vectors": int(
            cos.numel()
        ),
        "seq_len": int(
            a.shape[0]
        ),
        "num_heads": int(
            a.shape[1]
        ),
        "cosine_mean": float(
            cos.mean()
        ),
        "cosine_median": float(
            cos.median()
        ),
        "cosine_std": float(
            cos.std(
                unbiased=False
            )
        ),
        "cosine_p05": float(
            torch.quantile(
                cos,
                0.05,
            )
        ),
        "cosine_p95": float(
            torch.quantile(
                cos,
                0.95,
            )
        ),
        "fraction_cos_ge_095": float(
            (cos >= 0.95)
            .float()
            .mean()
        ),
        "fraction_cos_ge_099": float(
            (cos >= 0.99)
            .float()
            .mean()
        ),
        "global_cosine": float(
            global_cos
        ),
        "symmetric_rel_l2": float(
            symmetric_rel_l2
        ),
    }


def _delta_stats(
    a: torch.Tensor,
    b: torch.Tensor,
    base: torch.Tensor,
) -> dict[str, float]:
    if (
        a.shape != b.shape
        or a.shape != base.shape
    ):
        raise RuntimeError(
            "Delta feature shape mismatch."
        )

    a = a.float()
    b = b.float()
    base = base.float()

    da = a - base
    db = b - base

    base_norm = (
        torch.linalg.vector_norm(
            base
        )
    )

    da_norm = (
        torch.linalg.vector_norm(
            da
        )
    )

    db_norm = (
        torch.linalg.vector_norm(
            db
        )
    )

    delta_global_cosine = (
        F.cosine_similarity(
            da.reshape(1, -1),
            db.reshape(1, -1),
            dim=-1,
            eps=1e-8,
        )
        .item()
    )

    delta_distance = (
        torch.linalg.vector_norm(
            da - db
        )
        / (
            base_norm
            + EPS
        )
    ).item()

    return {
        "delta_global_cosine": float(
            delta_global_cosine
        ),
        "delta_norm_ratio_a": float(
            da_norm
            / (
                base_norm
                + EPS
            )
        ),
        "delta_norm_ratio_b": float(
            db_norm
            / (
                base_norm
                + EPS
            )
        ),
        "delta_distance_vs_base": float(
            delta_distance
        ),
    }


def compare_feature_sets(
    features_a: dict[
        str,
        list[torch.Tensor],
    ],
    features_b: dict[
        str,
        list[torch.Tensor],
    ],
    *,
    base_features: dict[
        str,
        list[torch.Tensor],
    ] | None = None,
) -> list[dict[str, Any]]:
    rows = []

    common_types = sorted(
        set(features_a)
        & set(features_b)
    )

    for tensor_type in common_types:
        layers_a = features_a[
            tensor_type
        ]

        layers_b = features_b[
            tensor_type
        ]

        if len(layers_a) != len(
            layers_b
        ):
            raise RuntimeError(
                "Layer count mismatch."
            )

        for layer_idx, (
            tensor_a,
            tensor_b,
        ) in enumerate(
            zip(
                layers_a,
                layers_b,
            )
        ):
            row = {
                "tensor_type": (
                    tensor_type
                ),
                "layer": int(
                    layer_idx
                ),
            }

            row.update(
                _absolute_stats(
                    tensor_a,
                    tensor_b,
                )
            )

            if base_features is not None:
                base_tensor = (
                    base_features[
                        tensor_type
                    ][layer_idx]
                )

                row.update(
                    _delta_stats(
                        tensor_a,
                        tensor_b,
                        base_tensor,
                    )
                )
            else:
                row.update(
                    {
                        "delta_global_cosine": (
                            math.nan
                        ),
                        "delta_norm_ratio_a": (
                            math.nan
                        ),
                        "delta_norm_ratio_b": (
                            math.nan
                        ),
                        "delta_distance_vs_base": (
                            math.nan
                        ),
                    }
                )

            rows.append(row)

    return rows