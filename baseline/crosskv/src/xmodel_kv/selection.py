from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .ridge import head_covariance, single_source_head_r2_from_stats
from .store import ActivationStore, validate_pair


def select_source_layers(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    top_k: int,
    ridge: float = 0.0,
    device: str = "cuda:0",
    max_observations: int | None = None,
) -> dict:
    source = ActivationStore(source_dir)
    target = ActivationStore(target_dir)
    validate_pair(source, target)
    if not 1 <= top_k <= source.metadata.num_layers:
        raise ValueError(f"top_k must be in [1,{source.metadata.num_layers}]")
    n = min(source.metadata.num_observations, max_observations or source.metadata.num_observations)
    scores = np.empty((target.metadata.num_layers, source.metadata.num_layers), dtype=np.float32)

    # The Qwen3 8B source store is ~19 GiB in fp16 / ~38 GiB in fp32, which fits
    # comfortably on an H100/H200. Keeping it resident avoids re-reading it for
    # every one of the 64 target layers.
    source_tensors: dict[tuple[str, int], torch.Tensor] = {}
    source_stats: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for kind in ("k", "v"):
        for source_layer in tqdm(range(source.metadata.num_layers), desc=f"preload source {kind}"):
            tensor = _to_device(source.open(kind, source_layer)[:n], device)
            source_tensors[(kind, source_layer)] = tensor
            source_stats[(kind, source_layer)] = head_covariance(tensor)

    for target_layer in tqdm(range(target.metadata.num_layers), desc="target layers"):
        target_k = _to_device(target.open("k", target_layer)[:n], device)
        target_v = _to_device(target.open("v", target_layer)[:n], device)
        target_k_mean = target_k.mean(dim=0)
        target_v_mean = target_v.mean(dim=0)
        target_k_variance = target_k.square().mean(dim=0) - target_k_mean.square()
        target_v_variance = target_v.square().mean(dim=0) - target_v_mean.square()
        for source_layer in range(source.metadata.num_layers):
            source_k = source_tensors[("k", source_layer)]
            source_v = source_tensors[("v", source_layer)]
            source_k_mean, source_k_covariance = source_stats[("k", source_layer)]
            source_v_mean, source_v_covariance = source_stats[("v", source_layer)]
            k_r2 = single_source_head_r2_from_stats(
                source_k,
                target_k,
                source_mean=source_k_mean,
                source_covariance=source_k_covariance,
                target_mean=target_k_mean,
                target_variance=target_k_variance,
                ridge=ridge,
            ).mean()
            v_r2 = single_source_head_r2_from_stats(
                source_v,
                target_v,
                source_mean=source_v_mean,
                source_covariance=source_v_covariance,
                target_mean=target_v_mean,
                target_variance=target_v_variance,
                ridge=ridge,
            ).mean()
            scores[target_layer, source_layer] = float((k_r2 + v_r2).mul(0.5).cpu())
        del target_k, target_v

    selected = []
    for row in scores:
        # Keep descending predictive order. This exact order is also the feature order used at fit/inference.
        indices = np.argsort(-row, kind="stable")[:top_k]
        selected.append([int(value) for value in indices])
    return {
        "source_dir": str(Path(source_dir).resolve()),
        "target_dir": str(Path(target_dir).resolve()),
        "top_k": top_k,
        "ridge": ridge,
        "num_observations": n,
        "selected_layers": selected,
        "scores": scores.tolist(),
    }


def save_selection(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def _to_device(array, device: str) -> torch.Tensor:
    # Copy because read-only NumPy memmaps cannot safely back writable torch tensors.
    return torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(device)
