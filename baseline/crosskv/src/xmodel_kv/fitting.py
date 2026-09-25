from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from .artifact import MapperConfig
from .ridge import fit_ridge
from .store import ActivationStore, validate_pair


def fit_mapper(
    *,
    source_dir: str | Path,
    target_dir: str | Path,
    selection_path: str | Path,
    output_dir: str | Path,
    ridge: float = 0.01,
    device: str = "cuda:0",
    max_observations: int | None = None,
    weight_dtype: str = "float32",
    resume: bool = True,
    target_layer_start: int = 0,
    target_layer_end: int | None = None,
) -> MapperConfig:
    source = ActivationStore(source_dir)
    target = ActivationStore(target_dir)
    validate_pair(source, target)
    with Path(selection_path).open() as handle:
        selection = json.load(handle)
    selected_layers = selection["selected_layers"]
    if len(selected_layers) != target.metadata.num_layers:
        raise ValueError("selection target-layer count does not match activation store")
    k_values = {len(layers) for layers in selected_layers}
    if len(k_values) != 1:
        raise ValueError("all target layers must select the same k")
    target_layer_end = target_layer_end if target_layer_end is not None else target.metadata.num_layers
    if not 0 <= target_layer_start <= target_layer_end <= target.metadata.num_layers:
        raise ValueError(
            f"invalid target layer range [{target_layer_start}, {target_layer_end}) "
            f"for {target.metadata.num_layers} layers"
        )

    output_dir = Path(output_dir)
    layers_dir = output_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    n = min(source.metadata.num_observations, max_observations or source.metadata.num_observations)
    config = MapperConfig(
        source_model=source.metadata.model_path,
        target_model=target.metadata.model_path,
        source_layers=source.metadata.num_layers,
        target_layers=target.metadata.num_layers,
        source_kv_heads=source.metadata.num_kv_heads,
        target_kv_heads=target.metadata.num_kv_heads,
        source_head_dim=source.metadata.head_dim,
        target_head_dim=target.metadata.head_dim,
        selected_layers=selected_layers,
        ridge=ridge,
        weight_dtype=weight_dtype,
        num_observations=n,
    )
    if resume and (output_dir / "config.json").exists():
        previous = MapperConfig.load(output_dir)
        if previous != config:
            raise ValueError(
                "existing mapper config differs from this fit; choose a new output directory "
                "or pass --no-resume to replace it"
            )
    config.save(output_dir)

    target_layer_range = range(target_layer_start, target_layer_end)
    for target_layer in tqdm(target_layer_range, desc="fit target layers"):
        selected = selected_layers[target_layer]
        output_path = layers_dir / f"layer_{target_layer:03d}.safetensors"
        if resume and output_path.exists():
            continue
        tensors: dict[str, torch.Tensor] = {}
        for kind in ("k", "v"):
            x_parts = []
            for source_layer in selected:
                array = np.array(source.open(kind, source_layer)[:n], dtype=np.float32, copy=True)
                x_parts.append(torch.from_numpy(array.reshape(n, -1)))
            x = torch.cat(x_parts, dim=1).to(device)
            y_array = np.array(target.open(kind, target_layer)[:n], dtype=np.float32, copy=True)
            y = torch.from_numpy(y_array.reshape(n, -1)).to(device)
            result = fit_ridge(x, y, ridge=ridge)
            tensors[f"{kind}_weight"] = _cast_for_storage(
                result.weight.reshape(
                    result.weight.shape[0],
                    target.metadata.num_kv_heads,
                    target.metadata.head_dim,
                ).cpu(),
                weight_dtype,
            ).contiguous()
            tensors[f"{kind}_bias"] = _cast_for_storage(
                result.bias.reshape(target.metadata.num_kv_heads, target.metadata.head_dim).cpu(),
                weight_dtype,
            ).contiguous()
            tensors[f"{kind}_r2"] = result.r2.reshape(
                target.metadata.num_kv_heads, target.metadata.head_dim
            ).float().cpu().contiguous()
            del x, y, x_parts, y_array, result
            torch.cuda.empty_cache()
        save_file(tensors, output_path)
    return config


def _cast_for_storage(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtypes:
        raise ValueError(f"unsupported storage dtype: {dtype}")
    return tensor.to(dtypes[dtype])
