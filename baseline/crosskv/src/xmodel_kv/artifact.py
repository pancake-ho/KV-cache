from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import DynamicCache

from .rope import apply_rope, model_rope_cos_sin, remove_rope


@dataclass
class MapperConfig:
    source_model: str
    target_model: str
    source_layers: int
    target_layers: int
    source_kv_heads: int
    target_kv_heads: int
    source_head_dim: int
    target_head_dim: int
    selected_layers: list[list[int]]
    ridge: float = 0.01
    weight_dtype: str = "float32"
    num_observations: int | None = None
    covariance_normalization: str = "mean"
    format_version: int = 1

    @classmethod
    def load(cls, root: str | Path) -> "MapperConfig":
        with (Path(root) / "config.json").open() as handle:
            return cls(**json.load(handle))

    def save(self, root: str | Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        temporary = root / f".config.{os.getpid()}.tmp"
        with temporary.open("w") as handle:
            json.dump(asdict(self), handle, indent=2)
            handle.write("\n")
        os.replace(temporary, root / "config.json")


class LinearKVMapper:
    """Apply a fitted per-target-layer/per-head linear KV mapper."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.config = MapperConfig.load(root)
        self._host_layers: dict[int, dict[str, torch.Tensor]] = {}
        self._device_layers: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    def layer_path(self, layer: int) -> Path:
        return self.root / "layers" / f"layer_{layer:03d}.safetensors"

    def load_layer(self, layer: int) -> dict[str, torch.Tensor]:
        if layer not in self._host_layers:
            self._host_layers[layer] = load_file(self.layer_path(layer), device="cpu")
        return self._host_layers[layer]

    def preload(self) -> None:
        for layer in range(self.config.target_layers):
            self.load_layer(layer)

    def preload_for_model(self, target_model) -> None:
        """Page the complete mapper onto the devices holding target layers."""
        target_backbone = getattr(target_model, "model", target_model)
        for layer in range(self.config.target_layers):
            self._layer_on_device(layer, _layer_device(target_backbone.layers[layer]))

    def clear_device_cache(self) -> None:
        self._device_layers.clear()

    @torch.inference_mode()
    def map_cache(
        self,
        source_cache: Any,
        *,
        source_model,
        target_model,
        positions: torch.Tensor | None = None,
    ) -> DynamicCache:
        source_legacy = _legacy_cache(source_cache)
        self._validate_runtime(source_legacy, source_model, target_model)
        sequence_length = source_legacy[0][0].shape[-2]
        batch_size = source_legacy[0][0].shape[0]
        if positions is None:
            positions = torch.arange(sequence_length, dtype=torch.long)
        valid_unbatched = positions.ndim == 1 and positions.shape[0] == sequence_length
        valid_batched = positions.ndim == 2 and positions.shape == (batch_size, sequence_length)
        if not (valid_unbatched or valid_batched):
            raise ValueError(
                "positions must have shape [cached_tokens] or [batch, cached_tokens]"
            )

        source_rope_device = _model_input_device(source_model)
        source_cos, source_sin = model_rope_cos_sin(
            source_model,
            positions,
            device=source_rope_device,
            dtype=torch.float32,
        )
        target_rope_device = _model_input_device(target_model)
        target_cos, target_sin = model_rope_cos_sin(
            target_model,
            positions,
            device=target_rope_device,
            dtype=torch.float32,
        )

        mapped_layers = []
        target_backbone = getattr(target_model, "model", target_model)
        for target_layer, selected in enumerate(self.config.selected_layers):
            device = _layer_device(target_backbone.layers[target_layer])
            tensors = self._layer_on_device(target_layer, device)
            source_cos_device = source_cos.to(device=device)
            source_sin_device = source_sin.to(device=device)

            key_features = []
            value_features = []
            for source_layer in selected:
                source_key, source_value = source_legacy[source_layer]
                source_key = source_key.to(device=device, dtype=torch.float32)
                content_key = remove_rope(
                    source_key,
                    source_cos_device,
                    source_sin_device,
                    sequence_dim=2,
                )
                key_features.append(content_key.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1))
                value_features.append(
                    source_value.to(device=device, dtype=torch.float32)
                    .permute(0, 2, 1, 3)
                    .reshape(batch_size, sequence_length, -1)
                )

            x_key = torch.cat(key_features, dim=-1)
            x_value = torch.cat(value_features, dim=-1)
            mapped_key = _project(x_key, tensors["k_weight"], tensors["k_bias"])
            mapped_value = _project(x_value, tensors["v_weight"], tensors["v_bias"])
            mapped_key = apply_rope(
                mapped_key,
                target_cos.to(device=device),
                target_sin.to(device=device),
                sequence_dim=2,
            )
            target_dtype = _layer_dtype(target_backbone.layers[target_layer])
            mapped_layers.append((mapped_key.to(target_dtype), mapped_value.to(target_dtype)))

        return DynamicCache.from_legacy_cache(tuple(mapped_layers))

    def _validate_runtime(self, source_legacy, source_model, target_model) -> None:
        if len(source_legacy) != self.config.source_layers:
            raise ValueError(f"source cache has {len(source_legacy)} layers; expected {self.config.source_layers}")
        if len(self.config.selected_layers) != self.config.target_layers:
            raise ValueError("mapper selected_layers does not match target layer count")
        source_config = getattr(source_model, "config")
        target_config = getattr(target_model, "config")
        checks = (
            (source_config.num_hidden_layers, self.config.source_layers, "source layers"),
            (target_config.num_hidden_layers, self.config.target_layers, "target layers"),
            (source_config.num_key_value_heads, self.config.source_kv_heads, "source KV heads"),
            (target_config.num_key_value_heads, self.config.target_kv_heads, "target KV heads"),
            (source_config.head_dim, self.config.source_head_dim, "source head dim"),
            (target_config.head_dim, self.config.target_head_dim, "target head dim"),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise ValueError(f"{label}: runtime={actual}, mapper={expected}")

    def _layer_on_device(self, layer: int, device: torch.device) -> dict[str, torch.Tensor]:
        cache_key = (layer, str(device))
        if cache_key not in self._device_layers:
            self._device_layers[cache_key] = {
                name: value.to(device=device) for name, value in self.load_layer(layer).items()
            }
        return self._device_layers[cache_key]


def _project(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Stored weight is [source_features, target_heads, target_head_dim].
    if weight.ndim != 3 or bias.shape != weight.shape[1:]:
        raise ValueError(f"invalid weight/bias shapes: {weight.shape}, {bias.shape}")
    output = x @ weight.to(dtype=x.dtype).flatten(1)
    output = output.view(x.shape[0], x.shape[1], weight.shape[1], weight.shape[2])
    return (output + bias.to(dtype=x.dtype)).permute(0, 2, 1, 3).contiguous()


def _legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    if isinstance(cache, (tuple, list)):
        return tuple(cache)
    raise TypeError(f"unsupported cache type: {type(cache)!r}")


def _model_input_device(model) -> torch.device:
    backbone = getattr(model, "model", model)
    return backbone.embed_tokens.weight.device


def _layer_device(layer) -> torch.device:
    return next(layer.parameters()).device


def _layer_dtype(layer) -> torch.dtype:
    return next(layer.parameters()).dtype
