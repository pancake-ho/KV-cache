"""RoPE-aware neural KV transfer. Preserves CrossKV's map_cache interface."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .model import HeadMLP
from .spec import MapperSpec, checkpoint_path, expected_checkpoints


class NeuralKVMapper:
    """Layer-streamed MLP weights for one directional matched-KV model pair.

    Model files are intentionally loaded one head at a time. This avoids silently
    exhausting the Seraph GPU while *not* claiming online speed advantages.
    """

    def __init__(self, root: str | Path, *, token_chunk: int = 128):
        if token_chunk < 1:
            raise ValueError("token_chunk must be positive")
        self.root = Path(root)
        self.config = MapperSpec.load(root)
        self.token_chunk = token_chunk

    def assert_complete(self) -> None:
        missing = [str(p.relative_to(self.root)) for p in expected_checkpoints(self.config, self.root) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete neural mapper: {len(missing)} missing; first={missing[0]}")

    @torch.inference_mode()
    def map_cache(self, source_cache: Any, *, source_model, target_model,
                  positions: torch.Tensor | None = None):
        # Baseline CrossKV provides the already-verified HF/Qwen RoPE convention.
        from transformers import DynamicCache
        from xmodel_kv.rope import apply_rope, model_rope_cos_sin, remove_rope

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: refusing CPU inference fallback")
        self.assert_complete()
        cache = (source_cache.to_legacy_cache() if hasattr(source_cache, "to_legacy_cache")
                 else source_cache)
        spec = self.config
        if len(cache) != spec.source_layers:
            raise ValueError(f"source cache layers={len(cache)}, expected {spec.source_layers}")
        for model, name, expected_layers, heads, dim in (
            (source_model, "source", spec.source_layers, spec.source_kv_heads, spec.source_head_dim),
            (target_model, "target", spec.target_layers, spec.target_kv_heads, spec.target_head_dim),
        ):
            cfg = model.config
            if (cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim) != (expected_layers, heads, dim):
                raise ValueError(f"{name} model architecture mismatch with mapper config")
        batch, heads, length, dim = cache[0][0].shape
        if (heads, dim) != (spec.source_kv_heads, spec.source_head_dim):
            raise ValueError("source cache KV head dimensions mismatch")
        for layer, (key, value) in enumerate(cache):
            if key.shape != (batch, heads, length, dim) or value.shape != key.shape:
                raise ValueError(f"inconsistent K/V cache shape at layer {layer}")
        if positions is None:
            positions = torch.arange(length, dtype=torch.long)
        if not ((positions.ndim == 1 and positions.shape[0] == length) or
                (positions.ndim == 2 and positions.shape == (batch, length))):
            raise ValueError("positions must be [tokens] or [batch,tokens]")
        sdev = source_model.model.embed_tokens.weight.device
        tdev = target_model.model.embed_tokens.weight.device
        cos_s, sin_s = model_rope_cos_sin(source_model, positions, device=sdev, dtype=torch.float32)
        cos_t, sin_t = model_rope_cos_sin(target_model, positions, device=tdev, dtype=torch.float32)
        target_backbone = target_model.model
        mapped_layers = []
        for target_layer, selected in enumerate(spec.selected_layers):
            layer_module = target_backbone.layers[target_layer]
            device = next(layer_module.parameters()).device
            if device.type != "cuda":
                raise RuntimeError(f"target layer {target_layer} not on GPU")
            source_cos = cos_s.to(device)
            source_sin = sin_s.to(device)
            output = []
            for kind in ("k", "v"):
                mapped = torch.empty((batch, spec.target_kv_heads, length, spec.target_head_dim),
                                     dtype=torch.float32, device=device)
                for head in range(spec.target_kv_heads):
                    network = self._load_network(target_layer, head, kind, device)
                    for begin in range(0, length, self.token_chunk):
                        end = min(begin+self.token_chunk, length)
                        parts = []
                        for source_layer in selected:
                            source_tensor = cache[source_layer][0 if kind == "k" else 1]
                            piece = source_tensor[:, :, begin:end, :].to(device=device, dtype=torch.float32)
                            if kind == "k":
                                sc = source_cos[:, begin:end] if source_cos.ndim == 3 else source_cos[begin:end]
                                ss = source_sin[:, begin:end] if source_sin.ndim == 3 else source_sin[begin:end]
                                piece = remove_rope(piece, sc, ss, sequence_dim=2)
                            parts.append(piece.permute(0, 2, 1, 3).reshape(batch*(end-begin), -1))
                        features = torch.cat(parts, dim=-1)
                        if features.shape[-1] != spec.input_dim:
                            raise ValueError("feature width mismatch")
                        predicted = network(features).reshape(batch, end-begin, spec.target_head_dim)
                        mapped[:, head, begin:end, :] = predicted
                    del network
                if kind == "k":
                    mapped = apply_rope(mapped, cos_t.to(device), sin_t.to(device), sequence_dim=2)
                mapped = mapped.to(dtype=next(layer_module.parameters()).dtype)
                output.append(mapped)
            mapped_layers.append((output[0], output[1]))
        return DynamicCache.from_legacy_cache(tuple(mapped_layers))

    def _load_network(self, layer: int, head: int, kind: str, device: torch.device) -> HeadMLP:
        path = checkpoint_path(self.root, layer, head, kind)
        network = HeadMLP(self.config.input_dim, self.config.target_head_dim,
                          self.config.hidden_dim).to(device)
        state = load_file(path, device="cpu")
        network.load_state_dict(state, strict=True)
        return network.eval()
