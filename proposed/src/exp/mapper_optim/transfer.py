"""Load paper-specific affine artifacts and map caches at the requested token boundary."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import DynamicCache
from xmodel_kv.rope import model_rope_cos_sin

from .cache import prepare_key, prepare_value, unrotate_rows
from .inputs import ModelIdentity


class AffineMapper:
    def __init__(self, root: Path, method: str, device: torch.device, score_fn=None):
        self.spec = json.loads((root / "calibration.json").read_text())
        if (self.spec["method"] != method or self.spec["format"] != 1
                or (method == "rarecache") != (score_fn is not None)):
            raise ValueError(f"Artifact format/method mismatch for {method}")
        self.source = ModelIdentity(**self.spec["source"])
        self.target = ModelIdentity(**self.spec["target"])
        self.device = device
        self.head_local = self.spec["variant"] in {"head-local", "attn-repair"}
        self.score_fn = score_fn
        self.layers = [load_file(str(root / f"layer_{layer:03d}.safetensors"),
                                 device=str(device)) for layer in range(self.target.layers)]
        self.bytes = sum(t.numel() * t.element_size() for layer in self.layers for t in layer.values())

    @torch.inference_mode()
    def map(self, source_model, target_model, source_cache, *, collect_scores: bool):
        source = unrotate_rows(source_model, source_cache)
        length = source[0][0].shape[0]
        if len(source) != self.source.layers or any(
                row.shape != (length, self.source.heads, self.source.width)
                for layer in source for row in layer):
            raise ValueError("Source KV shape differs from the calibration artifact")
        if collect_scores and self.score_fn is None:
            raise ValueError("Artifact does not contain a rank basis")
        positions = torch.arange(length, device=self.device)
        cos, sin = model_rope_cos_sin(target_model, positions, device=self.device, dtype=torch.float32)
        dtype = target_model.model.embed_tokens.weight.dtype
        score = torch.zeros(length, device=self.device, dtype=torch.float32) if collect_scores else None
        layers = []
        for target_layer, state in enumerate(self.layers):
            outputs = []
            for kind_index, kind in enumerate(("k", "v")):
                support = self.spec["selected_layers"][kind][target_layer]
                blocks = [source[index][kind_index] for index in support]
                if self.head_local:
                    # [T, H, P]; one source head per target head.
                    features = torch.cat(blocks, dim=-1).float().transpose(0, 1)
                    weight = state[kind + "_weight"]
                    bias = state[kind + "_bias"]
                    mapped = (features @ weight + bias[:, None, :]).transpose(0, 1)
                else:
                    features = torch.cat([block.flatten(start_dim=1) for block in blocks], dim=-1).float()
                    prefix = kind + "_0"
                    mapped = features @ state[prefix + "_weight"] + state[prefix + "_bias"]
                    if score is not None:
                        score.add_(self.score_fn(mapped, state[prefix + "_mean"],
                                                 state[prefix + "_basis"]))
                    mapped = mapped.reshape(length, self.target.heads, self.target.width)
                outputs.append(prepare_key(mapped, cos, sin, dtype) if kind == "k"
                               else prepare_value(mapped, dtype))
            layers.append(tuple(outputs))
        return DynamicCache.from_legacy_cache(tuple(layers)), score
