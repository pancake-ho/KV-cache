"""Serve the fitted CacheBridge or RaReCache projection on a directed model pair."""

from dataclasses import replace
from pathlib import Path

import torch
from transformers import DynamicCache

from ..exp.mapper_optim.inputs import ModelIdentity
from ..exp.mapper_optim.transfer import AffineMapper
from ..exp.rarecache.rank import disagreement, select_positions
from ..exp.rarecache.recompute import recompute
from .mapper_bf16 import BF16RankMapper


class AffineKVMapper:
    """Bind a fitted artifact to the local revisions and wrap request-local replay."""

    def __init__(self, artifact, method, identities, snapshots, device, *, repair_fraction=0.30):
        if method not in {"rarecache", "cachebridge", "cachebridge_v2"}:
            raise ValueError(f"Unknown affine mapper {method}")
        mapper = AffineMapper(Path(artifact), method, device,
                              disagreement if method == "rarecache" else None)
        for actual, expected, path in zip((mapper.source, mapper.target), identities, snapshots, strict=True):
            if (actual.name not in {expected.name, str(path)}
                    or replace(actual, name=expected.name) != expected):
                raise ValueError(f"Fitted {method} artifact does not match {expected.name} at {expected.revision}")
        if mapper.spec["dtype"] != "bfloat16":
            raise ValueError("Serving requires a BF16 KV fitting artifact")
        if method == "cachebridge" and mapper.spec["variant"] != "attn-repair":
            raise ValueError("Expected the verified attention-weighted CacheBridge artifact")
        if method == "cachebridge_v2" and mapper.spec["variant"] != "layer-ordered-readout":
            raise ValueError("Expected the layer-ordered CacheBridge v2 artifact")
        if not 0 < repair_fraction <= 1:
            raise ValueError("repair_fraction must be in (0, 1]")
        self.pair = tuple(identity.name for identity in identities)
        self.metadata = mapper.spec
        self.mapper = BF16RankMapper(mapper) if method == "rarecache" else mapper
        self.repair_fraction = repair_fraction if method == "rarecache" else 0

    @torch.inference_mode()
    def map_prefix(self, layers, prefix_ids, *, source_model, target_model):
        cache = DynamicCache.from_legacy_cache(layers)
        mapped, scores = self.mapper.map(source_model, target_model, cache,
                                         collect_scores=bool(self.repair_fraction))
        if self.repair_fraction:
            positions = select_positions(scores, self.repair_fraction)
            prefix = torch.tensor([prefix_ids], dtype=torch.long, device=positions.device)
            mapped = recompute(target_model, mapped, prefix, positions)
        return mapped
