"""BF16 affine projection with shared source blocks and FP32 rank scores."""

from collections import Counter

import torch
from transformers import DynamicCache
from xmodel_kv.rope import model_rope_cos_sin

from proposed.src.exp.mapper_optim.cache import prepare_key, prepare_value, unrotate_rows


class BF16RankMapper:
    """Keep the calibrated map and rank basis, changing only the online arithmetic."""

    def __init__(self, affine):
        if affine.head_local or affine.score_fn is None:
            raise ValueError("Expected the full-head RaReCache rank-disagreement artifact")
        self.affine = affine
        self.device = affine.device
        self.supports = {
            kind: [tuple(group) for group in affine.spec["selected_layers"][kind]]
            for kind in ("k", "v")
        }
        self.uses = {kind: Counter(groups) for kind, groups in self.supports.items()}
        for state in affine.layers:
            for kind in ("k", "v"):
                for name in ("weight", "bias"):
                    key = f"{kind}_0_{name}"
                    state[key] = state[key].to(dtype=torch.bfloat16)

    @torch.inference_mode()
    def map(self, source_model, target_model, source_cache, *, collect_scores):
        affine = self.affine
        source = unrotate_rows(source_model, source_cache)
        length = source[0][0].shape[0]
        shape = (length, affine.source.heads, affine.source.width)
        if len(source) != affine.source.layers or any(
                row.shape != shape for layer in source for row in layer):
            raise ValueError("Source KV shape differs from the RaReCache artifact")
        blocks = [(k.reshape(length, -1).to(torch.bfloat16), v.reshape(length, -1))
                  for k, v in source]
        del source
        pos = torch.arange(length, device=self.device)
        cos, sin = model_rope_cos_sin(target_model, pos, device=self.device, dtype=torch.float32)
        dtype = target_model.model.embed_tokens.weight.dtype
        if dtype != torch.bfloat16:
            raise ValueError("The optimized RaReCache path expects BF16 models and KV")
        score = torch.zeros(length, device=self.device, dtype=torch.float32) if collect_scores else None
        remaining = {kind: counts.copy() for kind, counts in self.uses.items()}
        shared = {"k": {}, "v": {}}
        layers = []
        for index, state in enumerate(affine.layers):
            outputs = []
            for side, kind in enumerate(("k", "v")):
                support = self.supports[kind][index]
                features = shared[kind].get(support)
                remaining[kind][support] -= 1
                if features is None:
                    features = torch.cat([blocks[source_layer][side] for source_layer in support],
                                         dim=-1)
                    if remaining[kind][support]:
                        shared[kind][support] = features
                elif not remaining[kind][support]:
                    del shared[kind][support]
                name = f"{kind}_0"
                mapped = torch.addmm(state[f"{name}_bias"], features, state[f"{name}_weight"])
                if score is not None:
                    score.add_(affine.score_fn(mapped.float(), state[f"{name}_mean"],
                                               state[f"{name}_basis"]))
                mapped = mapped.reshape(length, affine.target.heads, affine.target.width)
                outputs.append(prepare_key(mapped.float(), cos, sin, dtype) if kind == "k"
                               else prepare_value(mapped, dtype))
            layers.append(tuple(outputs))
        return DynamicCache.from_legacy_cache(tuple(layers)), score
