"""CacheBridge v1 query-conditioned K/V sample weights (Equations 15–19)."""
from __future__ import annotations

import numpy as np
import torch
from xmodel_kv.rope import apply_rope


def boundaries(length: int, count: int, first: int) -> np.ndarray:
    if count < 1 or not 1 <= first < length:
        raise ValueError("Expected at least one boundary in [1, sequence_length)")
    return np.unique(np.geomspace(first, length - 1, num=count).round().astype(np.int64))


def raw_sensitivity(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    sampled_positions: torch.Tensor, scaling: float) -> torch.Tensor:
    """Return unit-mass K and V scores [2, sampled tokens, KV heads] for one boundary.

    K and V contain exactly the eligible history. Query heads sharing one KV head
    are contiguous in Transformers Qwen3 GQA.
    """
    heads, _, dim = key.shape
    q = query.float().reshape(heads, -1, dim)
    k, v = key.float(), value.float()
    weights = ((q @ k.transpose(-1, -2)) * scaling).softmax(dim=-1)
    output = weights @ v
    a2 = weights.index_select(-1, sampled_positions).square()
    v_sampled = v.index_select(-2, sampled_positions)
    # Compute squared distances without materializing [head,group,token,width].
    diff2 = (v_sampled.square().sum(-1).unsqueeze(1) + output.square().sum(-1).unsqueeze(-1)
             - 2 * (output @ v_sampled.transpose(-1, -2))).clamp_min_(0)
    r_k = (a2 * diff2 * (q.square().sum(-1, keepdim=True) / dim)).sum(dim=1)
    r_v = a2.sum(dim=1)
    r = torch.stack((r_k.T, r_v.T))
    mass = r.sum(dim=1, keepdim=True)
    return torch.where(mass > 0, r / mass.clamp_min(torch.finfo(r.dtype).tiny), 1 / r.shape[1])


class QueryCapture:
    """Capture only the actual receiver queries at selected prefix boundaries."""

    def __init__(self, model, steps: torch.Tensor):
        self.steps = steps
        self.ends = steps.tolist()
        self.queries = {}
        self.handles = [
            layer.self_attn.register_forward_pre_hook(self._capture, with_kwargs=True)
            for layer in model.model.layers
        ]

    def _capture(self, module, args, kwargs):
        hidden = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])
        cos, sin = kwargs["position_embeddings"]
        q = module.q_proj(hidden.index_select(1, self.steps))
        q = module.q_norm(q.reshape(1, len(self.steps), -1, module.head_dim)).transpose(1, 2)
        q = apply_rope(q, cos.index_select(1, self.steps), sin.index_select(1, self.steps))
        self.queries[module.layer_idx] = (q[0].transpose(0, 1), module.scaling)

    def weights(self, cache_layers: tuple, sampled_positions: torch.Tensor):
        counts = torch.searchsorted(sampled_positions, self.steps).tolist()
        try:
            for index, (key, value) in enumerate(cache_layers):
                queries, scaling = self.queries[index]
                scores = torch.zeros((2, len(sampled_positions), key.shape[1]),
                                     device=key.device, dtype=torch.float32)
                # Each normalized boundary contributes equal mass. The query at
                # position c sees prefix tokens strictly below c in this statistic.
                for i, (end, count) in enumerate(zip(self.ends, counts, strict=True)):
                    if count:
                        scores[:, :count] += raw_sensitivity(
                            queries[i], key[0, :, :end], value[0, :, :end],
                            sampled_positions[:count], scaling,
                        )
                yield scores.cpu().numpy()
        finally:
            self.queries.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def shrinkage(raw: np.ndarray, features: int) -> tuple[float, float]:
    """Return the normalization and maximum mixing alpha satisfying the ESS floor."""
    mean = float(np.mean(raw, dtype=np.float64))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Non-finite or empty attention-weight mass")
    second = float(np.einsum("i,i->", raw, raw, dtype=np.float64))
    cv2 = max(0., second / (len(raw) * mean * mean) - 1.)
    floor = min(len(raw), 2 * features)
    alpha = min(1., ((len(raw) / floor - 1.) / cv2) ** 0.5) if cv2 else 1.
    return mean, alpha
