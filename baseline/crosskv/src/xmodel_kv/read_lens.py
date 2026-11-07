from __future__ import annotations

import math

import torch
from torch import nn
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class HeadwiseLowRank(nn.Module):
    def __init__(self, heads: int, head_dim: int, rank: int):
        super().__init__()
        if rank < 1 or rank > head_dim:
            raise ValueError("rank must lie in [1, head_dim]")
        self.down = nn.Parameter(torch.empty(heads, head_dim, rank))
        self.up = nn.Parameter(torch.zeros(heads, rank, head_dim))
        nn.init.normal_(self.down, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        value = value.to(dtype=self.down.dtype)
        hidden = torch.einsum("bhtd,hdr->bhtr", value, self.down)
        return torch.einsum("bhtr,hrd->bhtd", hidden, self.up).to(dtype=dtype)


class StaticReadLens(nn.Module):
    """A fixed per-layer lens for one source-to-target policy pair."""

    def __init__(self, heads: int, head_dim: int, rank: int):
        super().__init__()
        self.query_delta = HeadwiseLowRank(heads, head_dim, rank)
        self.output_delta = HeadwiseLowRank(heads, head_dim, rank)
        self.beta = nn.Parameter(torch.zeros(heads))


class StaticReadLensBank(nn.Module):
    def __init__(self, model, *, layers: tuple[int, ...], rank: int):
        super().__init__()
        self.layer_indices = tuple(layers)
        self.lenses = nn.ModuleDict()
        for layer_index in layers:
            attention = model.model.layers[layer_index].self_attn
            lens = StaticReadLens(
                model.config.num_attention_heads,
                model.config.head_dim,
                rank,
            )
            self.lenses[str(layer_index)] = lens
            # Keep the bank as the sole nn.Module owner. The attention interface
            # accesses the lens without registering a duplicate submodule.
            attention.__dict__["_static_read_lens"] = lens
            attention.__dict__["_read_lens_enabled"] = False
            attention.__dict__["_read_lens_stale_start"] = 0
            attention.__dict__["_read_lens_stale_end"] = 0

    def configure(self, model, *, stale_start: int, stale_end: int, enabled: bool) -> None:
        if not 0 <= stale_start < stale_end:
            raise ValueError("require 0 <= stale_start < stale_end")
        for layer_index in self.layer_indices:
            attention = model.model.layers[layer_index].self_attn
            attention.__dict__["_read_lens_stale_start"] = int(stale_start)
            attention.__dict__["_read_lens_stale_end"] = int(stale_end)
            attention.__dict__["_read_lens_enabled"] = bool(enabled)

    def disable(self, model) -> None:
        for layer_index in self.layer_indices:
            model.model.layers[layer_index].self_attn.__dict__["_read_lens_enabled"] = False


def register_read_lens_attention() -> None:
    ALL_ATTENTION_FUNCTIONS.register("static_read_lens", read_lens_attention_forward)
    # The read-lens implementation materializes the attention scores itself and
    # therefore cannot use SDPA's ``attention_mask=None, is_causal=True`` fast
    # path.  Reuse the eager mask builder so multi-token teacher forcing always
    # receives an explicit four-dimensional causal mask.
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        "static_read_lens", ALL_MASK_ATTENTION_FUNCTIONS["eager"]
    )


def read_lens_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
):
    lens = module.__dict__.get("_static_read_lens")
    enabled = module.__dict__.get("_read_lens_enabled", False)
    if lens is None or not enabled:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    key = _repeat_kv(key, module.num_key_value_groups)
    value = _repeat_kv(value, module.num_key_value_groups)
    stale_start = int(module.__dict__["_read_lens_stale_start"])
    stale_end = int(module.__dict__["_read_lens_stale_end"])
    if not 0 <= stale_start < stale_end <= key.shape[-2]:
        raise ValueError(
            f"invalid stale range [{stale_start}, {stale_end}) for KV length "
            f"{key.shape[-2]}"
        )
    scale = module.scaling if scaling is None else scaling
    query_stale = query + lens.query_delta(query)
    score_left = torch.matmul(
        query, key[:, :, :stale_start].transpose(2, 3)
    ).float()
    score_stale = torch.matmul(
        query_stale, key[:, :, stale_start:stale_end].transpose(2, 3)
    ).float()
    score_right = torch.matmul(
        query, key[:, :, stale_end:].transpose(2, 3)
    ).float()
    score_stale = score_stale + lens.beta.float().view(1, -1, 1, 1) / scale
    scores = torch.cat((score_left, score_stale, score_right), dim=-1) * scale
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        scores = scores + attention_mask
    weights = torch.softmax(scores, dim=-1).to(query.dtype)
    if dropout:
        weights = torch.dropout(weights, dropout, train=module.training)
    base_output = torch.matmul(weights, value)
    stale_output = torch.matmul(
        weights[:, :, :, stale_start:stale_end],
        value[:, :, stale_start:stale_end],
    )
    output = base_output + lens.output_delta(stale_output)
    return output.transpose(1, 2).contiguous(), weights


def lens_regularization(bank: StaticReadLensBank) -> torch.Tensor:
    parameters = tuple(bank.parameters())
    if not parameters:
        return torch.tensor(0.0)
    return sum(parameter.float().square().mean() for parameter in parameters)


def lens_norms(bank: StaticReadLensBank) -> dict[str, float]:
    with torch.no_grad():
        values = {}
        for layer, lens in bank.lenses.items():
            values[layer] = math.sqrt(
                sum(parameter.float().square().sum().item() for parameter in lens.parameters())
            )
        return values


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, length, head_dim
    )
    return hidden_states.reshape(batch, kv_heads * repeats, length, head_dim)
