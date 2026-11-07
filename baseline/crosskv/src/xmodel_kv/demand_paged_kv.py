from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


@dataclass(frozen=True)
class DemandPagedKVConfig:
    """Configuration for the decode-time document-KV working set.

    ``budget_ratio`` is applied independently at every layer. Selection is at
    page granularity, while the physical page contains every KV head for that
    layer. The first prototype deliberately keeps K/V in their native dtype;
    it studies paging independently from quantization.
    """

    page_size: int = 64
    budget_ratio: float = 0.05
    min_pages: int = 1
    cache_capacity_ratio: float = 1.0
    pin_cpu_memory: bool = True

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError("page_size must be positive")
        if not 0 < self.budget_ratio <= 1:
            raise ValueError("budget_ratio must lie in (0, 1]")
        if self.min_pages < 1:
            raise ValueError("min_pages must be positive")
        if self.cache_capacity_ratio < 1:
            raise ValueError("cache_capacity_ratio must be at least one")


@dataclass(frozen=True)
class LayerFetch:
    step: int
    layer: int
    cold_tokens: int
    cold_pages: int
    selected_tokens: int
    selected_pages: int
    page_hits: int
    page_misses: int
    loaded_bytes: int
    attended_tokens: int


@dataclass(frozen=True)
class DemandPagedKVSummary:
    steps: int
    layers: int
    cold_tokens: int
    cold_kv_bytes: int
    resident_kv_bytes: int
    directory_bytes: int
    loaded_bytes: int
    full_scan_bytes: int
    selected_token_fraction: float
    page_hit_fraction: float
    mean_attended_tokens: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class LayerPageStore:
    """One layer of immutable document KV backed by CPU micro-pages."""

    def __init__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cold_start: int,
        cold_end: int,
        config: DemandPagedKVConfig,
        device: torch.device,
        cold_page_ranges: Sequence[tuple[int, int]] | None = None,
        mandatory_page_ids: Sequence[int] | None = None,
    ) -> None:
        if key.ndim != 4 or key.shape != value.shape or key.shape[0] != 1:
            raise ValueError("K/V must have equal [1, kv_heads, tokens, dim] shapes")
        if not 0 <= cold_start <= cold_end <= key.shape[-2]:
            raise ValueError("cold range lies outside the cached sequence")
        self.config = config
        self.device = device
        self.dtype = key.dtype
        self.kv_heads = key.shape[1]
        self.head_dim = key.shape[-1]
        self.cold_tokens = cold_end - cold_start

        resident_indices = torch.cat(
            (
                torch.arange(0, cold_start, device=key.device),
                torch.arange(cold_end, key.shape[-2], device=key.device),
            )
        )
        self.resident_key = key.index_select(2, resident_indices).to(device)
        self.resident_value = value.index_select(2, resident_indices).to(device)

        cold_key = key[:, :, cold_start:cold_end].detach()
        cold_value = value[:, :, cold_start:cold_end].detach()
        if cold_page_ranges is None:
            cold_page_ranges = tuple(
                (
                    start,
                    min(start + config.page_size, self.cold_tokens),
                )
                for start in range(0, self.cold_tokens, config.page_size)
            )
        _validate_page_ranges(cold_page_ranges, self.cold_tokens)
        self.page_ranges = tuple(cold_page_ranges)
        self.page_lengths: list[int] = []
        self.key_pages: list[torch.Tensor] = []
        self.value_pages: list[torch.Tensor] = []
        minima: list[torch.Tensor] = []
        maxima: list[torch.Tensor] = []
        for start, end in self.page_ranges:
            key_page = cold_key[:, :, start:end]
            value_page = cold_value[:, :, start:end]
            minima.append(key_page.float().amin(dim=2).squeeze(0))
            maxima.append(key_page.float().amax(dim=2).squeeze(0))
            self.key_pages.append(self._to_cpu_page(key_page))
            self.value_pages.append(self._to_cpu_page(value_page))
            self.page_lengths.append(end - start)

        if minima:
            # [kv_heads, pages, head_dim]. These small Quest-style directory
            # tensors are the only document-key representation kept on GPU.
            self.key_min = torch.stack(minima, dim=1).to(device)
            self.key_max = torch.stack(maxima, dim=1).to(device)
        else:
            self.key_min = torch.empty(
                self.kv_heads, 0, self.head_dim, dtype=torch.float32, device=device
            )
            self.key_max = self.key_min.clone()

        self.mandatory_page_ids = tuple(
            dict.fromkeys(int(index) for index in mandatory_page_ids or ())
        )
        if any(
            index < 0 or index >= self.cold_pages
            for index in self.mandatory_page_ids
        ):
            raise ValueError("mandatory page index is out of range")
        if len(self.mandatory_page_ids) > self._page_budget():
            raise ValueError("mandatory pages exceed the configured page budget")

        self.cached_pages: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_used: dict[int, int] = {}
        self.generated_keys: list[torch.Tensor] = []
        self.generated_values: list[torch.Tensor] = []

    @property
    def cold_pages(self) -> int:
        return len(self.page_lengths)

    @property
    def cold_kv_bytes(self) -> int:
        return sum(
            (key.numel() + value.numel()) * key.element_size()
            for key, value in zip(self.key_pages, self.value_pages, strict=True)
        )

    @property
    def resident_kv_bytes(self) -> int:
        return (
            self.resident_key.numel() + self.resident_value.numel()
        ) * self.resident_key.element_size()

    @property
    def directory_bytes(self) -> int:
        return (
            self.key_min.numel() * self.key_min.element_size()
            + self.key_max.numel() * self.key_max.element_size()
        )

    def attend(
        self,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        *,
        step: int,
        layer: int,
        scaling: float,
    ) -> tuple[torch.Tensor, LayerFetch]:
        """Attend to resident state plus query-selected document pages."""

        if (
            query.ndim != 4
            or query.shape[0] != 1
            or query.shape[-2] != 1
            or current_key.shape != current_value.shape
            or current_key.shape[0] != 1
            or current_key.shape[-2] != 1
            or current_key.shape[1] != self.kv_heads
            or current_key.shape[-1] != self.head_dim
            or query.shape[1] % self.kv_heads
        ):
            raise ValueError("demand-paged attention currently requires batch=qlen=1")

        page_ids = self.select_pages(query)
        selected_key, selected_value, hits, misses, loaded_bytes = self.load_pages(
            page_ids, step=step
        )
        parts_key = [self.resident_key]
        parts_value = [self.resident_value]
        if selected_key is not None:
            parts_key.append(selected_key)
            parts_value.append(selected_value)
        if self.generated_keys:
            parts_key.append(torch.cat(self.generated_keys, dim=2))
            parts_value.append(torch.cat(self.generated_values, dim=2))
        parts_key.append(current_key)
        parts_value.append(current_value)
        key = torch.cat(parts_key, dim=2)
        value = torch.cat(parts_value, dim=2)

        repeats = query.shape[1] // self.kv_heads
        key = _repeat_kv(key, repeats)
        value = _repeat_kv(value, repeats)
        output = nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            scale=scaling,
        )

        self.generated_keys.append(current_key.detach())
        self.generated_values.append(current_value.detach())
        selected_tokens = sum(self.page_lengths[index] for index in page_ids)
        fetch = LayerFetch(
            step=step,
            layer=layer,
            cold_tokens=self.cold_tokens,
            cold_pages=self.cold_pages,
            selected_tokens=selected_tokens,
            selected_pages=len(page_ids),
            page_hits=hits,
            page_misses=misses,
            loaded_bytes=loaded_bytes,
            attended_tokens=key.shape[-2],
        )
        return output, fetch

    def select_pages(self, query: torch.Tensor) -> list[int]:
        """Select pages using a sign-aware upper bound on query-key logits."""

        if self.cold_pages == 0:
            return []
        groups = query.shape[1] // self.kv_heads
        grouped = query[:, :, 0].float().reshape(
            1, self.kv_heads, groups, self.head_dim
        )
        positive = grouped.clamp_min(0).unsqueeze(3)
        negative = grouped.clamp_max(0).unsqueeze(3)
        maxima = self.key_max.unsqueeze(0).unsqueeze(2)
        minima = self.key_min.unsqueeze(0).unsqueeze(2)
        bounds = (positive * maxima + negative * minima).sum(dim=-1)
        # A physical page contains all KV heads. Use the most demanding query
        # head as its score, then load that page once for every head.
        scores = bounds.amax(dim=(0, 1, 2))
        page_budget = self._page_budget()
        remaining = page_budget - len(self.mandatory_page_ids)
        if remaining == 0:
            return list(self.mandatory_page_ids)
        if self.mandatory_page_ids:
            scores = scores.clone()
            scores[list(self.mandatory_page_ids)] = -torch.inf
        dynamic = torch.topk(scores, k=remaining, sorted=False).indices.tolist()
        return list(self.mandatory_page_ids) + dynamic

    def load_pages(
        self, page_ids: Sequence[int], *, step: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int, int]:
        """Materialize selected CPU pages in the bounded GPU working set."""

        if any(index < 0 or index >= self.cold_pages for index in page_ids):
            raise ValueError("selected page index is out of range")
        hits = 0
        misses = 0
        loaded_bytes = 0
        for index in page_ids:
            if index in self.cached_pages:
                hits += 1
            else:
                key = self.key_pages[index].to(self.device, non_blocking=True)
                value = self.value_pages[index].to(self.device, non_blocking=True)
                self.cached_pages[index] = (key, value)
                loaded_bytes += (key.numel() + value.numel()) * key.element_size()
                misses += 1
            self.last_used[index] = step

        capacity = max(
            len(page_ids),
            math.ceil(len(page_ids) * self.config.cache_capacity_ratio),
        )
        selected = set(page_ids)
        while len(self.cached_pages) > capacity:
            candidates = [index for index in self.cached_pages if index not in selected]
            if not candidates:
                break
            victim = min(candidates, key=lambda index: self.last_used[index])
            del self.cached_pages[victim]
            del self.last_used[victim]

        if not page_ids:
            return None, None, hits, misses, loaded_bytes
        ordered = sorted(page_ids)
        key = torch.cat([self.cached_pages[index][0] for index in ordered], dim=2)
        value = torch.cat([self.cached_pages[index][1] for index in ordered], dim=2)
        return key, value, hits, misses, loaded_bytes

    def _to_cpu_page(self, tensor: torch.Tensor) -> torch.Tensor:
        page = tensor.to("cpu").contiguous()
        if self.config.pin_cpu_memory and torch.cuda.is_available():
            page = page.pin_memory()
        return page

    def _page_budget(self) -> int:
        budget = max(
            self.config.min_pages,
            math.ceil(self.cold_pages * self.config.budget_ratio),
        )
        return min(budget, self.cold_pages)


class DemandPagedKVController:
    """Own document pages and route each decoder layer through its store."""

    def __init__(
        self,
        stores: Sequence[LayerPageStore],
        *,
        cold_start: int,
        cold_end: int,
    ) -> None:
        if not stores:
            raise ValueError("at least one layer store is required")
        self.stores = tuple(stores)
        self.cold_start = cold_start
        self.cold_end = cold_end
        self.step = 0
        self.fetches: list[LayerFetch] = []

    @classmethod
    def from_cache(
        cls,
        cache: Any,
        *,
        cold_start: int,
        cold_end: int,
        config: DemandPagedKVConfig,
        device: torch.device | str,
        cold_page_ranges: Sequence[tuple[int, int]] | None = None,
        mandatory_page_ids: Sequence[int] | None = None,
    ) -> DemandPagedKVController:
        """Build a controller from a Transformers Cache or legacy KV tuple."""

        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        layers = tuple(cache)
        stores = [
            LayerPageStore(
                key,
                value,
                cold_start=cold_start,
                cold_end=cold_end,
                config=config,
                device=torch.device(device),
                cold_page_ranges=cold_page_ranges,
                mandatory_page_ids=mandatory_page_ids,
            )
            for key, value in layers
        ]
        return cls(stores, cold_start=cold_start, cold_end=cold_end)

    def attach(self, model: nn.Module) -> None:
        layers = tuple(model.model.layers)
        if len(layers) != len(self.stores):
            raise ValueError("model and cached state have different layer counts")
        for layer, store in zip(layers, self.stores, strict=True):
            attention = layer.self_attn
            attention.__dict__["_demand_paged_store"] = store
            attention.__dict__["_demand_paged_controller"] = self

    def detach(self, model: nn.Module) -> None:
        for layer in model.model.layers:
            layer.self_attn.__dict__.pop("_demand_paged_store", None)
            layer.self_attn.__dict__.pop("_demand_paged_controller", None)

    def next_step(self) -> None:
        self.step += 1

    def summary(self) -> DemandPagedKVSummary:
        cold_kv_bytes = sum(store.cold_kv_bytes for store in self.stores)
        resident_kv_bytes = sum(store.resident_kv_bytes for store in self.stores)
        directory_bytes = sum(store.directory_bytes for store in self.stores)
        loaded_bytes = sum(fetch.loaded_bytes for fetch in self.fetches)
        selected = sum(fetch.selected_tokens for fetch in self.fetches)
        available = sum(fetch.cold_tokens for fetch in self.fetches)
        hits = sum(fetch.page_hits for fetch in self.fetches)
        accesses = hits + sum(fetch.page_misses for fetch in self.fetches)
        attended = sum(fetch.attended_tokens for fetch in self.fetches)
        calls = len(self.fetches)
        return DemandPagedKVSummary(
            steps=self.step,
            layers=len(self.stores),
            cold_tokens=self.cold_end - self.cold_start,
            cold_kv_bytes=cold_kv_bytes,
            resident_kv_bytes=resident_kv_bytes,
            directory_bytes=directory_bytes,
            loaded_bytes=loaded_bytes,
            full_scan_bytes=cold_kv_bytes * self.step,
            selected_token_fraction=selected / available if available else 0.0,
            page_hit_fraction=hits / accesses if accesses else 1.0,
            mean_attended_tokens=attended / calls if calls else 0.0,
        )


def register_demand_paged_attention() -> None:
    """Register the experimental attention implementation with Transformers."""

    ALL_ATTENTION_FUNCTIONS.register(
        "demand_paged", demand_paged_attention_forward
    )
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        "demand_paged", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    )


def demand_paged_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
):
    """Transformers attention interface backed by a query-selected KV store."""

    store = module.__dict__.get("_demand_paged_store")
    controller = module.__dict__.get("_demand_paged_controller")
    if store is None or controller is None:
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
    scale = module.scaling if scaling is None else scaling
    output, fetch = store.attend(
        query,
        key,
        value,
        step=controller.step,
        layer=module.layer_idx,
        scaling=scale,
    )
    controller.fetches.append(fetch)
    return output.transpose(1, 2).contiguous(), None


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, length, head_dim
    )
    return hidden_states.reshape(batch, kv_heads * repeats, length, head_dim)


def _validate_page_ranges(
    ranges: Sequence[tuple[int, int]], cold_tokens: int
) -> None:
    if cold_tokens == 0:
        if ranges:
            raise ValueError("an empty cold region cannot contain pages")
        return
    cursor = 0
    for start, end in ranges:
        if start != cursor or not start < end <= cold_tokens:
            raise ValueError(
                "cold page ranges must be contiguous and cover the cold region"
            )
        cursor = end
    if cursor != cold_tokens:
        raise ValueError("cold page ranges must cover every cold token")
