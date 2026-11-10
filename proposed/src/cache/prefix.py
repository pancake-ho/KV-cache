"""Use the upstream R5 semantic index with separate Qwen KV ownership."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from ..serving.request import Request

if TYPE_CHECKING:
    from torch import Tensor

Layers = tuple[tuple["Tensor", "Tensor"], ...]


@dataclass(frozen=True)
class Entry:
    request: Request
    layers: Layers
    size_bytes: int
    adapter_sha256: str | None


@dataclass(frozen=True)
class Lookup:
    entry: Entry | None
    cluster_id: int
    candidates: int
    rejected: dict[str, int]
    semantic_encoding_ms: float
    cluster_assignment_ms: float
    candidate_lookup_ms: float


class PrefixCache:
    """One process, exact zero-based prefixes and byte-bounded LRU."""

    def __init__(self, index, capacity_bytes: int):
        if capacity_bytes < 1:
            raise ValueError("cache capacity must be positive")
        self.index = index
        self.capacity_bytes = capacity_bytes
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.bytes = 0

    def lookup(self, request: Request, *, adapter_sha256: str | None = None, transfer=None) -> Lookup:
        assignment = self.index.assign(request.text)
        start = perf_counter()
        candidates = self.index.discover(assignment["cluster_id"], request.token_ids)
        rejected: dict[str, int] = {}
        hit = None
        for candidate in candidates:
            key = candidate["metadata"]["entry_id"]
            entry = self.entries.get(key)
            if entry is None:
                raise RuntimeError("semantic index refers to an unowned cache entry")
            original = entry.request
            if original.share_group != request.share_group:
                reason = "different_share_group"
            elif (original.adapter_id != request.adapter_id
                  or entry.adapter_sha256 != adapter_sha256):
                reason = "adapter_mismatch"
            elif (original.model_id != request.model_id
                  and (transfer is None or not transfer.supports(original, request))):
                reason = "unsupported_model_pair"
            elif (len(original.token_ids) >= len(request.token_ids)
                  or original.token_ids != request.token_ids[:len(original.token_ids)]):
                reason = "exact_prefix_mismatch"
            else:
                self.entries.move_to_end(key)
                hit = entry
                break
            rejected[reason] = rejected.get(reason, 0) + 1
        return Lookup(hit, assignment["cluster_id"], len(candidates), rejected,
                      assignment["semantic_encoding_ms"], assignment["cluster_assignment_ms"],
                      (perf_counter() - start) * 1000)

    def snapshot(self, request: Request, layers: Layers, adapter_sha256: str | None = None) -> Entry | None:
        size = sum(t.numel() * t.element_size() for pair in layers for t in pair)
        if size > self.capacity_bytes or len(request.token_ids) < 3:
            return None
        # Freeze the prefill before the request's DynamicCache grows during decode.
        owned = tuple((k.detach().clone(), v.detach().clone()) for k, v in layers)
        return Entry(request, owned, size, adapter_sha256)

    def admit(self, entry: Entry, cluster_id: int) -> None:
        if not 0 < entry.size_bytes <= self.capacity_bytes:
            raise ValueError("invalid cache entry byte size")
        request = entry.request
        if (request.adapter_id is None) != (entry.adapter_sha256 is None):
            raise ValueError("adapter identity is required for a LoRA cache entry")
        key = request.request_id
        if key in self.entries:
            raise ValueError(f"duplicate cache entry: {key}")
        while self.bytes + entry.size_bytes > self.capacity_bytes:
            victim = next(iter(self.entries))
            self.index.invalidate(victim)
            self.bytes -= self.entries.pop(victim).size_bytes
        self.index.add(key, {
            "entry_id": key,
            "source_request_id": key,
            "completed": True,
            "cluster_id": cluster_id,
            "token_ids": list(request.token_ids),
        })
        self.entries[key] = entry
        self.bytes += entry.size_bytes
