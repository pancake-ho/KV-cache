"""Carry real prefix K/V and repair positions over the benchmark TCP link."""

from time import perf_counter

import torch
from transformers import DynamicCache

from ..src.cache.prefix import PrefixCache
from .metrics.record import elapsed_ms
from .rx.transport import decode_kv, decode_request
from .tx.transport import encode_kv, encode_request


@torch.inference_mode()
def move_kv(link, layers, *, from_ud, codec=None, positions=None):
    """Round-trip the agreed cache layout without creating a combined tensor copy."""
    started = perf_counter()
    chunks, tensor_layout, packed_layout = encode_kv(layers, codec, positions)
    received, metrics = link.exchange(chunks, from_ud=from_ud)
    del chunks
    layers, received_positions = decode_kv(
        received, tensor_layout, packed_layout, codec, positions is not None,
    )
    torch.cuda.synchronize(layers[0][0].device)
    metrics["total_ms"] = elapsed_ms(started)
    return layers, received_positions, metrics


class NetworkPrefixCache(PrefixCache):
    """An isolated one-source trace with a target request sent to the ES index."""

    def __init__(self, index, capacity_bytes, link, source_model, source_placement):
        super().__init__(index, capacity_bytes)
        self.link = link
        self.source_model = source_model
        self.source_placement = source_placement
        self.setup_upload = None
        self.query_metrics = {}

    def lookup(self, request, *, adapter_sha256=None, transfer=None):
        if request.model_id != self.source_model:
            self.query_metrics = {}
            packet = encode_request(request)
            (received,), self.query_metrics["lookup_uplink"] = self.link.exchange(
                [packet], from_ud=True,
            )
            request = decode_request(received, type(request))
        started = perf_counter()
        result = super().lookup(request, adapter_sha256=adapter_sha256, transfer=transfer)
        if request.model_id != self.source_model:
            self.query_metrics["lookup_ms"] = elapsed_ms(started)
        return result

    def snapshot(self, request, layers, adapter_sha256=None, *, metrics=None):
        if request.model_id == self.source_model and self.source_placement == "es":
            layers, _, self.setup_upload = move_kv(self.link, layers, from_ud=True)
        return super().snapshot(request, layers, adapter_sha256, metrics=metrics)


class NetworkRaReCache:
    """Reuse the existing route guard and repair on the received target KV."""

    def __init__(self, guard, mapper, link, codec, source_placement):
        self.guard = guard
        self.mapper = mapper
        self.link = link
        self.codec = codec
        self.source_placement = source_placement
        self.metrics = {}

    def supports(self, source, target):
        return self.guard.supports(source, target)

    def supports_adapter_route(self, source, target, *, source_sha256, target_sha256):
        supports = getattr(self.guard, "supports_adapter_route", None)
        return supports is not None and supports(
            source, target, source_sha256=source_sha256, target_sha256=target_sha256,
        )

    def map_cache(self, layers, *, source, target, source_model, target_model):
        self.metrics = {}
        device = layers[0][0].device
        if self.source_placement == "ud":
            layers, _, self.metrics["source_upload"] = move_kv(self.link, layers, from_ud=True)
        started = perf_counter()

        def deliver(mapped, positions):
            positions_on_wire = positions.tolist()
            self.metrics["mapping_and_selection_ms"] = elapsed_ms(started)
            received, received_positions, self.metrics["target_download"] = move_kv(
                self.link, mapped.to_legacy_cache(), from_ud=False,
                codec=self.codec, positions=positions_on_wire,
            )
            if received_positions != positions_on_wire:
                raise RuntimeError("Repair positions changed during TCP transfer")
            cache = DynamicCache.from_legacy_cache(received)
            indices = torch.tensor(received_positions, device=device, dtype=torch.long)
            return cache, indices

        result = self.mapper.map_prefix(
            layers, source.token_ids, source_model=source_model, target_model=target_model,
            transfer_prefix=deliver,
        )
        torch.cuda.synchronize(device)
        self.metrics["transfer_mapping_repair_ms"] = elapsed_ms(started)
        self.metrics["repair_and_rebuild_ms"] = (
            self.metrics["transfer_mapping_repair_ms"] - self.metrics["mapping_and_selection_ms"]
            - self.metrics["target_download"]["total_ms"]
        )
        return result
