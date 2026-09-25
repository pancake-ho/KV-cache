from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .capsule_codec import pack_int4_capsule, unpack_int4_capsule


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class Int4DeltaComposition:
    """Decoded caches and the two independent packets used to compose them."""

    cache: LegacyCache
    base_cache: LegacyCache
    delta_cache: LegacyCache
    base_packet: bytes
    delta_packet: bytes

    @property
    def base_packet_bytes(self) -> int:
        return len(self.base_packet)

    @property
    def delta_packet_bytes(self) -> int:
        return len(self.delta_packet)

    @property
    def cold_packet_bytes(self) -> int:
        return self.base_packet_bytes + self.delta_packet_bytes

    @property
    def incremental_packet_bytes(self) -> int:
        return self.delta_packet_bytes


def compose_int4_base_delta(
    base_cache: Any,
    task_cache: Any,
    *,
    suffix_tokens: int,
    base_layers: frozenset[int],
    delta_layers: frozenset[int],
    key_codec: str = "cartesian",
) -> Int4DeltaComposition:
    """Independently encode a reusable base and an additive task sidecar.

    The sender computes ``task - base`` in FP32 only for ``delta_layers``.
    Base and delta are then independently encoded with the production INT4
    packet codec.  The receiver decodes both and adds the sidecar only at the
    declared layers.  Consequently, all non-delta layers in the returned cache
    are the exact decoded base tensors and base-only fallback uses the same
    packet bytes as composition.
    """

    base = _validate_compatible_caches(
        base_cache, task_cache, suffix_tokens=suffix_tokens
    )
    task = tuple(task_cache)
    total_layers = len(base)
    _validate_layer_sets(
        total_layers=total_layers,
        base_layers=base_layers,
        delta_layers=delta_layers,
    )

    base_packet = pack_int4_capsule(
        base,
        suffix_tokens=suffix_tokens,
        active_layers=base_layers,
        key_codec=key_codec,
    )
    delta_source = cache_layer_delta(
        base,
        task,
        delta_layers=delta_layers,
    )
    delta_packet = pack_int4_capsule(
        delta_source,
        suffix_tokens=suffix_tokens,
        active_layers=delta_layers,
        key_codec=key_codec,
    )

    reference = task[0][0]
    decoded_base = unpack_int4_capsule(
        base_packet,
        dtype=reference.dtype,
        device=reference.device,
    )
    decoded_delta = unpack_int4_capsule(
        delta_packet,
        dtype=reference.dtype,
        device=reference.device,
    )
    composed = tuple(
        tuple(
            base_tensor + delta_tensor
            if layer_index in delta_layers
            else base_tensor
            for base_tensor, delta_tensor in zip(
                base_layer, delta_layer, strict=True
            )
        )
        for layer_index, (base_layer, delta_layer) in enumerate(
            zip(decoded_base, decoded_delta, strict=True)
        )
    )
    return Int4DeltaComposition(
        cache=composed,
        base_cache=decoded_base,
        delta_cache=decoded_delta,
        base_packet=base_packet,
        delta_packet=delta_packet,
    )


def cache_layer_delta(
    base_cache: Any,
    task_cache: Any,
    *,
    delta_layers: frozenset[int],
) -> LegacyCache:
    """Return an FP32 cache containing ``task - base`` on selected layers."""

    base = tuple(base_cache)
    task = tuple(task_cache)
    if len(base) != len(task) or not base:
        raise ValueError("base and task caches must have the same non-zero depth")
    if not delta_layers or min(delta_layers) < 0 or max(delta_layers) >= len(base):
        raise ValueError("delta_layers must be a non-empty in-range subset")
    result = []
    for layer_index, (base_layer, task_layer) in enumerate(
        zip(base, task, strict=True)
    ):
        if len(base_layer) != 2 or len(task_layer) != 2:
            raise ValueError("every cache layer must contain one key and one value")
        tensors = []
        for base_tensor, task_tensor in zip(base_layer, task_layer, strict=True):
            if tuple(base_tensor.shape) != tuple(task_tensor.shape):
                raise ValueError("base and task cache tensors must have equal shapes")
            if layer_index in delta_layers:
                tensors.append(task_tensor.float() - base_tensor.float())
            else:
                tensors.append(torch.zeros_like(task_tensor, dtype=torch.float32))
        result.append(tuple(tensors))
    return tuple(result)


def _validate_compatible_caches(
    base_cache: Any,
    task_cache: Any,
    *,
    suffix_tokens: int,
) -> LegacyCache:
    base = tuple(base_cache)
    task = tuple(task_cache)
    if not base or len(base) != len(task):
        raise ValueError("base and task caches must have the same non-zero depth")
    if suffix_tokens < 1:
        raise ValueError("suffix_tokens must be positive")
    reference_shape = tuple(base[0][0].shape)
    if (
        len(reference_shape) != 4
        or reference_shape[0] != 1
        or reference_shape[-2] < suffix_tokens
    ):
        raise ValueError("cache tensors must have shape [1, heads, tokens, channels]")
    for base_layer, task_layer in zip(base, task, strict=True):
        if len(base_layer) != 2 or len(task_layer) != 2:
            raise ValueError("every cache layer must contain one key and one value")
        for base_tensor, task_tensor in zip(base_layer, task_layer, strict=True):
            if (
                tuple(base_tensor.shape) != reference_shape
                or tuple(task_tensor.shape) != reference_shape
            ):
                raise ValueError("all base and task tensors must share one shape")
            if not base_tensor.is_floating_point() or not task_tensor.is_floating_point():
                raise ValueError("base and task cache tensors must be floating point")
    return base


def _validate_layer_sets(
    *,
    total_layers: int,
    base_layers: frozenset[int],
    delta_layers: frozenset[int],
) -> None:
    if not base_layers or min(base_layers) < 0 or max(base_layers) >= total_layers:
        raise ValueError("base_layers must be a non-empty in-range subset")
    if not delta_layers or min(delta_layers) < 0 or max(delta_layers) >= total_layers:
        raise ValueError("delta_layers must be a non-empty in-range subset")
    if not delta_layers.issubset(base_layers):
        raise ValueError("delta_layers must be a subset of base_layers")
