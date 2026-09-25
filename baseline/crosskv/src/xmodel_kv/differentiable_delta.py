from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class DifferentiableInt4Composition:
    """Production-equivalent forward caches with an STE task-delta path."""

    cache: LegacyCache
    base_cache: LegacyCache
    delta_cache: LegacyCache


def fake_int4_cache_round_trip(
    cache: Any,
    *,
    suffix_tokens: int,
    active_layers: frozenset[int],
    straight_through: bool,
) -> LegacyCache:
    """Production-equivalent Cartesian K4/V4 cache with an optional STE.

    This is the monolithic-cache counterpart of
    :func:`compose_fake_int4_base_delta`: it uses transmitted FP16 scales and
    signed INT4 codes exactly like ``pack_int4_capsule``/``unpack_int4_capsule``.
    Only the suffix is returned; inactive layers are explicit zero tensors.
    """

    layers = tuple(cache)
    if not layers or suffix_tokens < 1:
        raise ValueError("cache and suffix_tokens must be non-empty")
    if (
        not active_layers
        or min(active_layers) < 0
        or max(active_layers) >= len(layers)
    ):
        raise ValueError("active_layers must be a non-empty in-range subset")
    reference_shape = tuple(layers[0][0].shape)
    if (
        len(reference_shape) != 4
        or reference_shape[0] != 1
        or reference_shape[-2] < suffix_tokens
    ):
        raise ValueError("cache tensors must have shape [1, heads, tokens, channels]")
    wire_dtype = layers[0][0].dtype
    output = []
    for layer_index, layer in enumerate(layers):
        if len(layer) != 2:
            raise ValueError("every cache layer must contain one key and one value")
        tensors = []
        for tensor in layer:
            if tuple(tensor.shape) != reference_shape or not tensor.is_floating_point():
                raise ValueError("all cache tensors must share one floating shape")
            suffix = tensor[:, :, -suffix_tokens:]
            if layer_index in active_layers:
                decoded = fake_int4_wire_round_trip(
                    suffix,
                    wire_dtype=wire_dtype,
                    straight_through=straight_through,
                )
            else:
                decoded = torch.zeros_like(suffix)
            tensors.append(decoded)
        output.append(tuple(tensors))
    return tuple(output)


def compose_fake_int4_base_delta(
    base_cache: Any,
    task_cache: Any,
    *,
    suffix_tokens: int,
    base_layers: frozenset[int],
    delta_layers: frozenset[int],
    straight_through_delta: bool = True,
) -> DifferentiableInt4Composition:
    """Match Cartesian K4/V4 wire decoding while retaining delta gradients.

    The forward uses the same FP32 max-absolute scale computation, transmitted
    FP16 scale, signed INT4 range, and decoded dtype as the production packet.
    In the backward pass, only selected ``task - base`` tensors use an identity
    straight-through estimator. Base tensors are detached and non-delta task
    tensors do not participate in the output graph.
    """

    base, task = _validate_caches(
        base_cache,
        task_cache,
        suffix_tokens=suffix_tokens,
        base_layers=base_layers,
        delta_layers=delta_layers,
    )
    wire_dtype = base[0][0].dtype
    decoded_base = []
    decoded_delta = []
    composed = []
    for layer_index, (base_layer, task_layer) in enumerate(
        zip(base, task, strict=True)
    ):
        base_tensors = []
        delta_tensors = []
        composed_tensors = []
        for base_tensor, task_tensor in zip(base_layer, task_layer, strict=True):
            base_suffix = base_tensor[:, :, -suffix_tokens:].detach()
            shape = tuple(base_suffix.shape)
            if layer_index in base_layers:
                base_decoded = fake_int4_wire_round_trip(
                    base_suffix,
                    wire_dtype=wire_dtype,
                    straight_through=False,
                )
            else:
                base_decoded = torch.zeros(
                    shape, dtype=wire_dtype, device=base_tensor.device
                )
            if layer_index in delta_layers:
                delta = (
                    task_tensor[:, :, -suffix_tokens:].float()
                    - base_tensor[:, :, -suffix_tokens:].detach().float()
                )
                delta_decoded = fake_int4_wire_round_trip(
                    delta,
                    wire_dtype=wire_dtype,
                    straight_through=straight_through_delta,
                )
                combined = base_decoded + delta_decoded
            else:
                delta_decoded = torch.zeros(
                    shape, dtype=wire_dtype, device=base_tensor.device
                )
                combined = base_decoded
            base_tensors.append(base_decoded)
            delta_tensors.append(delta_decoded)
            composed_tensors.append(combined)
        decoded_base.append(tuple(base_tensors))
        decoded_delta.append(tuple(delta_tensors))
        composed.append(tuple(composed_tensors))
    return DifferentiableInt4Composition(
        cache=tuple(composed),
        base_cache=tuple(decoded_base),
        delta_cache=tuple(decoded_delta),
    )


def fake_int4_wire_round_trip(
    tensor: torch.Tensor,
    *,
    wire_dtype: torch.dtype,
    straight_through: bool,
) -> torch.Tensor:
    """Fake the production signed-INT4/FP16-scale forward exactly."""

    if not tensor.is_floating_point() or tensor.ndim != 4:
        raise ValueError("tensor must be floating point with four dimensions")
    source = tensor.float()
    scale = source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    transmitted_scale = scale.to(torch.float16).float()
    # The integer cast is semantically important for an all-zero vector: its
    # FP16 scale underflows to zero, so the intermediate division is NaN, while
    # the production NaN-to-int8 cast yields code zero and decodes to zero.
    codes = (
        (source / transmitted_scale).round().clamp(-7, 7).to(torch.int8).float()
    )
    quantized = (codes * transmitted_scale).to(wire_dtype)
    if not straight_through:
        return quantized
    # Preserve the exact decoded forward while treating quantization as the
    # identity in backward. The final cast also matches the packet decoder.
    return (source + (quantized.float() - source).detach()).to(wire_dtype)


def _validate_caches(
    base_cache: Any,
    task_cache: Any,
    *,
    suffix_tokens: int,
    base_layers: frozenset[int],
    delta_layers: frozenset[int],
) -> tuple[LegacyCache, LegacyCache]:
    base = tuple(base_cache)
    task = tuple(task_cache)
    if not base or len(base) != len(task):
        raise ValueError("base and task caches must have the same non-zero depth")
    if suffix_tokens < 1:
        raise ValueError("suffix_tokens must be positive")
    if not base_layers or min(base_layers) < 0 or max(base_layers) >= len(base):
        raise ValueError("base_layers must be a non-empty in-range subset")
    if not delta_layers or min(delta_layers) < 0 or max(delta_layers) >= len(base):
        raise ValueError("delta_layers must be a non-empty in-range subset")
    if not delta_layers.issubset(base_layers):
        raise ValueError("delta_layers must be a subset of base_layers")
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
            if base_tensor.device != task_tensor.device:
                raise ValueError("base and task tensors must share one device")
    return base, task
