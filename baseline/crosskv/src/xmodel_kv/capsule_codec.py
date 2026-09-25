from __future__ import annotations

import struct
import math
import zlib

import torch


MAGIC = b"XKVCAP01"
VERSION = 1
_HEADER = struct.Struct("<8s7H")
_CHECKSUM = struct.Struct("<I")
_FLAG_ROPE_POLAR_KEY = 1
_FLAG_CARTESIAN_INT8_KEY = 2
_KEY_CODECS = {
    "cartesian": 0,
    "rope_polar": _FLAG_ROPE_POLAR_KEY,
    "cartesian_k8": _FLAG_CARTESIAN_INT8_KEY,
}
_KEY_CODECS_BY_FLAG = {flag: name for name, flag in _KEY_CODECS.items()}
_POLAR_MAGNITUDE_BITS = 3
_POLAR_PHASE_BITS = 5


def pack_int4_capsule(
    cache,
    *,
    suffix_tokens: int,
    active_layers: frozenset[int],
    key_codec: str = "cartesian",
) -> bytes:
    """Serialize a compact native KV suffix with four bits/element and FP16 scales.

    Cartesian tensors use symmetric signed INT4 with one scale per
    layer/head/token vector.  ``rope_polar`` Keys instead use one byte per
    rotate-half pair (3-bit radius and 5-bit phase), preserving exactly the same
    byte count. ``cartesian_k8`` uses signed INT8 Keys. Values always retain
    signed INT4. Only active layers are carried on the wire.
    """

    layers = _validate_cache(cache, suffix_tokens=suffix_tokens)
    if key_codec not in _KEY_CODECS:
        raise ValueError(
            "key_codec must be cartesian, rope_polar, or cartesian_k8"
        )
    total_layers = len(layers)
    indices = tuple(sorted(int(index) for index in active_layers))
    if not indices or indices[0] < 0 or indices[-1] >= total_layers:
        raise ValueError("active_layers must be a non-empty in-range subset")
    key0 = layers[0][0]
    heads, head_dim = int(key0.shape[1]), int(key0.shape[3])
    if key_codec == "rope_polar" and head_dim % 2:
        raise ValueError("rope_polar Key codec requires an even head dimension")
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        total_layers,
        len(indices),
        heads,
        suffix_tokens,
        head_dim,
        _KEY_CODECS[key_codec],
    ) + struct.pack(f"<{len(indices)}H", *indices)
    body = bytearray()
    for layer_index in indices:
        for tensor_index, tensor in enumerate(layers[layer_index]):
            suffix = tensor[:, :, -suffix_tokens:, :].detach().float().cpu()
            if tensor_index == 0 and key_codec == "rope_polar":
                body.extend(_pack_rope_polar_key(suffix))
                continue
            qmax = 127.0 if tensor_index == 0 and key_codec == "cartesian_k8" else 7.0
            scale = suffix.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
            # The transmitted FP16 scale, rather than an untransmitted FP32
            # value, defines the actual reconstruction.
            scale_half = scale.to(torch.float16)
            effective_scale = scale_half.float()
            quantized = (suffix / effective_scale).round().clamp(-qmax, qmax)
            body.extend(scale_half.squeeze(0).squeeze(-1).contiguous().numpy().tobytes())
            quantized = quantized.to(torch.int8)
            if tensor_index == 0 and key_codec == "cartesian_k8":
                body.extend(quantized.contiguous().numpy().tobytes())
            else:
                body.extend(_pack_nibbles(quantized))
    packet_without_checksum = header + bytes(body)
    checksum = zlib.crc32(packet_without_checksum)
    return packet_without_checksum + _CHECKSUM.pack(checksum)


def unpack_int4_capsule(
    packet: bytes,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
):
    """Validate and reconstruct a full-layer compact cache from a packet."""

    if len(packet) < _HEADER.size + _CHECKSUM.size:
        raise ValueError("capsule packet is truncated")
    payload, checksum_bytes = packet[:-_CHECKSUM.size], packet[-_CHECKSUM.size :]
    expected_checksum = _CHECKSUM.unpack(checksum_bytes)[0]
    if zlib.crc32(payload) != expected_checksum:
        raise ValueError("capsule packet checksum mismatch")
    (
        magic,
        version,
        total_layers,
        active_count,
        heads,
        tokens,
        head_dim,
        flags,
    ) = _HEADER.unpack_from(payload)
    if magic != MAGIC or version != VERSION or flags not in _KEY_CODECS_BY_FLAG:
        raise ValueError("unsupported capsule packet header")
    key_codec = _KEY_CODECS_BY_FLAG[flags]
    if key_codec == "rope_polar" and head_dim % 2:
        raise ValueError("rope_polar packet has an odd head dimension")
    if min(total_layers, active_count, heads, tokens, head_dim) < 1:
        raise ValueError("capsule packet dimensions must be positive")
    index_bytes = active_count * 2
    body_start = _HEADER.size + index_bytes
    if len(payload) < body_start:
        raise ValueError("capsule packet layer index is truncated")
    indices = struct.unpack_from(f"<{active_count}H", payload, _HEADER.size)
    if tuple(sorted(set(indices))) != indices or indices[-1] >= total_layers:
        raise ValueError("capsule packet layer indices are invalid")

    vector_count = heads * tokens
    element_count = vector_count * head_dim
    scale_bytes = vector_count * 2
    quantized_bytes = (element_count + 1) // 2
    key_bytes = element_count if key_codec == "cartesian_k8" else quantized_bytes
    expected_body = active_count * (
        2 * scale_bytes + key_bytes + quantized_bytes
    )
    if len(payload) - body_start != expected_body:
        raise ValueError("capsule packet body length disagrees with its header")

    output = [None] * total_layers
    cursor = body_start
    for layer_index in indices:
        tensors = []
        for tensor_index in range(2):
            scale_buffer = bytearray(payload[cursor : cursor + scale_bytes])
            cursor += scale_bytes
            scales = torch.frombuffer(scale_buffer, dtype=torch.float16).clone()
            scales = scales.reshape(1, heads, tokens, 1).float()
            encoded_bytes = (
                element_count
                if tensor_index == 0 and key_codec == "cartesian_k8"
                else quantized_bytes
            )
            encoded = payload[cursor : cursor + encoded_bytes]
            cursor += encoded_bytes
            if tensor_index == 0 and key_codec == "rope_polar":
                tensors.append(
                    _unpack_rope_polar_key(
                        encoded,
                        scales=scales,
                        heads=heads,
                        tokens=tokens,
                        head_dim=head_dim,
                        dtype=dtype,
                        device=device,
                    )
                )
            elif tensor_index == 0 and key_codec == "cartesian_k8":
                quantized = torch.frombuffer(
                    bytearray(encoded), dtype=torch.int8
                ).clone()
                quantized = quantized.reshape(1, heads, tokens, head_dim).float()
                tensors.append((quantized * scales).to(dtype=dtype, device=device))
            else:
                quantized = _unpack_nibbles(encoded, elements=element_count)
                quantized = quantized.reshape(1, heads, tokens, head_dim).float()
                tensors.append((quantized * scales).to(dtype=dtype, device=device))
        output[layer_index] = tuple(tensors)
    zero = torch.zeros((1, heads, tokens, head_dim), dtype=dtype, device=device)
    return tuple(
        layer if layer is not None else (zero.clone(), zero.clone()) for layer in output
    )


def int4_capsule_packet_bytes(
    *,
    total_layers: int,
    active_layers: int,
    heads: int,
    tokens: int,
    head_dim: int,
    key_codec: str = "cartesian",
) -> int:
    if min(total_layers, active_layers, heads, tokens, head_dim) < 1:
        raise ValueError("packet dimensions must be positive")
    if active_layers > total_layers:
        raise ValueError("active_layers exceeds total_layers")
    if key_codec not in _KEY_CODECS:
        raise ValueError(
            "key_codec must be cartesian, rope_polar, or cartesian_k8"
        )
    if key_codec == "rope_polar" and head_dim % 2:
        raise ValueError("rope_polar Key codec requires an even head dimension")
    vectors = heads * tokens
    elements = vectors * head_dim
    scale_bytes = vectors * 2
    value_bytes = scale_bytes + (elements + 1) // 2
    key_bytes = scale_bytes + (
        elements if key_codec == "cartesian_k8" else (elements + 1) // 2
    )
    return (
        _HEADER.size
        + active_layers * 2
        + active_layers * (key_bytes + value_bytes)
        + _CHECKSUM.size
    )


def _validate_cache(cache, *, suffix_tokens: int):
    layers = tuple(cache)
    if not layers or suffix_tokens < 1:
        raise ValueError("cache and suffix_tokens must be non-empty")
    reference_shape = tuple(layers[0][0].shape)
    if len(reference_shape) != 4 or reference_shape[0] != 1:
        raise ValueError("capsule tensors must have shape [1, heads, tokens, channels]")
    if reference_shape[2] < suffix_tokens:
        raise ValueError("suffix_tokens exceeds cache length")
    for layer in layers:
        if len(layer) != 2 or any(tuple(tensor.shape) != reference_shape for tensor in layer):
            raise ValueError("all K/V tensors must share one shape")
        if any(not tensor.is_floating_point() for tensor in layer):
            raise ValueError("capsule K/V tensors must be floating point")
    return layers


def _pack_nibbles(values: torch.Tensor) -> bytes:
    flat = (values.reshape(-1).to(torch.int16) + 8).to(torch.uint8)
    if flat.numel() % 2:
        flat = torch.cat((flat, torch.zeros(1, dtype=torch.uint8)))
    packed = flat[0::2] | (flat[1::2] << 4)
    return packed.contiguous().numpy().tobytes()


def _unpack_nibbles(data: bytes, *, elements: int) -> torch.Tensor:
    packed = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
    values = torch.empty(packed.numel() * 2, dtype=torch.int8)
    values[0::2] = (packed & 0x0F).to(torch.int8) - 8
    values[1::2] = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    return values[:elements]


def _pack_rope_polar_key(key: torch.Tensor) -> bytes:
    """Encode each rotate-half pair with 3-bit radius and 5-bit phase."""

    first, second = key.chunk(2, dim=-1)
    magnitude = torch.sqrt(first.square() + second.square())
    qmax = float((1 << _POLAR_MAGNITUDE_BITS) - 1)
    scale = magnitude.amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    scale_half = scale.to(torch.float16)
    effective_scale = scale_half.float()
    magnitude_code = (
        (magnitude / effective_scale).round().clamp(0, qmax).to(torch.uint8)
    )
    phase = torch.atan2(second, first).remainder(2 * math.pi)
    phase_levels = 1 << _POLAR_PHASE_BITS
    phase_code = (
        (phase * (phase_levels / (2 * math.pi))).round().to(torch.int16)
        % phase_levels
    ).to(torch.uint8)
    code = magnitude_code | (phase_code << _POLAR_MAGNITUDE_BITS)
    return (
        scale_half.squeeze(0).squeeze(-1).contiguous().numpy().tobytes()
        + code.contiguous().numpy().tobytes()
    )


def _unpack_rope_polar_key(
    data: bytes,
    *,
    scales: torch.Tensor,
    heads: int,
    tokens: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    half = head_dim // 2
    code = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
    code = code.reshape(1, heads, tokens, half)
    magnitude_mask = (1 << _POLAR_MAGNITUDE_BITS) - 1
    magnitude = (code & magnitude_mask).float() * scales
    phase_code = (code >> _POLAR_MAGNITUDE_BITS).float()
    phase = phase_code * (2 * math.pi / (1 << _POLAR_PHASE_BITS))
    key = torch.cat((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
    return key.to(dtype=dtype, device=device)
