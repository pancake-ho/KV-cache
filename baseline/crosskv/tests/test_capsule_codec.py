import pytest
import torch
import math

from xmodel_kv.capsule_codec import (
    int4_capsule_packet_bytes,
    pack_int4_capsule,
    unpack_int4_capsule,
)


def _cache():
    generator = torch.Generator().manual_seed(3)
    return tuple(
        (
            torch.randn(1, 2, 3, 4, generator=generator),
            torch.randn(1, 2, 3, 4, generator=generator),
        )
        for _ in range(4)
    )


def _expected(tensor):
    scale = tensor.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7
    scale = scale.to(torch.float16).float()
    return (tensor / scale).round().clamp(-7, 7) * scale


def _expected_k8(tensor):
    scale = tensor.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127
    scale = scale.to(torch.float16).float()
    return (tensor / scale).round().clamp(-127, 127) * scale


def _expected_polar(tensor):
    first, second = tensor.chunk(2, dim=-1)
    magnitude = torch.sqrt(first.square() + second.square())
    scale = (magnitude.amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7).half().float()
    magnitude = (magnitude / scale).round().clamp(0, 7) * scale
    phase_code = (
        (torch.atan2(second, first).remainder(2 * math.pi) * (32 / (2 * math.pi)))
        .round()
        .to(torch.int16)
        % 32
    )
    phase = phase_code.float() * (2 * math.pi / 32)
    return torch.cat((magnitude * phase.cos(), magnitude * phase.sin()), dim=-1)


def test_int4_capsule_round_trip_matches_wire_scale_quantization():
    cache = _cache()
    packet = pack_int4_capsule(
        cache, suffix_tokens=3, active_layers=frozenset({1, 3})
    )
    restored = unpack_int4_capsule(packet, dtype=torch.float32)

    assert len(packet) == int4_capsule_packet_bytes(
        total_layers=4, active_layers=2, heads=2, tokens=3, head_dim=4
    )
    for index, layer in enumerate(restored):
        if index in (1, 3):
            for actual, source in zip(layer, cache[index], strict=True):
                torch.testing.assert_close(actual, _expected(source), rtol=0, atol=0)
        else:
            assert all(torch.count_nonzero(tensor) == 0 for tensor in layer)


def test_int4_capsule_rejects_corruption():
    packet = bytearray(
        pack_int4_capsule(_cache(), suffix_tokens=3, active_layers=frozenset({1}))
    )
    packet[-5] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        unpack_int4_capsule(bytes(packet))


def test_rope_polar_key_round_trip_preserves_packet_size_and_value_codec():
    cache = _cache()
    packet = pack_int4_capsule(
        cache,
        suffix_tokens=3,
        active_layers=frozenset({1, 3}),
        key_codec="rope_polar",
    )
    restored = unpack_int4_capsule(packet, dtype=torch.float32)

    assert len(packet) == int4_capsule_packet_bytes(
        total_layers=4, active_layers=2, heads=2, tokens=3, head_dim=4
    )
    for index, (key, value) in enumerate(restored):
        if index in (1, 3):
            torch.testing.assert_close(
                key, _expected_polar(cache[index][0]), rtol=0, atol=1e-6
            )
            torch.testing.assert_close(
                value, _expected(cache[index][1]), rtol=0, atol=0
            )
        else:
            assert torch.count_nonzero(key) == 0
            assert torch.count_nonzero(value) == 0


def test_rope_polar_key_rejects_odd_head_dimension():
    cache = tuple(
        (torch.randn(1, 2, 3, 3), torch.randn(1, 2, 3, 3)) for _ in range(2)
    )
    with pytest.raises(ValueError, match="even head dimension"):
        pack_int4_capsule(
            cache,
            suffix_tokens=3,
            active_layers=frozenset({0}),
            key_codec="rope_polar",
        )


def test_cartesian_k8_key_round_trip_keeps_int4_values_and_reports_rate():
    cache = _cache()
    packet = pack_int4_capsule(
        cache,
        suffix_tokens=3,
        active_layers=frozenset({1, 3}),
        key_codec="cartesian_k8",
    )
    restored = unpack_int4_capsule(packet, dtype=torch.float32)

    assert len(packet) == int4_capsule_packet_bytes(
        total_layers=4,
        active_layers=2,
        heads=2,
        tokens=3,
        head_dim=4,
        key_codec="cartesian_k8",
    )
    for index, (key, value) in enumerate(restored):
        if index in (1, 3):
            torch.testing.assert_close(
                key, _expected_k8(cache[index][0]), rtol=0, atol=0
            )
            torch.testing.assert_close(
                value, _expected(cache[index][1]), rtol=0, atol=0
            )
            int4_key_error = (cache[index][0] - _expected(cache[index][0])).norm()
            k8_key_error = (cache[index][0] - key).norm()
            assert k8_key_error < int4_key_error
        else:
            assert torch.count_nonzero(key) == 0
            assert torch.count_nonzero(value) == 0


def test_llama_last16_four_slot_packet_has_small_framing_overhead():
    assert int4_capsule_packet_bytes(
        total_layers=32,
        active_layers=16,
        heads=8,
        tokens=4,
        head_dim=128,
    ) == 67_642
    assert int4_capsule_packet_bytes(
        total_layers=32,
        active_layers=16,
        heads=8,
        tokens=4,
        head_dim=128,
        key_codec="cartesian_k8",
    ) == 100_410
    assert int4_capsule_packet_bytes(
        total_layers=32,
        active_layers=1,
        heads=8,
        tokens=4,
        head_dim=128,
        key_codec="cartesian_k8",
    ) == 6_300
