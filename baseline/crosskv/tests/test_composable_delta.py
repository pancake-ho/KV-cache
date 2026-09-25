import pytest
import torch

from xmodel_kv.capsule_codec import pack_int4_capsule, unpack_int4_capsule
from xmodel_kv.composable_delta import (
    cache_layer_delta,
    compose_int4_base_delta,
)


def _cache(*, seed: int, layers: int = 4):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        (
            torch.randn(1, 2, 3, 4, generator=generator),
            torch.randn(1, 2, 3, 4, generator=generator),
        )
        for _ in range(layers)
    )


def test_composition_uses_exact_base_packet_and_changes_only_delta_layer():
    base = _cache(seed=4)
    task = tuple(
        tuple(
            tensor + (0.2 * (layer_index + 1) if layer_index == 2 else 0.0)
            for tensor in layer
        )
        for layer_index, layer in enumerate(base)
    )
    base_layers = frozenset({0, 1, 2})
    delta_layers = frozenset({2})

    composition = compose_int4_base_delta(
        base,
        task,
        suffix_tokens=3,
        base_layers=base_layers,
        delta_layers=delta_layers,
    )
    independently_packed_base = pack_int4_capsule(
        base, suffix_tokens=3, active_layers=base_layers
    )
    independently_decoded_base = unpack_int4_capsule(
        independently_packed_base, dtype=torch.float32
    )

    assert composition.base_packet == independently_packed_base
    for layer_index, (composed_layer, base_layer) in enumerate(
        zip(composition.cache, independently_decoded_base, strict=True)
    ):
        for composed_tensor, base_tensor in zip(
            composed_layer, base_layer, strict=True
        ):
            if layer_index == 2:
                assert not torch.equal(composed_tensor, base_tensor)
            else:
                assert torch.equal(composed_tensor, base_tensor)
    assert all(
        torch.count_nonzero(tensor) == 0 for tensor in composition.cache[3]
    )


def test_zero_delta_composes_to_decoded_base_bitwise():
    base = _cache(seed=5)
    composition = compose_int4_base_delta(
        base,
        base,
        suffix_tokens=3,
        base_layers=frozenset({0, 1, 2}),
        delta_layers=frozenset({2}),
    )

    for composed_layer, base_layer in zip(
        composition.cache, composition.base_cache, strict=True
    ):
        for composed_tensor, base_tensor in zip(
            composed_layer, base_layer, strict=True
        ):
            assert torch.equal(composed_tensor, base_tensor)


def test_delta_is_computed_in_fp32_only_on_selected_layers():
    base = tuple(
        (torch.full((1, 1, 2, 2), 1.0), torch.full((1, 1, 2, 2), 2.0))
        for _ in range(3)
    )
    task = tuple(
        (key + 0.25, value - 0.5) for key, value in base
    )

    delta = cache_layer_delta(base, task, delta_layers=frozenset({1}))

    assert all(tensor.dtype == torch.float32 for layer in delta for tensor in layer)
    assert all(torch.count_nonzero(tensor) == 0 for tensor in delta[0])
    assert all(torch.count_nonzero(tensor) == 0 for tensor in delta[2])
    torch.testing.assert_close(delta[1][0], torch.full_like(delta[1][0], 0.25))
    torch.testing.assert_close(delta[1][1], torch.full_like(delta[1][1], -0.5))


def test_composition_reports_actual_independent_packet_sizes():
    base = _cache(seed=6, layers=32)
    task = _cache(seed=7, layers=32)

    composition = compose_int4_base_delta(
        base,
        task,
        suffix_tokens=3,
        base_layers=frozenset(range(16)),
        delta_layers=frozenset({15}),
    )

    assert composition.base_packet_bytes == len(composition.base_packet)
    assert composition.delta_packet_bytes == len(composition.delta_packet)
    assert composition.cold_packet_bytes == (
        composition.base_packet_bytes + composition.delta_packet_bytes
    )
    assert composition.incremental_packet_bytes == composition.delta_packet_bytes

    polar = compose_int4_base_delta(
        base,
        task,
        suffix_tokens=3,
        base_layers=frozenset(range(16)),
        delta_layers=frozenset({15}),
        key_codec="rope_polar",
    )
    assert polar.base_packet_bytes == composition.base_packet_bytes
    assert polar.delta_packet_bytes == composition.delta_packet_bytes

    k8v4 = compose_int4_base_delta(
        base,
        task,
        suffix_tokens=3,
        base_layers=frozenset(range(16)),
        delta_layers=frozenset({15}),
        key_codec="cartesian_k8",
    )
    assert k8v4.base_packet_bytes > composition.base_packet_bytes
    assert k8v4.delta_packet_bytes > composition.delta_packet_bytes
    for layer_index, (composed_layer, base_layer) in enumerate(
        zip(k8v4.cache, k8v4.base_cache, strict=True)
    ):
        if layer_index == 15:
            continue
        for composed_tensor, base_tensor in zip(
            composed_layer, base_layer, strict=True
        ):
            assert torch.equal(composed_tensor, base_tensor)


@pytest.mark.parametrize(
    "base_layers,delta_layers,message",
    [
        (frozenset({0, 1}), frozenset({2}), "subset"),
        (frozenset(), frozenset({1}), "base_layers"),
        (frozenset({0, 1}), frozenset(), "delta_layers"),
    ],
)
def test_composition_rejects_invalid_layer_sets(
    base_layers, delta_layers, message
):
    base = _cache(seed=8)
    with pytest.raises(ValueError, match=message):
        compose_int4_base_delta(
            base,
            base,
            suffix_tokens=3,
            base_layers=base_layers,
            delta_layers=delta_layers,
        )
