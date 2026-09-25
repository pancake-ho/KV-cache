import torch

from xmodel_kv.capsule_codec import pack_int4_capsule, unpack_int4_capsule
from xmodel_kv.composable_delta import compose_int4_base_delta
from xmodel_kv.differentiable_delta import (
    compose_fake_int4_base_delta,
    fake_int4_cache_round_trip,
)


def _cache(*, seed: int, layers: int = 4, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        (
            torch.randn(1, 2, 3, 4, generator=generator).to(torch.bfloat16),
            torch.randn(1, 2, 3, 4, generator=generator).to(torch.bfloat16),
        )
        for _ in range(layers)
    )


def test_differentiable_forward_is_bitwise_production_equivalent():
    base = _cache(seed=11)
    task = _cache(seed=12)
    expected = compose_int4_base_delta(
        base,
        task,
        suffix_tokens=3,
        base_layers=frozenset({0, 1, 2}),
        delta_layers=frozenset({2}),
    )
    actual = compose_fake_int4_base_delta(
        base,
        task,
        suffix_tokens=3,
        base_layers=frozenset({0, 1, 2}),
        delta_layers=frozenset({2}),
    )

    for actual_cache, expected_cache in (
        (actual.base_cache, expected.base_cache),
        (actual.delta_cache, expected.delta_cache),
        (actual.cache, expected.cache),
    ):
        for actual_layer, expected_layer in zip(
            actual_cache, expected_cache, strict=True
        ):
            for actual_tensor, expected_tensor in zip(
                actual_layer, expected_layer, strict=True
            ):
                assert torch.equal(actual_tensor, expected_tensor)


def test_differentiable_composition_routes_gradients_only_through_delta_layer():
    base = _cache(seed=13)
    raw_task = _cache(seed=14)
    task = tuple(
        tuple(tensor.detach().clone().requires_grad_(True) for tensor in layer)
        for layer in raw_task
    )
    composition = compose_fake_int4_base_delta(
        base,
        task,
        suffix_tokens=3,
        base_layers=frozenset({0, 1, 2}),
        delta_layers=frozenset({2}),
    )

    loss = sum(tensor.float().sum() for layer in composition.cache for tensor in layer)
    loss.backward()

    for layer_index, layer in enumerate(task):
        for tensor in layer:
            if layer_index == 2:
                assert tensor.grad is not None
                torch.testing.assert_close(
                    tensor.grad,
                    torch.ones_like(tensor.grad),
                    rtol=0,
                    atol=0,
                )
            else:
                assert tensor.grad is None


def test_zero_delta_reconstructs_decoded_base_bitwise():
    base = _cache(seed=15)
    composition = compose_fake_int4_base_delta(
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


def test_monolithic_fake_int4_matches_production_packet_and_routes_gradient():
    raw = _cache(seed=16)
    cache = tuple(
        tuple(tensor.detach().clone().requires_grad_(True) for tensor in layer)
        for layer in raw
    )
    active = frozenset({0, 1, 2})
    packet = pack_int4_capsule(raw, suffix_tokens=3, active_layers=active)
    expected = unpack_int4_capsule(packet, dtype=torch.bfloat16)
    actual = fake_int4_cache_round_trip(
        cache,
        suffix_tokens=3,
        active_layers=active,
        straight_through=True,
    )

    for layer_index, (actual_layer, expected_layer) in enumerate(
        zip(actual, expected, strict=True)
    ):
        for actual_tensor, expected_tensor in zip(
            actual_layer, expected_layer, strict=True
        ):
            assert torch.equal(actual_tensor, expected_tensor)
        if layer_index not in active:
            assert not any(tensor.requires_grad for tensor in actual_layer)

    sum(tensor.float().sum() for layer in actual for tensor in layer).backward()
    for layer_index, layer in enumerate(cache):
        for tensor in layer:
            if layer_index in active:
                assert tensor.grad is not None
                torch.testing.assert_close(tensor.grad, torch.ones_like(tensor.grad))
            else:
                assert tensor.grad is None
