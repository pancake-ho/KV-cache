import torch

from xmodel_kv.cli.evaluate_musique_tail_transplant import (
    active_head_indices,
    active_layer_indices,
    low_rank_head_reconstruction,
    mask_and_fake_quantize_suffix,
)


def test_active_layer_patterns_partition_layers():
    assert active_layer_indices("all", total_layers=6) == frozenset(range(6))
    assert active_layer_indices("even", total_layers=6) == frozenset({0, 2, 4})
    assert active_layer_indices("odd", total_layers=6) == frozenset({1, 3, 5})
    assert active_layer_indices("first_half", total_layers=6) == frozenset({0, 1, 2})
    assert active_layer_indices("last_half", total_layers=6) == frozenset({3, 4, 5})
    assert active_layer_indices("first_3", total_layers=6) == frozenset({0, 1, 2})
    assert active_layer_indices("last_2", total_layers=6) == frozenset({4, 5})
    assert active_layer_indices("mod4_2", total_layers=6) == frozenset({2})


def test_active_head_patterns_select_gqa_heads():
    assert active_head_indices("all", total_heads=8) == frozenset(range(8))
    assert active_head_indices("even", total_heads=8) == frozenset({0, 2, 4, 6})
    assert active_head_indices("last_3", total_heads=8) == frozenset({5, 6, 7})


def test_mask_and_fake_quantize_suffix_preserves_prefix_and_masks_unsent_layers():
    torch.manual_seed(4)
    cache = tuple(
        (
            torch.randn(1, 2, 5, 8),
            torch.randn(1, 2, 5, 8),
        )
        for _ in range(3)
    )
    result = mask_and_fake_quantize_suffix(
        cache,
        suffix_tokens=2,
        bits=4,
        active_layers=frozenset({1}),
    )

    for layer_index, ((key, value), (new_key, new_value)) in enumerate(
        zip(cache, result, strict=True)
    ):
        torch.testing.assert_close(new_key[:, :, :3], key[:, :, :3])
        torch.testing.assert_close(new_value[:, :, :3], value[:, :, :3])
        if layer_index == 1:
            assert torch.count_nonzero(new_key[:, :, 3:])
            assert torch.count_nonzero(new_value[:, :, 3:])
        else:
            assert not torch.count_nonzero(new_key[:, :, 3:])
            assert not torch.count_nonzero(new_value[:, :, 3:])


def test_full_head_rank_reconstructs_suffix_without_factor_quantization():
    torch.manual_seed(8)
    suffix = torch.randn(1, 4, 3, 7)
    reconstructed = low_rank_head_reconstruction(suffix, rank=4, bits=16)
    torch.testing.assert_close(reconstructed, suffix, rtol=1e-5, atol=1e-5)


def test_packet_fake_quant_straight_through_matches_forward_and_routes_gradients():
    base = torch.linspace(-1, 1, 16).reshape(1, 2, 2, 4).requires_grad_()
    value = (base.detach() * 0.5).requires_grad_()
    cache = ((base, value),)
    exact = mask_and_fake_quantize_suffix(
        cache,
        suffix_tokens=2,
        bits=4,
        active_layers=frozenset({0}),
    )
    ste = mask_and_fake_quantize_suffix(
        cache,
        suffix_tokens=2,
        bits=4,
        active_layers=frozenset({0}),
        straight_through=True,
    )
    torch.testing.assert_close(ste[0][0], exact[0][0])
    torch.testing.assert_close(ste[0][1], exact[0][1])
    (ste[0][0].sum() + ste[0][1].sum()).backward()
    assert torch.count_nonzero(base.grad) == base.numel()
    assert torch.count_nonzero(value.grad) == value.numel()
