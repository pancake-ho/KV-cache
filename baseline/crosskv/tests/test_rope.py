import torch

from xmodel_kv.rope import apply_rope, remove_rope


def test_rope_inverse_for_bhtd_cache():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 8)
    angle = torch.randn(7, 4).repeat(1, 2)
    cos, sin = angle.cos(), angle.sin()
    restored = remove_rope(apply_rope(x, cos, sin, sequence_dim=2), cos, sin, sequence_dim=2)
    torch.testing.assert_close(restored, x, atol=2e-6, rtol=2e-6)


def test_rope_inverse_with_per_example_positions():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 7, 8)
    angle = torch.randn(2, 7, 4).repeat(1, 1, 2)
    cos, sin = angle.cos(), angle.sin()
    restored = remove_rope(apply_rope(x, cos, sin, sequence_dim=2), cos, sin, sequence_dim=2)
    torch.testing.assert_close(restored, x, atol=2e-6, rtol=2e-6)
