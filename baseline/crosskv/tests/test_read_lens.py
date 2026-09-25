from types import SimpleNamespace

import torch

from xmodel_kv.read_lens import StaticReadLens, read_lens_attention_forward


def test_zero_static_lens_matches_unsplit_attention():
    torch.manual_seed(3)
    module = SimpleNamespace(
        num_key_value_groups=1,
        scaling=0.5,
        training=False,
        _static_read_lens=StaticReadLens(heads=2, head_dim=4, rank=2),
        _read_lens_enabled=True,
        _read_lens_stale_start=2,
        _read_lens_stale_end=6,
    )
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 8, 4)
    value = torch.randn(1, 2, 8, 4)
    output, _ = read_lens_attention_forward(
        module, query, key, value, None, scaling=module.scaling
    )
    weights = torch.softmax(
        torch.matmul(query, key.transpose(2, 3)) * module.scaling,
        dim=-1,
    )
    expected = torch.matmul(weights, value).transpose(1, 2)
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)


def test_beta_changes_only_stale_attention_mass():
    module = SimpleNamespace(
        num_key_value_groups=1,
        scaling=1.0,
        training=False,
        _static_read_lens=StaticReadLens(heads=1, head_dim=2, rank=1),
        _read_lens_enabled=True,
        _read_lens_stale_start=1,
        _read_lens_stale_end=3,
    )
    query = torch.zeros(1, 1, 1, 2)
    key = torch.zeros(1, 1, 4, 2)
    value = torch.eye(4, 2).view(1, 1, 4, 2)
    _, base_weights = read_lens_attention_forward(module, query, key, value, None)
    module._static_read_lens.beta.data.fill_(2.0)
    _, shifted_weights = read_lens_attention_forward(module, query, key, value, None)
    assert shifted_weights[..., 1:3].sum() > base_weights[..., 1:3].sum()
    torch.testing.assert_close(
        shifted_weights[..., 0] / shifted_weights[..., 3],
        base_weights[..., 0] / base_weights[..., 3],
    )
