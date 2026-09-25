import pytest
import torch

from xmodel_kv.attention_operator import (
    attention_memory_statistics,
    attention_operator_distillation_loss,
)


def test_attention_operator_is_invariant_to_joint_kv_permutation():
    generator = torch.Generator().manual_seed(7)
    queries = torch.randn(1, 4, 5, 8, generator=generator)
    keys = torch.randn(1, 2, 3, 8, generator=generator)
    values = torch.randn(1, 2, 3, 8, generator=generator)
    permutation = torch.tensor([2, 0, 1])

    loss = attention_operator_distillation_loss(
        queries,
        keys[:, :, permutation],
        values[:, :, permutation],
        keys,
        values,
    )

    assert loss.total.item() == pytest.approx(0.0, abs=1e-12)


def test_log_mass_detects_duplicate_memory_even_when_conditional_value_matches():
    queries = torch.tensor([[[[1.0, 0.0]]]])
    keys = torch.tensor([[[[1.0, 0.0]]]])
    values = torch.tensor([[[[2.0, 3.0]]]])
    duplicated_keys = keys.repeat(1, 1, 2, 1)
    duplicated_values = values.repeat(1, 1, 2, 1)

    loss = attention_operator_distillation_loss(
        queries,
        duplicated_keys,
        duplicated_values,
        keys,
        values,
    )

    assert loss.value.item() == pytest.approx(0.0)
    assert loss.log_mass.item() == pytest.approx(torch.log(torch.tensor(2.0)).square().item())
    assert loss.total.item() > 0


def test_attention_memory_statistics_supports_grouped_query_attention():
    queries = torch.zeros(1, 4, 3, 2)
    keys = torch.zeros(1, 2, 2, 2)
    values = torch.tensor([[[[1.0, 3.0], [3.0, 5.0]], [[2.0, 4.0], [4.0, 6.0]]]])

    statistics = attention_memory_statistics(queries, keys, values)

    assert statistics.log_mass.shape == (1, 4, 3)
    assert statistics.conditional_value.shape == (1, 4, 3, 2)
    assert torch.equal(
        statistics.conditional_value[0, 0, 0], torch.tensor([2.0, 4.0])
    )
    assert torch.equal(
        statistics.conditional_value[0, 2, 0], torch.tensor([3.0, 5.0])
    )
