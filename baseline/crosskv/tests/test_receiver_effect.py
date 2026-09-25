import pytest
import torch

from xmodel_kv.receiver_effect import (
    counterfactual_receiver_effect_loss,
    slice_cache_tokens,
    zero_cache_like,
)


def _trajectory(effect):
    null = (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
    actual = tuple(value + effect for value in null)
    return actual, null


def test_counterfactual_effect_is_zero_for_equal_effects_despite_baseline_shift():
    student, student_null = _trajectory(torch.tensor(2.0))
    teacher, teacher_null = _trajectory(torch.tensor(2.0))
    student_null = tuple(value + 7 for value in student_null)
    student = tuple(value + 7 for value in student)

    loss = counterfactual_receiver_effect_loss(
        student, student_null, teacher, teacher_null
    )

    assert loss.direction.item() == pytest.approx(0.0, abs=1e-7)
    assert loss.log_norm.item() == pytest.approx(0.0, abs=1e-7)
    assert loss.total.item() == pytest.approx(0.0, abs=1e-7)


def test_counterfactual_effect_separates_direction_and_magnitude():
    student, student_null = _trajectory(torch.tensor(-4.0))
    teacher, teacher_null = _trajectory(torch.tensor(2.0))

    loss = counterfactual_receiver_effect_loss(
        student, student_null, teacher, teacher_null, log_norm_weight=0.1
    )

    assert loss.direction.item() == pytest.approx(2.0)
    assert loss.log_norm.item() == pytest.approx(torch.log(torch.tensor(2.0)).square())


def test_zero_and_slice_cache_preserve_topology():
    cache = tuple(
        (torch.randn(1, 2, 5, 3), torch.randn(1, 2, 5, 3)) for _ in range(2)
    )

    sliced = slice_cache_tokens(cache, 2)
    zero = zero_cache_like(sliced)

    assert len(zero) == 2
    assert zero[0][0].shape == (1, 2, 3, 3)
    assert torch.count_nonzero(zero[1][1]).item() == 0
