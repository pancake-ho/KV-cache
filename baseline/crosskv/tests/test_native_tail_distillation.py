import pytest
import torch

from xmodel_kv.native_tail_distillation import cache_prefix_relative_mse


def _cache(values):
    return tuple(
        (
            torch.tensor(value, dtype=torch.float32).reshape(1, 1, -1, 2),
            torch.tensor(value, dtype=torch.float32).reshape(1, 1, -1, 2),
        )
        for value in values
    )


def test_cache_prefix_relative_mse_aligns_short_teacher_prefix():
    teacher = _cache(([[1, 2], [3, 4]], [[2, 1], [4, 3]]))
    student = _cache(
        (
            [[1, 2], [3, 4], [20, 20]],
            [[2, 1], [4, 3], [20, 20]],
        )
    )

    loss = cache_prefix_relative_mse(
        student, teacher, active_layers=frozenset({0, 1})
    )

    assert loss.item() == pytest.approx(0.0)


def test_cache_prefix_relative_mse_is_teacher_normalized_and_layer_masked():
    teacher = _cache(([[1, 2]], [[2, 4]]))
    student = _cache(([[100, 200]], [[4, 8]]))

    loss = cache_prefix_relative_mse(
        student, teacher, active_layers=frozenset({1})
    )

    # Doubling the selected teacher vector gives squared error / teacher power = 1.
    assert loss.item() == pytest.approx(1.0)


def test_cache_prefix_relative_mse_rejects_short_student():
    with pytest.raises(ValueError, match="student must contain"):
        cache_prefix_relative_mse(
            _cache(([[1, 2]],)),
            _cache(([[1, 2], [3, 4]],)),
            active_layers=frozenset({0}),
        )
