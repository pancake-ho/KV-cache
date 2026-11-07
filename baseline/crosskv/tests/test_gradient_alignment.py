import pytest
import torch

from xmodel_kv.gradient_alignment import (
    gradient_alignment,
    loss_gradient_tuple,
    project_away_conflicting_component,
)


def test_gradient_alignment_and_conflict_projection():
    candidate = (torch.tensor([1.0, -1.0]),)
    protected = (torch.tensor([-1.0, 0.0]),)

    before = gradient_alignment(candidate, protected)
    projected = project_away_conflicting_component(candidate, protected)
    after = gradient_alignment(projected, protected)

    assert before.dot == -1.0
    assert before.cosine == pytest.approx(-0.5**0.5)
    assert after.dot == pytest.approx(0.0)


def test_loss_gradient_tuple_materializes_parameter_gradients():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    loss = parameter.square().sum()

    gradients = loss_gradient_tuple(loss, (parameter,), retain_graph=False)

    assert torch.equal(gradients[0], torch.tensor([4.0, -6.0]))
