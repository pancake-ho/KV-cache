import pytest
import torch

from xmodel_kv.cli.train_diverse_policy_capsule import first_difference
from xmodel_kv.cli.train_prompt_policy_capsule import _compiler_prompt
from xmodel_kv.policy_capsule import (
    PromptPolicyCompiler,
    SoftPolicyCapsule,
    _resample_rows,
    _resolve_prompt_states,
)


def test_resample_rows_preserves_shape_and_piecewise_means():
    source = torch.tensor([[0.0], [2.0], [4.0], [6.0]])
    result = _resample_rows(source, 2)
    torch.testing.assert_close(result, torch.tensor([[1.0], [5.0]]))


def test_capsule_parameter_count_and_drift():
    capsule = SoftPolicyCapsule(tokens=3, hidden_size=5)
    capsule.embeddings.data.zero_()
    capsule.initial_embeddings.zero_()
    assert sum(parameter.numel() for parameter in capsule.parameters()) == 15
    assert capsule.drift_regularization() == 0


def test_first_difference_handles_prefix_and_mismatch():
    assert first_difference((1, 2, 3), (1, 9, 3)) == 1
    assert first_difference((1, 2), (1, 2, 3)) == 2


def test_compiler_prompt_supports_full_text_and_binding_oracle():
    row = {
        "target_prompt": "full target prompt",
        "target_policy": "tenant_scope",
        "target_policy_value": "TENANT-BETA",
    }
    assert _compiler_prompt(row, "target", "full_prompt") == "full target prompt"
    assert _compiler_prompt(row, "target", "binding") == "tenant_scope=TENANT-BETA"


def test_prompt_compiler_shape_initialization_and_gradients():
    compiler = PromptPolicyCompiler(
        tokens=3,
        model_hidden_size=5,
        compiler_dim=8,
        layers=1,
        heads=2,
        max_prompt_tokens=7,
    )
    compiler.initial_embeddings.zero_()
    prompt = torch.randn(4, 5)
    compiled = compiler(prompt)
    assert compiled.shape == (3, 5)
    torch.testing.assert_close(compiled, torch.zeros_like(compiled))
    compiled.sum().backward()
    assert compiler.output_projection.weight.grad is not None
    assert compiler.output_projection.weight.grad.abs().sum() > 0


def test_prompt_compiler_rejects_overlong_prompts():
    compiler = PromptPolicyCompiler(
        tokens=2,
        model_hidden_size=4,
        compiler_dim=8,
        layers=1,
        heads=2,
        max_prompt_tokens=3,
    )
    compiler.initial_embeddings.zero_()
    with torch.no_grad(), pytest.raises(ValueError, match="prompt length"):
        compiler(torch.zeros(4, 4))


def test_prompt_state_resolution_accepts_precomputed_hidden_states():
    embedding = torch.nn.Embedding(11, 4)
    states = torch.randn(3, 4)
    resolved = _resolve_prompt_states(
        embedding,
        prompt_ids=None,
        prompt_states=states,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(resolved, states.float())
    with pytest.raises(ValueError, match="not both"):
        _resolve_prompt_states(
            embedding,
            prompt_ids=(1, 2),
            prompt_states=states,
            device=torch.device("cpu"),
        )
