import torch

from xmodel_kv.ridge import (
    fit_ridge,
    head_covariance,
    single_source_head_r2,
    single_source_head_r2_from_stats,
)


def test_ridge_recovers_linear_map_and_bias():
    torch.manual_seed(1)
    x = torch.randn(400, 12)
    expected_weight = torch.randn(12, 5)
    expected_bias = torch.randn(5)
    y = x @ expected_weight + expected_bias
    result = fit_ridge(x, y, ridge=1e-6)
    assert result.solver == "primal"
    torch.testing.assert_close(result.weight.float(), expected_weight, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(result.bias.float(), expected_bias, atol=2e-5, rtol=2e-5)
    assert float(result.r2.min()) > 0.999999


def test_dual_solver_for_small_calibration_smoke():
    torch.manual_seed(2)
    x = torch.randn(8, 24)
    y = torch.randn(8, 3)
    result = fit_ridge(x, y, ridge=0.01)
    assert result.solver == "dual"
    assert result.weight.shape == (24, 3)
    assert result.bias.shape == (3,)


def test_analytic_batched_head_r2_matches_token_predictions():
    torch.manual_seed(5)
    source = torch.randn(300, 3, 8)
    target = torch.einsum("nhd,hdo->nho", source, torch.randn(3, 8, 8))
    target = target + 0.1 * torch.randn(300, 3, 8)
    expected = single_source_head_r2(source, target, ridge=0.01)
    source_mean, source_covariance = head_covariance(source)
    actual = single_source_head_r2_from_stats(
        source,
        target,
        source_mean=source_mean,
        source_covariance=source_covariance,
        ridge=0.01,
    ).mean(dim=-1)
    torch.testing.assert_close(actual, expected.float(), atol=2e-5, rtol=2e-5)
