import pytest

from xmodel_kv.cli.summarize_policy_triad import _exact_mcnemar, _wilson


def test_wilson_interval_contains_observed_proportion():
    low, high = _wilson(20, 25)
    assert low < 0.8 < high


def test_wilson_interval_for_all_successes_is_not_point_mass():
    low, high = _wilson(25, 25)
    assert 0 < low < 1
    assert high > 0.999999


def test_exact_mcnemar_for_all_discordance_in_one_direction():
    assert _exact_mcnemar(25, 0) == pytest.approx(2 / (2**25))
    assert _exact_mcnemar(0, 25) == pytest.approx(2 / (2**25))


def test_exact_mcnemar_for_no_discordance():
    assert _exact_mcnemar(0, 0) == 1.0
