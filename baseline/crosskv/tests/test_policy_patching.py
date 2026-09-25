import pytest

from xmodel_kv.cli.policy_patching import _parse_layer_ranges


def test_parse_layer_ranges():
    assert _parse_layer_ranges("") == ()
    assert _parse_layer_ranges("18-29, 30-35") == ((18, 29), (30, 35))


@pytest.mark.parametrize("value", ["18", "a-2", "-1-2", "4-3", "1-2,1-2"])
def test_parse_layer_ranges_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _parse_layer_ranges(value)
