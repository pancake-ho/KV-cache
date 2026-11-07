import pytest

from xmodel_kv.cli.evaluate_musique_soft_tail_capsule import shifted_source_cases


def test_shifted_source_cases_rotates_without_changing_targets():
    cases = [(10, {"id": "a"}), (11, {"id": "b"}), (12, {"id": "c"})]

    shifted = shifted_source_cases(cases, 1)

    assert [case[1]["id"] for case in shifted] == ["b", "c", "a"]


def test_shifted_source_cases_rejects_nonzero_identity_rotation():
    with pytest.raises(ValueError, match="must change"):
        shifted_source_cases([(10, {"id": "a"}), (11, {"id": "b"})], 2)
