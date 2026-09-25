import pytest

from xmodel_kv.hotpotqa import answer_em, answer_f1, clean_short_answer, normalize_answer
from xmodel_kv.cli.hotpot_agents import _proposal_answer


def test_hotpot_normalization_and_em():
    assert normalize_answer("The Miller v. California!") == "miller v california"
    assert answer_em("Miller v. California", ["Miller v. California"]) == 1.0


def test_hotpot_token_f1():
    assert answer_f1("Miller California", ["Miller v. California"]) == pytest.approx(0.8)


def test_clean_short_answer():
    assert clean_short_answer("Final answer: Paris\nExplanation") == "Paris"
    assert clean_short_answer("Dracula<|im_end|>") == "Dracula"


def test_proposal_answer():
    report = "EVIDENCE: two linked facts\nPROPOSED_ANSWER: West Lafayette, Indiana"
    assert _proposal_answer(report) == "West Lafayette, Indiana"
    assert _proposal_answer("EVIDENCE: no candidate") == ""
