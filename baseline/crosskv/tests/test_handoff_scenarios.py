import os
from pathlib import Path

import pytest

from xmodel_kv.handoff_scenarios import (
    collaboration_event_cases,
    rag_policy_cases,
    score_arguments,
)


HOTPOT_DATASET = Path(
    os.environ.get("XKV_HOTPOT_DATASET", "datasets/longbench/hotpotqa.jsonl")
)
BFCL_ROOT = Path(
    os.environ.get(
        "XKV_BFCL_ROOT",
        "datasets/bfcl_official/berkeley-function-call-leaderboard/bfcl_eval/data",
    )
)


@pytest.mark.skipif(
    not HOTPOT_DATASET.exists(),
    reason="set XKV_HOTPOT_DATASET to run the HotpotQA integration test",
)
def test_rag_policy_case_has_counterfactual_namespaces_and_distinct_answers():
    cases = rag_policy_cases(
        HOTPOT_DATASET,
        max_cases=2,
        seed=11,
    )
    assert len(cases) == 2
    for case in cases:
        assert case.scenario == "rag_namespace"
        assert case.source_expected != case.target_expected
        shared = case.history[0]["content"]
        assert "CORPUS-ONYX" in shared
        assert "CORPUS-IVORY" in shared
        assert case.source_expected["arguments"]["answer"] in shared
        assert case.target_expected["arguments"]["answer"] in shared


@pytest.mark.skipif(
    not BFCL_ROOT.exists(), reason="set XKV_BFCL_ROOT to run BFCL integration tests"
)
def test_collaboration_case_requires_two_distinct_cross_tenant_jobs():
    case = collaboration_event_cases(
        BFCL_ROOT, max_cases=1, seed=13, background_records=2
    )[0]
    source = case.source_expected["arguments"]["job_ids"]
    target = case.target_expected["arguments"]["job_ids"]
    assert len(source) == len(set(source)) == 2
    assert len(target) == len(set(target)) == 2
    assert set(source).isdisjoint(target)
    assert case.metadata["job_count"] == 8
    assert case.metadata["event_count"] == 16
    assert "AGENT A HANDOFF OUTPUT" in case.history[-2]["content"]


def test_rag_scoring_normalizes_short_answer_format():
    expected = {"arguments": {"answer": "Miller v. California"}}
    assert score_arguments(
        "rag_namespace", {"answer": "The Miller v. California."}, expected
    )
