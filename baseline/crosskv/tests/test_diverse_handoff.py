from xmodel_kv.diverse_handoff import (
    HELDOUT_POLICIES,
    TRAIN_POLICIES,
    _candidate_metadata,
    _role_pair_pools,
    select_candidate,
)
import random


def test_policy_selectors_return_valid_and_diverse_candidates():
    rows = _candidate_metadata(random.Random(4), 8)
    choices = set()
    for offset, policy in enumerate((*TRAIN_POLICIES, *HELDOUT_POLICIES)):
        expected = offset % 8
        choices.add(select_candidate(rows, policy, rows[expected][policy]))
    assert choices <= set(range(8))
    assert len(choices) == 8


def test_role_pair_ood_is_disjoint_and_roles_remain_seen():
    pools = _role_pair_pools()
    assert set(pools["train"]).isdisjoint(pools["heldout"])
    train_roles = {role for pair in pools["train"] for role in pair}
    heldout_roles = {role for pair in pools["heldout"] for role in pair}
    assert heldout_roles <= train_roles
