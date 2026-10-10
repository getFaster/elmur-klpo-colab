import pytest
import torch

from elmur_klpo.manager import CategoricalManager
from elmur_klpo.memory import ActionSpace


def test_action_mapping_is_slot_major_with_one_global_noop():
    space = ActionSpace(2)
    assert len(space) == 11
    assert space.actions() == (
        (0, 0.0),
        (0, 0.25),
        (0, 0.5),
        (0, 0.8),
        (0, 0.95),
        (1, 0.0),
        (1, 0.25),
        (1, 0.5),
        (1, 0.8),
        (1, 0.95),
        (None, 1.0),
    )
    assert space.noop_id == 10
    assert len(set(space.actions())) == len(space)


@pytest.mark.parametrize("slots", [1, 2, 3])
def test_action_count(slots):
    assert len(ActionSpace(slots)) == 5 * slots + 1


@pytest.mark.parametrize("action", [-1, 11])
def test_invalid_action_id(action):
    with pytest.raises(ValueError):
        ActionSpace().decode(action)


def test_invalid_slot_count():
    with pytest.raises(ValueError):
        ActionSpace(0)


def test_policy_probabilities_sampling_and_frozen_snapshot(batch):
    policy = CategoricalManager(seed=7)
    states = batch.states[:13]
    probabilities = policy.probs(states)
    assert probabilities.dtype == torch.float64
    assert bool((probabilities > 0).all())
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(states[..., 0]))
    actions, logs = policy.sample(states)
    torch.testing.assert_close(logs, policy.action_log_probs(states, actions))
    torch.testing.assert_close(logs, policy.distribution(states).log_prob(actions))
    snapshot = policy.frozen_copy()
    assert not any(p.requires_grad for p in snapshot.parameters())
    with torch.no_grad():
        policy.weight[0, 0] += 1
    assert not torch.equal(snapshot.weight, policy.weight)
