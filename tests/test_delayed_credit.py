import torch

from elmur_klpo.exact_enumeration import evaluate_behavior
from elmur_klpo.manager import CategoricalManager
from elmur_klpo.verification import DIFFERENCE_FLOOR, early_action_gradient


def test_g_early_credit_comes_only_from_future_reader_performance(batch):
    policy = CategoricalManager(scale=0)
    fixed = evaluate_behavior(policy.frozen_copy(), batch)
    early = early_action_gradient(policy, batch)
    root_q = policy.probs(batch.node_states[0][0]).detach()
    analytic_credit = root_q * fixed.advantages[0][0]
    torch.testing.assert_close(early, analytic_credit, atol=1e-12, rtol=0)
    assert early.norm().item() > DIFFERENCE_FLOOR
    assert early[0].item() > 0  # Save cue in slot 0.
    assert early[5].item() > 0  # Save cue in slot 1.
    assert early[10].item() < 0  # Discard cue via global no-op.
    assert fixed.q_values[0][0, 0] > fixed.q_values[0][0, 10]
    assert bool((batch.rewards[:, :-1] == batch.rewards[0, 0]).all())


def test_later_writes_compete_with_preservation_of_the_early_cue(batch):
    def sequence_index(actions):
        matches = (batch.actions == torch.tensor(actions)).all(-1)
        return matches.nonzero().item()

    preserved = sequence_index([0, 5, 5])
    overwritten = sequence_index([0, 5, 0])
    torch.testing.assert_close(
        batch.memories[preserved, -1], torch.tensor([1.0, -0.5], dtype=torch.float64)
    )
    torch.testing.assert_close(
        batch.memories[overwritten, -1], torch.tensor([-0.5, -1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        batch.rewards[preserved, :-1], batch.rewards[overwritten, :-1]
    )
    assert batch.rewards[preserved, -1] > batch.rewards[overwritten, -1]
