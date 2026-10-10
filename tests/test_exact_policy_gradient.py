import pytest
import torch

from elmur_klpo.exact_enumeration import (
    evaluate_behavior,
    exact_expected_return,
    population_surrogate,
    regression_objective,
    trajectory_probabilities,
)
from elmur_klpo.manager import CategoricalManager
from elmur_klpo.verification import (
    ABS_TOL,
    DIFFERENCE_FLOOR,
    SEEDS,
    gradient,
    gradient_error,
    nondegenerate_pair,
)


@pytest.mark.parametrize("seed", SEEDS)
def test_b_klpo_ascent_matches_every_parameter_of_true_on_policy_gradient(batch, seed):
    policy = CategoricalManager(seed=seed, scale=0.35)
    fixed = evaluate_behavior(policy.frozen_copy(), batch)
    ascent = -gradient(population_surrogate(policy, batch, fixed, 0.7), policy)
    exact = gradient(exact_expected_return(policy, batch), policy)
    assert exact.norm().item() > DIFFERENCE_FLOOR
    assert exact.numel() == sum(p.numel() for p in policy.parameters())
    torch.testing.assert_close(ascent, exact, atol=ABS_TOL, rtol=0)


def test_c_explicit_off_policy_example_differs_from_true_current_policy_gradient(batch):
    policy, sampler = nondegenerate_pair()
    fixed = evaluate_behavior(sampler, batch)
    ascent = -gradient(regression_objective(policy, batch, fixed, 0.7), policy)
    exact = gradient(exact_expected_return(policy, batch), policy)
    assert exact.norm().item() > DIFFERENCE_FLOOR
    assert gradient_error(ascent, exact).absolute > DIFFERENCE_FLOOR


def test_expected_return_gradient_against_analytic_linear_softmax_scores(batch):
    policy = CategoricalManager(seed=19, scale=0.35)
    exact = gradient(exact_expected_return(policy, batch), policy)
    with torch.no_grad():
        probabilities = policy.probs(batch.states)
        one_hot = torch.nn.functional.one_hot(batch.actions, 11).to(torch.float64)
        # Per-decision score matrices, summed to get the trajectory score.
        scores = (one_hot - probabilities)[..., None] * batch.states[:, :, None, :]
        weights = trajectory_probabilities(policy, batch) * batch.rewards.sum(-1)
        independent = (weights[:, None, None] * scores.sum(1)).sum(0).flatten()
    torch.testing.assert_close(exact, independent, atol=ABS_TOL, rtol=0)


def test_exact_q_values_against_complete_trajectory_conditional_sums(batch):
    sampler = CategoricalManager(seed=107)
    fixed = evaluate_behavior(sampler, batch)
    assert not any(value.requires_grad for value in fixed.q_values)
    torch.testing.assert_close(
        fixed.trajectory_probs.sum(), torch.tensor(1.0, dtype=torch.float64)
    )
    prefix = torch.zeros(len(batch.actions), dtype=torch.long)
    for time, (q_values, values, advantage, log_q, visits) in enumerate(
        zip(
            fixed.q_values,
            fixed.values,
            fixed.advantages,
            fixed.node_log_probs,
            fixed.visitation,
            strict=True,
        )
    ):
        joint_ids = prefix * 11 + batch.actions[:, time]
        masses = torch.zeros(q_values.numel(), dtype=torch.float64)
        weighted_returns = torch.zeros_like(masses)
        masses.scatter_add_(0, joint_ids, fixed.trajectory_probs)
        weighted_returns.scatter_add_(
            0, joint_ids, fixed.trajectory_probs * batch.returns[:, time]
        )
        conditional = (weighted_returns / masses).reshape_as(q_values)
        torch.testing.assert_close(q_values, conditional, atol=1e-12, rtol=0)
        torch.testing.assert_close(values, (log_q.exp() * q_values).sum(-1))
        torch.testing.assert_close(
            (log_q.exp() * advantage).sum(-1),
            torch.zeros_like(values),
            atol=1e-12,
            rtol=0,
        )
        torch.testing.assert_close(visits.sum(), torch.tensor(1.0, dtype=torch.float64))
        assert not log_q.requires_grad
        prefix = joint_ids
    torch.testing.assert_close(
        exact_expected_return(sampler, batch).detach(),
        batch.node_rewards[0][0] + fixed.values[0][0],
    )
    # No history is merged, even for multiple paths yielding identical memories.
    assert [len(states) for states in batch.node_states] == [1, 11, 121]
