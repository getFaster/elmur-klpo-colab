import math
from dataclasses import replace

import pytest
import torch

from elmur_klpo.exact_enumeration import (
    evaluate_behavior,
    population_surrogate,
    regression_objective,
)
from elmur_klpo.klpo import backward_surrogate
from elmur_klpo.manager import CategoricalManager
from elmur_klpo.verification import (
    ABS_TOL,
    BETAS,
    DIFFERENCE_FLOOR,
    SEEDS,
    finite_difference_gradient,
    gradient,
    gradient_error,
    incorrect_surrogate,
    nondegenerate_pair,
    score_centering_errors,
)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("beta", BETAS)
def test_a_off_policy_surrogate_matches_explicit_regression(batch, seed, beta):
    policy = CategoricalManager(seed=seed, scale=0.35)
    sampler = CategoricalManager(seed=seed + 100).frozen_copy()
    fixed = evaluate_behavior(sampler, batch)
    assert not torch.equal(policy.weight, sampler.weight)
    estimated = gradient(population_surrogate(policy, batch, fixed, beta), policy)
    ground_truth = gradient(regression_objective(policy, batch, fixed, beta), policy)
    torch.testing.assert_close(estimated, ground_truth, atol=ABS_TOL, rtol=0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("beta", BETAS)
def test_d_rtg_and_full_return_have_equal_population_gradients(batch, seed, beta):
    policy = CategoricalManager(seed=seed, scale=0.35)
    fixed = evaluate_behavior(CategoricalManager(seed=seed + 100), batch)
    rtg = gradient(population_surrogate(policy, batch, fixed, beta), policy)
    full = gradient(
        population_surrogate(policy, batch, fixed, beta, full_return=True), policy
    )
    torch.testing.assert_close(rtg, full, atol=ABS_TOL, rtol=0)


def test_d_population_identity_does_not_require_equal_sample_gradients(batch):
    policy, sampler = nondegenerate_pair()
    fixed = evaluate_behavior(sampler, batch)

    def sample_loss(full):
        returns = batch.returns[0]
        if full:
            returns = batch.rewards[0].sum().expand_as(returns)
        return backward_surrogate(
            policy.log_probs(batch.states[0]),
            fixed.log_probs[0],
            batch.actions[0],
            returns,
            0.7,
        )

    rtg = gradient(sample_loss(False), policy)
    full = gradient(sample_loss(True), policy)
    assert gradient_error(rtg, full).absolute > DIFFERENCE_FLOOR


@pytest.mark.parametrize("node_index", [0, 7, 120])
def test_e_center_matches_independent_score_vectors(batch, node_index):
    policy, sampler = nondegenerate_pair()
    errors = score_centering_errors(policy, sampler, batch.node_states[2][node_index])
    assert max(errors) < ABS_TOL


def test_e_detaching_center_is_an_incorrect_off_policy_gradient(batch):
    policy, sampler = nondegenerate_pair()
    fixed = evaluate_behavior(sampler, batch)
    correct = gradient(regression_objective(policy, batch, fixed, 0.7), policy)
    wrong = gradient(
        incorrect_surrogate(policy, batch, fixed, 0.7, detach_center=True), policy
    )
    assert gradient_error(correct, wrong).absolute > DIFFERENCE_FLOOR


def test_f_differentiating_feedback_changes_the_gradient(batch):
    policy, sampler = nondegenerate_pair()
    fixed = evaluate_behavior(sampler, batch)
    correct = gradient(population_surrogate(policy, batch, fixed, 0.7), policy)
    wrong = gradient(
        incorrect_surrogate(policy, batch, fixed, 0.7, detach_feedback=False), policy
    )
    assert gradient_error(correct, wrong).absolute > DIFFERENCE_FLOOR


def test_feedback_behavior_and_returns_are_detached():
    log_p = (
        torch.linspace(-3.0, -1.0, 22, dtype=torch.float64)
        .reshape(2, 11)
        .requires_grad_()
    )
    log_q = torch.full_like(log_p, -math.log(11.0)).requires_grad_()
    returns = torch.tensor([-1.5, -2.0], dtype=torch.float64, requires_grad=True)
    actions = torch.tensor([0, 7])
    loss = backward_surrogate(log_p, log_q, actions, returns, 0.7)
    dp, dq, dr = torch.autograd.grad(loss, (log_p, log_q, returns), allow_unused=True)
    assert dq is None
    assert dr is None
    with torch.no_grad():
        h = returns - 0.7 * (
            log_p.gather(-1, actions[:, None]).squeeze(-1) - log_q[:, 0]
        )
        one_hot = torch.nn.functional.one_hot(actions, 11).to(torch.float64)
        expected = -h[:, None] * (one_hot - log_q.exp())
    torch.testing.assert_close(dp, expected, atol=1e-12, rtol=0)


def test_population_surrogate_detaches_recorded_states(batch):
    policy, sampler = nondegenerate_pair()
    recorded = batch.states.clone().requires_grad_()
    batch_with_grad = replace(batch, states=recorded)
    fixed = evaluate_behavior(sampler, batch)
    loss = population_surrogate(policy, batch_with_grad, fixed, 0.7)
    state_gradient = torch.autograd.grad(loss, recorded, allow_unused=True)[0]
    assert state_gradient is None


def test_regression_autograd_matches_all_parameter_finite_differences(batch):
    policy, sampler = nondegenerate_pair()
    fixed = evaluate_behavior(sampler, batch)
    original = policy.weight.detach().clone()

    def objective():
        return regression_objective(policy, batch, fixed, 0.7)

    exact = gradient(objective(), policy)
    numerical = finite_difference_gradient(objective, policy)
    torch.testing.assert_close(exact, numerical, atol=ABS_TOL, rtol=0)
    torch.testing.assert_close(policy.weight, original, atol=0, rtol=0)


@pytest.mark.parametrize("beta", [0, -0.1, float("nan"), float("inf")])
def test_invalid_regularization_coefficient(batch, beta):
    policy, sampler = nondegenerate_pair()
    fixed = evaluate_behavior(sampler, batch)
    with pytest.raises(ValueError):
        population_surrogate(policy, batch, fixed, beta)
    with pytest.raises(ValueError):
        regression_objective(policy, batch, fixed, beta)
