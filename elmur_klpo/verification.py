"""Reproducible diagnostics for the seven mathematical claims in Step 1."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from elmur_klpo.exact_enumeration import (
    BehaviorEvaluation,
    TrajectoryBatch,
    enumerate_trajectories,
    evaluate_behavior,
    exact_expected_return,
    population_surrogate,
    regression_objective,
)
from elmur_klpo.klpo import score_center
from elmur_klpo.manager import CategoricalManager
from elmur_klpo.toy_env import ToyEnvironment

SEEDS = (7, 19, 41)
BETAS = (0.1, 0.7, 2.0)
ABS_TOL = 1e-8
DIFFERENCE_FLOOR = 1e-5


def gradient(loss: Tensor, policy: CategoricalManager) -> Tensor:
    """Flatten every trainable parameter gradient, preserving parameter order."""
    parameters = [p for p in policy.parameters() if p.requires_grad]
    return torch.cat(
        [g.flatten() for g in torch.autograd.grad(loss, parameters)]
    ).detach()


@dataclass(frozen=True)
class GradientError:
    absolute: float
    relative: float


def gradient_error(left: Tensor, right: Tensor) -> GradientError:
    difference = left - right
    denominator = max(left.norm().item(), right.norm().item(), 1e-15)
    return GradientError(
        difference.abs().max().item(), difference.norm().item() / denominator
    )


def nondegenerate_pair() -> tuple[CategoricalManager, CategoricalManager]:
    """Explicit uniform q and state-dependent p for C/E/F counterexamples.

    The coefficients are fixed rather than picked anew until a test passes.
    Diagnostics additionally check that true return gradients are nonzero.
    """
    learner = CategoricalManager(scale=0)
    behavior = learner.frozen_copy()
    with torch.no_grad():
        learner.weight[:, 0] = torch.linspace(-0.45, 0.55, 11, dtype=torch.float64)
        learner.weight[:, 3] = torch.linspace(0.6, -0.4, 11, dtype=torch.float64)
        learner.weight[:, 4] = torch.linspace(-0.3, 0.5, 11, dtype=torch.float64)
    return learner, behavior


def incorrect_surrogate(
    policy: CategoricalManager,
    batch: TrajectoryBatch,
    behavior: BehaviorEvaluation,
    beta: float,
    *,
    detach_center: bool = False,
    detach_feedback: bool = True,
) -> Tensor:
    """Deliberately incorrect controls used only for E/F sensitivity checks."""
    logs = policy.log_probs(batch.states.detach())
    chosen_p = logs.gather(-1, batch.actions[..., None]).squeeze(-1)
    chosen_q = behavior.log_probs.gather(-1, batch.actions[..., None]).squeeze(-1)
    h = batch.returns - beta * (chosen_p - chosen_q)
    if detach_feedback:
        h = h.detach()
    center = score_center(logs, behavior.log_probs.exp())
    if detach_center:
        center = center.detach()
    losses = -(h * (chosen_p - center)).sum(-1)
    return (behavior.trajectory_probs * losses).sum()


def score_centering_errors(
    policy: CategoricalManager, behavior: CategoricalManager, state: Tensor
) -> tuple[float, float, float]:
    """Compare analytic softmax scores, autograd scores, and grad(C).

    The independent formula for a linear softmax is
    d log p(u) / d W = (one_hot(u) - p) outer state.
    """
    logs = policy.log_probs(state.detach())
    q = behavior.probs(state.detach()).detach()
    scores = torch.stack(
        [
            torch.autograd.grad(logs[u], policy.weight, retain_graph=True)[0].flatten()
            for u in range(len(policy.action_space))
        ]
    )
    analytic = (
        (torch.eye(len(q), dtype=q.dtype) - logs.detach().exp()[None, :])[:, :, None]
        * state.detach()[None, None, :]
    ).flatten(1)
    center_grad = torch.autograd.grad(score_center(logs, q), policy.weight)[0].flatten()
    independent_center = (q[:, None] * scores).sum(0)
    centered_mean = (q[:, None] * (scores - center_grad)).sum(0)
    return (
        (scores - analytic).abs().max().item(),
        (center_grad - independent_center).abs().max().item(),
        centered_mean.abs().max().item(),
    )


def finite_difference_gradient(
    objective: Callable[[], Tensor], policy: CategoricalManager, step: float = 1e-5
) -> Tensor:
    """Central differences for every parameter of the ordinary scalar L_reg.

    Never use this on the backward surrogate: stop-gradient deliberately makes
    that surrogate's backward behavior differ from its scalar finite difference.
    """
    derivatives = []
    with torch.no_grad():
        for parameter in policy.parameters():
            for coordinate in parameter.view(-1):
                original = coordinate.clone()
                try:
                    coordinate.copy_(original + step)
                    plus = objective().item()
                    coordinate.copy_(original - step)
                    minus = objective().item()
                finally:
                    coordinate.copy_(original)
                derivatives.append((plus - minus) / (2 * step))
    return policy.weight.new_tensor(derivatives)


def early_action_gradient(policy: CategoricalManager, batch: TrajectoryBatch) -> Tensor:
    """Isolate credit to the first decision, holding all suffix policies fixed."""
    with torch.no_grad():
        chosen_logs = policy.action_log_probs(batch.states, batch.actions)
        suffix_probs = chosen_logs[:, 1:].sum(-1).exp()
    root_logits = policy(batch.node_states[0][0]).detach().requires_grad_()
    probabilities = root_logits.softmax(-1)[batch.actions[:, 0]] * suffix_probs
    expected_return = (probabilities * batch.rewards.sum(-1)).sum()
    return torch.autograd.grad(expected_return, root_logits)[0].detach()


@dataclass(frozen=True)
class Check:
    name: str
    metrics: dict[str, float]
    passed: bool
    expected_difference: bool = False


def run_checks(batch: TrajectoryBatch) -> list[Check]:
    """Run A-G over all declared seeds/betas and an all-parameter FD check."""
    checks = []
    for seed in SEEDS:
        policy = CategoricalManager(seed=seed, scale=0.35)
        sampler = CategoricalManager(seed=seed + 100).frozen_copy()
        fixed = evaluate_behavior(sampler, batch)
        for beta in BETAS:
            bp = gradient(population_surrogate(policy, batch, fixed, beta), policy)
            reg = gradient(regression_objective(policy, batch, fixed, beta), policy)
            error = gradient_error(bp, reg)
            checks.append(
                Check(
                    f"Test A: KLPO regression gradient identity (seed={seed}, beta={beta})",
                    {
                        "Absolute error": error.absolute,
                        "Relative error": error.relative,
                    },
                    error.absolute < ABS_TOL,
                )
            )
            full = gradient(
                population_surrogate(policy, batch, fixed, beta, full_return=True),
                policy,
            )
            error = gradient_error(bp, full)
            checks.append(
                Check(
                    f"Test D: RTG vs full return (seed={seed}, beta={beta})",
                    {
                        "Absolute error": error.absolute,
                        "Relative error": error.relative,
                    },
                    error.absolute < ABS_TOL,
                )
            )

        on_policy = evaluate_behavior(policy.frozen_copy(), batch)
        ascent = -gradient(population_surrogate(policy, batch, on_policy, 0.7), policy)
        true_gradient = gradient(exact_expected_return(policy, batch), policy)
        error = gradient_error(ascent, true_gradient)
        checks.append(
            Check(
                f"Test B: On-policy equality at p=q (seed={seed}, beta=0.7)",
                {"Absolute error": error.absolute, "Relative error": error.relative},
                error.absolute < ABS_TOL,
            )
        )

    policy, sampler = nondegenerate_pair()
    fixed = evaluate_behavior(sampler, batch)
    beta = 0.7
    correct = gradient(population_surrogate(policy, batch, fixed, beta), policy)
    reg_ascent = -gradient(regression_objective(policy, batch, fixed, beta), policy)
    true_gradient = gradient(exact_expected_return(policy, batch), policy)
    discrepancy = gradient_error(reg_ascent, true_gradient).absolute
    checks.append(
        Check(
            "Test C: Off-policy difference (explicit coefficients, beta=0.7)",
            {
                "Gradient discrepancy": discrepancy,
                "True-gradient norm": true_gradient.norm().item(),
            },
            discrepancy > DIFFERENCE_FLOOR
            and true_gradient.norm().item() > DIFFERENCE_FLOOR,
            expected_difference=True,
        )
    )
    errors = score_centering_errors(policy, sampler, batch.node_states[2][7])
    wrong = gradient(
        incorrect_surrogate(policy, batch, fixed, beta, detach_center=True), policy
    )
    difference = gradient_error(correct, wrong).absolute
    checks.append(
        Check(
            "Test E: Exact score centering (explicit coefficients, beta=0.7)",
            {"Error": max(errors), "Detached-center gradient difference": difference},
            max(errors) < ABS_TOL and difference > DIFFERENCE_FLOOR,
        )
    )
    wrong = gradient(
        incorrect_surrogate(policy, batch, fixed, beta, detach_feedback=False), policy
    )
    difference = gradient_error(correct, wrong).absolute
    checks.append(
        Check(
            "Test F: Stop-gradient (explicit coefficients, beta=0.7)",
            {"Gradient difference": difference},
            difference > DIFFERENCE_FLOOR,
        )
    )
    early = early_action_gradient(sampler, batch)
    checks.append(
        Check(
            "Test G: Delayed credit assignment (uniform on-policy manager)",
            {"Early-action gradient norm": early.norm().item()},
            early.norm().item() > DIFFERENCE_FLOOR
            and bool(torch.all(batch.rewards[:, :-1] == batch.rewards[0, 0]))
            and early[0].item() > 0
            and early[-1].item() < 0,
        )
    )

    def objective() -> Tensor:
        return regression_objective(policy, batch, fixed, beta)

    exact = gradient(objective(), policy)
    numerical = finite_difference_gradient(objective, policy)
    error = gradient_error(exact, numerical)
    checks.append(
        Check(
            "Finite differences: explicit L_reg (all 55 parameters, step=1e-5, beta=0.7)",
            {"Absolute error": error.absolute, "Relative error": error.relative},
            error.absolute < ABS_TOL,
        )
    )
    return checks


def main() -> int:
    # Tiny exact sums run faster and more reproducibly without a large BLAS pool.
    torch.set_num_threads(1)
    env = ToyEnvironment()
    batch = enumerate_trajectories(env)
    print("ELMUR-KLPO Step 1: Gradient Verification\n")
    print(
        f"Action space: {len(env.action_space)} (slot-major; no-op ID={env.action_space.noop_id})"
    )
    print(f"Memory slots: {env.action_space.slots}")
    print(f"Effective manager decisions: {env.decisions}")
    print(f"Enumerated trajectories: {len(batch.actions)}")
    print(
        f"dtype: float64; seeds: {SEEDS}; betas: {BETAS}; absolute tolerance: {ABS_TOL}"
    )
    print("Seeded policies: learner scale=0.35; behavior seed=seed+100, scale=0.2")
    print(
        "Explicit p: bias=linspace(-.45,.55), m0=linspace(.6,-.4), m1=linspace(-.3,.5)"
    )
    print("Explicit q: uniform; all other coefficients zero; 11 points per linspace\n")
    checks = run_checks(batch)
    # Report the worst error over repeated cases, retaining the detailed cases
    # above for programmatic inspection and parametrized pytest coverage.
    report = []
    for letter in "ABCDEFG":
        group = [check for check in checks if check.name.startswith(f"Test {letter}:")]
        if len(group) == 1:
            report.extend(group)
        else:
            report.append(
                Check(
                    group[0].name.split(" (")[0]
                    + f" ({len(group)} cases; worst errors)",
                    {
                        key: max(c.metrics[key] for c in group)
                        for key in group[0].metrics
                    },
                    all(c.passed for c in group),
                )
            )
    report.extend(
        check for check in checks if check.name.startswith("Finite differences:")
    )
    for check in report:
        print(check.name)
        for label, value in check.metrics.items():
            print(f"  {label}: {value:.12e}")
        status = "EXPECTED DIFFERENCE" if check.expected_difference else "PASS"
        print(f"  {status if check.passed else 'FAIL'}\n")
    passed = all(check.passed for check in checks)
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1
