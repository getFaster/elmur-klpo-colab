"""Exact trajectory sums and independently constructed conditional Q-values.

Every action prefix retains its own node, even when two histories have equal
memory. The timestep is also present in policy features, making the toy state
sufficient. Nothing in the behavior evaluation depends on learner parameters.
"""

import itertools
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from elmur_klpo.klpo import backward_surrogate
from elmur_klpo.manager import CategoricalManager
from elmur_klpo.returns import post_write_rtg
from elmur_klpo.toy_env import ToyEnvironment


@dataclass(frozen=True)
class TrajectoryBatch:
    """All complete action sequences, plus the full unmerged prefix tree."""

    actions: Tensor  # [N, H]
    states: Tensor  # [N, H, F], before each write
    rewards: Tensor  # [N, H+1], including terminal read, with no terminal action
    memories: Tensor  # [N, H+1, B]
    node_states: tuple[Tensor, ...]  # [U**t, F] at each decision time
    node_rewards: tuple[Tensor, ...]  # [U**t], including terminal time

    @property
    def returns(self) -> Tensor:
        return post_write_rtg(self.rewards)


@dataclass(frozen=True)
class BehaviorEvaluation:
    """Frozen sampler distribution, state visitation, and conditional moments."""

    log_probs: Tensor  # [N, H, U]
    trajectory_probs: Tensor  # [N]
    node_log_probs: tuple[Tensor, ...]
    visitation: tuple[Tensor, ...]  # probability of reaching each prefix node
    q_values: tuple[Tensor, ...]  # Q^q(x,u), excluding current reward
    values: tuple[Tensor, ...]  # V^q(x)
    advantages: tuple[Tensor, ...]  # A^q(x,u)


@torch.no_grad()
def enumerate_trajectories(env: ToyEnvironment) -> TrajectoryBatch:
    """Enumerate U**H sequences; transitions and recorded states are detached."""
    width, horizon = len(env.action_space), env.decisions
    actions = torch.tensor(list(itertools.product(range(width), repeat=horizon)))
    memory_levels = [env.initial_memory(batch_size=1)]
    state_levels, reward_levels = [], []
    for time in range(horizon):
        memory = memory_levels[-1]
        state_levels.append(env.features(time, memory))
        reward_levels.append(env.reader_log_likelihood(time, memory))
        child_actions = torch.arange(width).repeat(len(memory))
        children = memory.repeat_interleave(width, dim=0)
        memory_levels.append(env.transition(time, children, child_actions))
    reward_levels.append(env.reader_log_likelihood(horizon, memory_levels[-1]))

    prefix_ids = torch.zeros(len(actions), dtype=torch.long)
    states, rewards, memories = [], [], []
    for time in range(horizon):
        states.append(state_levels[time][prefix_ids])
        rewards.append(reward_levels[time][prefix_ids])
        memories.append(memory_levels[time][prefix_ids])
        prefix_ids = prefix_ids * width + actions[:, time]
    rewards.append(reward_levels[-1][prefix_ids])
    memories.append(memory_levels[-1][prefix_ids])
    return TrajectoryBatch(
        actions=actions,
        states=torch.stack(states, dim=1),
        rewards=torch.stack(rewards, dim=1),
        memories=torch.stack(memories, dim=1),
        node_states=tuple(state_levels),
        node_rewards=tuple(reward_levels),
    )


def trajectory_probabilities(
    policy: CategoricalManager, batch: TrajectoryBatch
) -> Tensor:
    """Differentiable product of the policy's conditional action probabilities."""
    log_probs = policy.action_log_probs(batch.states.detach(), batch.actions)
    return log_probs.sum(-1).exp()


@torch.no_grad()
def evaluate_behavior(
    behavior: CategoricalManager, batch: TrajectoryBatch
) -> BehaviorEvaluation:
    """Sum all future branches under q to obtain exact Q^q, V^q, and A^q.

    Q_t(x,u) = r_{t+1}(x') + V_{t+1}(x'), with V_H=0. This sums the full
    remaining subtree, independently of the complete-trajectory surrogate.
    """
    node_logs = tuple(behavior.log_probs(states) for states in batch.node_states)
    node_probs = tuple(logs.exp() for logs in node_logs)
    width = node_logs[0].shape[-1]
    future_values = torch.zeros_like(batch.node_rewards[-1])
    q_values, values, advantages = [], [], []
    for time in reversed(range(len(node_logs))):
        conditional = (batch.node_rewards[time + 1] + future_values).reshape(-1, width)
        value = (node_probs[time] * conditional).sum(-1)
        q_values.append(conditional)
        values.append(value)
        advantages.append(conditional - value[:, None])
        future_values = value

    visitation = [torch.ones(1, dtype=node_logs[0].dtype)]
    for probs in node_probs[:-1]:
        visitation.append((visitation[-1][:, None] * probs).flatten())
    log_probs = behavior.log_probs(batch.states)
    chosen = log_probs.gather(-1, batch.actions[..., None]).squeeze(-1)
    return BehaviorEvaluation(
        log_probs=log_probs,
        trajectory_probs=chosen.sum(-1).exp(),
        node_log_probs=node_logs,
        visitation=tuple(visitation),
        q_values=tuple(reversed(q_values)),
        values=tuple(reversed(values)),
        advantages=tuple(reversed(advantages)),
    )


def exact_expected_return(policy: CategoricalManager, batch: TrajectoryBatch) -> Tensor:
    """J(theta): full trajectory returns weighted by differentiable p(tau)."""
    return (
        trajectory_probabilities(policy, batch) * batch.rewards.detach().sum(-1)
    ).sum()


def regression_objective(
    policy: CategoricalManager,
    batch: TrajectoryBatch,
    behavior: BehaviorEvaluation,
    beta: float,
) -> Tensor:
    """Explicit L_reg, using fixed q visitation and exact conditional advantages.

    Sum over all prefix states and all actions. This is a scalar objective with
    an ordinary autograd gradient, not the stop-gradient backward surrogate.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    losses = []
    for states, log_q, state_mass, advantage in zip(
        batch.node_states,
        behavior.node_log_probs,
        behavior.visitation,
        behavior.advantages,
        strict=True,
    ):
        log_p = policy.log_probs(states.detach())
        log_q = log_q.detach()
        probs_q = log_q.exp()
        ell = log_p - log_q
        kl = (probs_q * (log_q - log_p)).sum(-1, keepdim=True)
        residual = advantage.detach() - beta * (ell + kl)
        joint_mass = state_mass.detach()[:, None] * probs_q
        losses.append((joint_mass * residual.square()).sum() / (2 * beta))
    return torch.stack(losses).sum()


def population_surrogate(
    policy: CategoricalManager,
    batch: TrajectoryBatch,
    behavior: BehaviorEvaluation,
    beta: float,
    *,
    full_return: bool = False,
) -> Tensor:
    """Exact q-weighted expectation of the sample backward surrogate."""
    returns = batch.returns
    if full_return:
        returns = batch.rewards.sum(-1, keepdim=True).expand_as(returns)
    losses = backward_surrogate(
        policy.log_probs(batch.states.detach()),
        behavior.log_probs,
        batch.actions,
        returns,
        beta,
    )
    return (behavior.trajectory_probs.detach() * losses).sum()
