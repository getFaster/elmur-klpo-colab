"""Reusable RTG-KLPO gradient surrogate. Its scalar value is not L_reg."""

import math

from torch import Tensor


def score_center(log_p: Tensor, q_probs: Tensor) -> Tensor:
    """C_theta(x) = sum_u q(u|x) log p_theta(u|x), keeping p differentiable."""
    if log_p.shape != q_probs.shape:
        raise ValueError(
            "Learner log probabilities and behavior probabilities must match"
        )
    return (q_probs.detach() * log_p).sum(-1)


def backward_surrogate(
    log_p: Tensor,
    log_q: Tensor,
    actions: Tensor,
    returns: Tensor,
    beta: float,
) -> Tensor:
    """Return one summed loss per trajectory, with action dimension last.

    log_p, log_q: [..., H, U]; actions, returns: [..., H]. Inputs must describe
    detached recorded states. Only current-policy log probabilities receive
    gradients; h, behavior probabilities, and returns are always detached.
    The caller chooses how to average trajectories (or sum exact q weights).
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if log_p.ndim < 2 or log_p.shape != log_q.shape:
        raise ValueError("Expected matching [..., H, U] log-probability arrays")
    if actions.shape != log_p.shape[:-1] or returns.shape != actions.shape:
        raise ValueError("Actions and returns must have shape [..., H]")
    log_q = log_q.detach()
    chosen_p = log_p.gather(-1, actions[..., None]).squeeze(-1)
    chosen_q = log_q.gather(-1, actions[..., None]).squeeze(-1)
    h = (returns - beta * (chosen_p - chosen_q)).detach()
    centered_log_p = chosen_p - score_center(log_p, log_q.exp())
    return -(h * centered_log_p).sum(-1)
