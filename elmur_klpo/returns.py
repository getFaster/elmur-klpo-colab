"""Return timing for read-before-write episodes."""

import torch
from torch import Tensor


def post_write_rtg(rewards: Tensor) -> Tensor:
    """Map rewards [..., H+1] to H returns G_t = sum_{j=t+1}^H r_j.

    There is no manager action after the terminal reward. In particular r_t is
    excluded from the return assigned to action u_t. No padding is used here.
    """
    if rewards.ndim < 1 or rewards.shape[-1] < 2:
        raise ValueError("Need at least an initial and a terminal reader reward")
    future = rewards[..., 1:]
    return torch.flip(torch.flip(future, dims=(-1,)).cumsum(-1), dims=(-1,))
