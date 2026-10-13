"""Reproducible offline demonstrations from RATE's actual T-Maze generator."""

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from elmur_klpo.vendor.tmaze_dataset import generate_trajectory


@dataclass
class Demonstrations:
    observations: Tensor
    actions: Tensor
    valid: Tensor
    episode_ids: Tensor

    def select(self, indices: Tensor) -> "Demonstrations":
        return Demonstrations(*(value[indices] for value in self.__dict__.values()))


def tmaze_demonstrations(
    count: int, horizon: int, seed: int, device: str = "cpu"
) -> Demonstrations:
    """Generate variable corridors, both cues; drop the post-terminal observation.

    Every episode has a cue outside the final reader context, including the
    shortest corridor. Splits use independent noise seeds and corridor draws.
    """
    if count < 2 or horizon < 5:
        raise ValueError("Need at least two episodes and horizon >=5")
    rng = random.Random(seed)
    observations = torch.zeros(count, horizon, 4, device=device)
    actions = torch.zeros(count, horizon, dtype=torch.long, device=device)
    valid = torch.zeros(count, horizon, dtype=torch.bool, device=device)
    for index in range(count):
        length = rng.randint(max(3, horizon // 2), horizon - 1)
        obs, act, _, done = generate_trajectory(
            length + 2, length, 1, True, index % 2, seed * count + index
        )
        if not done[-1]:
            raise RuntimeError("Expert trajectory did not terminate")
        size = len(act)
        observations[index, :size] = torch.as_tensor(np.copy(obs[:size]), device=device)
        actions[index, :size] = torch.as_tensor(act, device=device)
        valid[index, :size] = True
    return Demonstrations(
        observations, actions, valid, torch.arange(count, device=device) + seed * count
    )
