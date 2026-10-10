"""A linear softmax policy with explicit, strictly positive joint action probabilities."""

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from elmur_klpo.memory import ActionSpace


class CategoricalManager(nn.Module):
    """Small manager over joint (slot, alpha) actions, including one global no-op.

    States are feature vectors (including an intercept). The default float64 is
    intentional for verification; later callers can choose a different dtype.
    """

    def __init__(
        self,
        state_dim: int = 5,
        action_space: ActionSpace | None = None,
        *,
        seed: int = 0,
        scale: float = 0.2,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.action_space = action_space or ActionSpace()
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(
            scale
            * torch.randn(
                len(self.action_space), state_dim, generator=generator, dtype=dtype
            )
        )

    def forward(self, states: Tensor) -> Tensor:
        return states @ self.weight.T

    def log_probs(self, states: Tensor) -> Tensor:
        return self(states).log_softmax(dim=-1)

    def probs(self, states: Tensor) -> Tensor:
        return self.log_probs(states).exp()

    def distribution(self, states: Tensor) -> Categorical:
        return Categorical(logits=self(states))

    def action_log_probs(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.log_probs(states).gather(-1, actions[..., None]).squeeze(-1)

    def sample(self, states: Tensor) -> tuple[Tensor, Tensor]:
        distribution = self.distribution(states)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions)

    def frozen_copy(self) -> "CategoricalManager":
        """Snapshot the behavior policy without sharing storage or an autograd graph."""
        from copy import deepcopy

        return deepcopy(self).requires_grad_(False)
