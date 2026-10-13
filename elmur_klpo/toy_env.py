"""Deterministic cue/distractor environment with a frozen binary reader."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from elmur_klpo.memory import ActionSpace
from elmur_klpo.training_memory import write_vectors


@dataclass(frozen=True)
class ToyEnvironment:
    """Three writes, then a terminal query. Time and memory form a Markov state.

    f(c)=c; candidates are cue +1, then distractors -1 and -0.5. The expert
    label is always 1. At non-query steps the frozen reader predicts 1/2,
    independently of memory. At the final query it predicts
    sigmoid(4 * (sum(memory) - 0.5)). Thus no immediate reward favors the cue.
    """

    candidates: tuple[float, ...] = (1.0, -1.0, -0.5)
    action_space: ActionSpace = ActionSpace(2)
    reader_gain: float = 4.0
    reader_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("Need at least one effective manager decision")

    @property
    def decisions(self) -> int:
        return len(self.candidates)

    @property
    def state_dim(self) -> int:
        return 3 + self.action_space.slots

    def initial_memory(self, batch_size: int | None = None) -> Tensor:
        shape = (self.action_space.slots,)
        if batch_size is not None:
            shape = (batch_size, *shape)
        return torch.zeros(shape, dtype=torch.float64)

    def features(self, time: int, memory: Tensor) -> Tensor:
        """[intercept, time/H, c_t, m_1, ..., m_B]; includes the timestep."""
        if not 0 <= time < self.decisions:
            raise ValueError("There is no manager state at the terminal query")
        prefix = memory.new_tensor([1.0, time / self.decisions, self.candidates[time]])
        prefix = prefix.expand(*memory.shape[:-1], 3)
        return torch.cat((prefix, memory), dim=-1).detach()

    @torch.no_grad()
    def reader_log_likelihood(self, time: int, memory: Tensor) -> Tensor:
        if not 0 <= time <= self.decisions:
            raise ValueError("Reader time is outside the episode")
        if time == self.decisions:
            logit = self.reader_gain * (memory.sum(-1) - self.reader_threshold)
        else:
            logit = torch.zeros_like(memory[..., 0])
        return F.logsigmoid(logit)

    def transition(self, time: int, memory: Tensor, actions: Tensor | int) -> Tensor:
        if not 0 <= time < self.decisions:
            raise ValueError("There is no terminal manager action")
        if memory.ndim < 1 or memory.shape[-1] != self.action_space.slots:
            raise ValueError("Memory's last dimension must match the slot count")
        # Represent the fixed scalar toy observations as width-one slot vectors.
        vectors = memory.reshape(-1, self.action_space.slots, 1)
        candidates = torch.full_like(vectors, self.candidates[time])
        ids = torch.as_tensor(actions, device=memory.device)
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Action IDs must be integers")
        ids = torch.broadcast_to(ids, memory.shape[:-1]).reshape(-1).long()
        return write_vectors(vectors, candidates, ids).reshape(memory.shape)
