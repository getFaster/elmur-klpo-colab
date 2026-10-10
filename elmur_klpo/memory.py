"""Stable categorical action IDs and detached, out-of-place scalar memory writes."""

from dataclasses import dataclass

import torch
from torch import Tensor

MERGE_COEFFICIENTS = (0.0, 0.25, 0.5, 0.8, 0.95)


@dataclass(frozen=True)
class ActionSpace:
    """Slot-major IDs: 5*k+i -> (k, alpha[i]); final ID -> global no-op."""

    slots: int = 2

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError("Memory must have at least one slot")

    def __len__(self) -> int:
        return len(MERGE_COEFFICIENTS) * self.slots + 1

    @property
    def noop_id(self) -> int:
        return len(self) - 1

    def decode(self, action_id: int) -> tuple[int | None, float]:
        if not 0 <= action_id < len(self):
            raise ValueError(f"Invalid action ID: {action_id}")
        if action_id == self.noop_id:
            return None, 1.0
        slot, coefficient = divmod(action_id, len(MERGE_COEFFICIENTS))
        return slot, MERGE_COEFFICIENTS[coefficient]

    def actions(self) -> tuple[tuple[int | None, float], ...]:
        return tuple(self.decode(i) for i in range(len(self)))


@torch.no_grad()
def update_memory(
    memory: Tensor,
    candidate: Tensor | float,
    action_ids: Tensor | int,
    action_space: ActionSpace,
) -> Tensor:
    """Write scalar candidates into [..., B] memory; detach every transition.

    Candidates and action IDs broadcast over the memory's leading dimensions.
    The input is never mutated, including for no-op actions.
    """
    if memory.ndim < 1 or memory.shape[-1] != action_space.slots:
        raise ValueError("Memory's last dimension must match the slot count")
    if not memory.is_floating_point():
        raise ValueError("Memory must have a floating-point dtype")
    ids = torch.as_tensor(action_ids, device=memory.device)
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Action IDs must be integers")
    ids = torch.broadcast_to(ids, memory.shape[:-1])
    if ((ids < 0) | (ids >= len(action_space))).any():
        raise ValueError("Action IDs are outside the action space")
    z = torch.as_tensor(candidate, dtype=memory.dtype, device=memory.device)
    z = torch.broadcast_to(z, memory.shape[:-1])
    slots = ids // len(MERGE_COEFFICIENTS)
    alphas = memory.new_tensor(MERGE_COEFFICIENTS)[ids % len(MERGE_COEFFICIENTS)]
    selected = (
        torch.arange(action_space.slots, device=memory.device) == slots[..., None]
    )
    blended = alphas[..., None] * memory + (1 - alphas[..., None]) * z[..., None]
    return torch.where(selected, blended, memory).detach()
