"""Vector memory transitions for Step 2, separate from scalar verification."""

import torch
from torch import Tensor

from elmur_klpo.memory import MERGE_COEFFICIENTS, ActionSpace


@torch.no_grad()
def write_vectors(memory: Tensor, candidate: Tensor, ids: Tensor) -> Tensor:
    """Write each chosen slot's [N, B, D] candidate without history gradients."""
    if memory.ndim != 3 or candidate.shape != memory.shape:
        raise ValueError("Expected memory and slot-conditioned candidates [N,B,D]")
    space = ActionSpace(memory.shape[-2])
    if ids.shape != memory.shape[:1] or ids.dtype != torch.long:
        raise ValueError("Expected integer actions [N]")
    if ((ids < 0) | (ids >= len(space))).any():
        raise ValueError("Invalid action")
    slots = (ids // 5).clamp_max(space.slots - 1)
    index = slots[:, None, None].expand(-1, 1, memory.shape[-1])
    old = memory.gather(1, index).squeeze(1)
    selected_candidate = candidate.gather(1, index).squeeze(1)
    alpha = memory.new_tensor((*MERGE_COEFFICIENTS, 1.0))[
        torch.where(ids == space.noop_id, 5, ids % 5)
    ][:, None]
    value = torch.where(
        (ids == space.noop_id)[:, None],
        old,
        alpha * old + (1 - alpha) * selected_candidate,
    )
    return memory.detach().clone().scatter_(1, index, value[:, None]).detach()
