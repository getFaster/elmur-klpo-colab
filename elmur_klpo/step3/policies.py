"""Task-independent initialization and controlled slot/coefficient ablations."""

import math

import torch
from torch import Tensor

from elmur_klpo.manager import CategoricalManager
from elmur_klpo.memory import MERGE_COEFFICIENTS, ActionSpace
from elmur_klpo.training_memory import write_vectors
from elmur_klpo.vendor.elmur.layers.memory import MemoryState

INITIAL_ALPHA_PROBS = (0.1, 0.1, 0.6, 0.1, 0.1)
FIXED_GRID = (0.0, 0.1, 0.2, 0.5, 0.6, 0.8, 0.95, 1.0)


def retention_from_native(new_weight: float) -> float:
    """Convert upstream new-candidate weight lambda to old-memory retention."""
    if not 0 <= new_weight <= 1:
        raise ValueError("Native blending weight must lie in [0,1]")
    return 1 - new_weight


def initial_probabilities(slots: int) -> Tensor:
    """Return float64 joint probabilities [5*slots+1], including global no-op."""
    space = ActionSpace(slots)
    probabilities = torch.tensor(INITIAL_ALPHA_PROBS, dtype=torch.float64)
    return torch.cat(
        (
            probabilities.repeat(space.slots) * (0.98 / slots),
            torch.tensor([0.02], dtype=torch.float64),
        )
    )


class BlendingManager(CategoricalManager):
    """Linear policy with an intercept in feature zero; all other weights start zero.

    Coefficient-only shares one distribution across every slot. Slot-only has
    fixed alpha=0.5, no no-op, and uniform initial slots. These restrictions
    hold in sampling, replay, training, and greedy evaluation.
    """

    def __init__(self, state_dim: int, slots: int, variant: str = "full") -> None:
        if variant not in {"full", "slot_only", "coefficient_only"}:
            raise ValueError("Unknown manager variant")
        super().__init__(state_dim, ActionSpace(slots), scale=0, dtype=torch.float32)
        self.variant = variant
        with torch.no_grad():
            self.weight[:, 0].copy_(initial_probabilities(slots).log())
        # Dedicated smaller parameter tensors avoid redundant non-identifiable
        # joint weights in the constrained variants.
        if variant != "full":
            rows = slots if variant == "slot_only" else 6
            self.weight = torch.nn.Parameter(torch.zeros(rows, state_dim))
            if variant == "coefficient_only":
                self.weight.data[:, 0] = torch.tensor(
                    (*[0.98 * p for p in INITIAL_ALPHA_PROBS], 0.02)
                ).log()

    def log_probs(self, states: Tensor) -> Tensor:
        logits = states @ self.weight.T
        if self.variant == "full":
            return logits.log_softmax(-1)
        slots = self.action_space.slots
        if self.variant == "coefficient_only":
            coeff = logits.log_softmax(-1)
            writes = (coeff[..., None, :5] - math.log(slots)).expand(
                *coeff.shape[:-1], slots, 5
            )
            return torch.cat((writes.flatten(-2), coeff[..., 5:]), -1)
        # Finite sentinel keeps the shared Step 2 KL diagnostics well-defined;
        # exp(-1e9) is exactly zero in float32.
        log_slots = logits.log_softmax(-1)
        joint = logits.new_full((*logits.shape[:-1], slots, 5), -1e9)
        joint[..., 2] = log_slots
        return torch.cat(
            (joint.flatten(-2), logits.new_full((*logits.shape[:-1], 1), -1e9)), -1
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.log_probs(states)


@torch.no_grad()
def transition(
    memory: MemoryState,
    candidate: Tensor,
    ids: Tensor,
    time: int,
    *,
    fixed_alpha: float | None = None,
) -> MemoryState:
    """Detached [N,S,D] writes; no-op preserves positions, alpha=1 refreshes age.

    Fixed sweeps use independent continuous retention, without quantizing onto
    the learned action grid. For these sweeps ids encode the uniformly drawn slot.
    """
    if fixed_alpha is None:
        vec = write_vectors(memory.vec, candidate, ids)
    else:
        if not 0 <= fixed_alpha <= 1 or candidate.shape != memory.vec.shape:
            raise ValueError("Invalid fixed coefficient or candidate shape")
        selected = (
            torch.arange(memory.vec.shape[1], device=ids.device)[None]
            == (ids // 5)[:, None]
        )
        vec = torch.where(
            selected[..., None],
            fixed_alpha * memory.vec + (1 - fixed_alpha) * candidate,
            memory.vec,
        )
    selected = (
        torch.arange(memory.vec.shape[1], device=ids.device)[None]
        == (ids // len(MERGE_COEFFICIENTS))[:, None]
    )
    return MemoryState(vec.detach(), torch.where(selected, time, memory.pos))
