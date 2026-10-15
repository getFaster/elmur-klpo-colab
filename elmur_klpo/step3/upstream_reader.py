"""Task-specific upstream ELMUR with independent layer-local fixed-LRU memory."""

import torch
from torch import Tensor, nn

from elmur_klpo.vendor.elmur.layers.memory import MemoryState
from elmur_klpo.vendor.elmur.model import ELMURModel


def detach_memories(memories: list[MemoryState]) -> list[MemoryState]:
    """Preserve [layer][B,S,D] content/positions; cut all section history graphs."""
    return [MemoryState(m.vec.detach(), m.pos.detach()) for m in memories]


class FixedLRUReader(nn.Module):
    """Unmodified pinned model; only observable states enter state-only tokens."""

    def __init__(self, task: dict, config: dict) -> None:
        super().__init__()
        if config.get("sequence_format") != "s" or config.get("use_shared_memory"):
            raise ValueError("Requires observation-only tokens and layer-local memory")
        if not all(
            config.get(key)
            for key in (
                "use_lru",
                "detach_memory",
                "use_tok2mem",
                "use_mem2tok",
                "use_causal_self_attn_mask",
            )
        ):
            raise ValueError(
                "Requires native read/write, causal mask, LRU, and detached memory"
            )
        self.task_name = task["name"]
        self.model_config_smoothing = config["label_smoothing"]
        self.model = ELMURModel(**config)
        self.observation_scale = 255.0 if self.task_name == "remember_color" else 1.0

    def initial(self, batch_size: int) -> list[MemoryState]:
        """Fresh random-normal [B,S,D] and -1 positions independently per layer."""
        return self.model.init_memory(batch_size, next(self.parameters()).device)

    def forward(
        self, observations: Tensor, memories: list[MemoryState], offset: int = 0
    ) -> dict:
        """Read [B,T,6] or raw RGB [B,T,6,128,128]; labels/RTG never accepted."""
        device = next(self.parameters()).device
        states = (
            observations.to(device=device, dtype=torch.float32) / self.observation_scale
        )
        batch, length = states.shape[:2]
        # Pinned model encodes RTG unconditionally but sequence 's' discards it.
        # A constant prevents any supervision/return leakage through this API.
        return self.model(
            states,
            None,
            states.new_zeros(batch, length, 1),
            None,
            None,
            memory_states=memories,
            pos_offset=offset,
        )

    def freeze(self) -> "FixedLRUReader":
        self.eval().requires_grad_(False)
        for p in self.parameters():
            p.grad = None
        return self
