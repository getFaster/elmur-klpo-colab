"""Adapter around the pinned authors' ELMUR model; no replacement encoder."""

from torch import Tensor, nn

from elmur_klpo.vendor.elmur.layers.memory import MemoryState
from elmur_klpo.vendor.elmur.model import ELMURModel


class ELMURReader(nn.Module):
    """One upstream layer, state-only tokens, with exposed frozen write candidates.

    Upstream tok2mem produces [N, slots, width] slot-conditioned vectors.
    Expose them detached so each categorical write uses its selected slot's
    candidate. Native LRU retains its original new-vector blending convention.
    """

    def __init__(self, slots: int = 2, dimension: int = 32) -> None:
        super().__init__()
        self.model = ELMURModel(
            state_dim=4,
            act_dim=4,
            d_model=dimension,
            d_ff=dimension * 2,
            n_layer=1,
            n_head=2,
            memory_size=slots,
            env_name="tmaze",
            sequence_format="s",
            memory_init_std=0.0,
            dropout=0.0,
            dropatt=0.0,
            memory_dropout=0.0,
            lru_blend_alpha=0.79,
            use_moe=False,
            max_seq_len=2048,
        )
        self._candidate: Tensor | None = None
        self.model.layers[0].memory_ffn.register_forward_hook(self._capture)

    def _capture(self, module: nn.Module, inputs: tuple, output: tuple) -> None:
        self._candidate = output[0].detach()

    def initial(self, batch_size: int) -> MemoryState:
        return self.model.init_memory(batch_size, next(self.parameters()).device)[0]

    def step(
        self, observations: Tensor, memory: MemoryState, time: int
    ) -> tuple[Tensor, Tensor, Tensor, MemoryState]:
        """Read pre-write memory; expose [N, slots, width] candidates and LRU."""
        output = self.model(
            observations[:, None],
            None,
            observations.new_ones(observations.shape[0], 1, 1),
            None,
            None,
            memory_states=[memory],
            pos_offset=time,
        )
        context = self.model.state_encoder(observations).detach()
        return (
            output["logits"][:, 0],
            context,
            self._candidate,
            output["memory_states"][0],
        )

    def freeze(self) -> "ELMURReader":
        self.eval().requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        return self
