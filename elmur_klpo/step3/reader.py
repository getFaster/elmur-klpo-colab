"""Legacy single-token adapter, retained for historical manager diagnostics only."""

import torch
from torch import Tensor, nn

from elmur_klpo.reader import ELMURReader
from elmur_klpo.vendor.elmur.layers.memory import MemoryState


class TaskReader(ELMURReader):
    """Historical adapter; repaired preparation uses FixedLRUReader."""

    def __init__(self, task: dict, config: dict) -> None:
        robot = task["name"] == "remember_color"
        super().__init__(
            config["slots"],
            config["dimension"],
            state_dim=6,
            action_dim=8 if robot else 4,
            env_name="mikasa_robo_RememberColor3-v0"
            if robot
            else "popgym-AutoencodeEasy",
            native_lambda=config.get(
                "native_lambda", 0.40683708068133906 if robot else 0.8
            ),
            memory_init_std=config.get("memory_init_std", 0.0),
            trainable_candidates=True,
        )
        self.action_kind = "continuous" if robot else "discrete"
        self.continuous_output = config.get("continuous_output", "bounded")
        if self.continuous_output not in ("bounded", "unbounded"):
            raise ValueError("Continuous output must be bounded or unbounded")
        if robot and self.continuous_output == "unbounded":
            # Adapt the output locally, preserving the vendored implementation.
            self.model.head[-1] = nn.Identity()
        self.action_std = config.get("action_std", 0.2)
        self.observation_scale = config.get("observation_scale", 1.0)
        if self.observation_scale <= 0:
            raise ValueError("Observation scale must be positive")
        if self.action_std <= 0:
            raise ValueError("Action standard deviation must be positive")

    def step(
        self, observations: Tensor, memory: MemoryState, time: int
    ) -> tuple[Tensor, Tensor, Tensor, MemoryState]:
        """Apply checkpoint-configured scaling consistently to recorded and live RGB."""
        return super().step(observations / self.observation_scale, memory, time)

    def task_action(self, logits: Tensor, greedy: bool = True) -> tuple[Tensor, Tensor]:
        """Return [N] discrete or [N,8] configured robot actions and log probabilities."""
        if self.action_kind == "discrete":
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(-1) if greedy else distribution.sample()
            return action, distribution.log_prob(action)
        if self.continuous_output == "unbounded":
            distribution = torch.distributions.Normal(logits, self.action_std)
            action = logits if greedy else distribution.sample()
            return action, distribution.log_prob(action).sum(-1)
        # The upstream continuous decoder already applies tanh. Sampling in its
        # inverse coordinates gives a bounded, correctly scored stochastic policy.
        mean = logits.clamp(-1 + 1e-6, 1 - 1e-6).atanh()
        distribution = torch.distributions.Normal(mean, self.action_std)
        latent = mean if greedy else distribution.sample()
        action = latent.tanh()
        log_prob = (
            distribution.log_prob(latent) - torch.log(1 - action.square() + 1e-6)
        ).sum(-1)
        return action, log_prob
