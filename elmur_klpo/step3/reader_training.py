"""One optimizer step per section, detached memories, exact persistent counters."""

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from elmur_klpo.step3.reader_data import ReaderBatch
from elmur_klpo.step3.upstream_reader import FixedLRUReader, detach_memories
from elmur_klpo.vendor.elmur.layers.memory import MemoryState
from elmur_klpo.vendor.rate_lr_scheduler import (
    LearningRateScheduler,
)


class ReaderScheduler:
    """Pinned scheduler units/order, including its minibatch-based epoch switch."""

    def __init__(
        self, optimizer: torch.optim.Optimizer, settings: dict, batches_per_epoch: int
    ) -> None:
        self.optimizer = optimizer
        self.settings = settings
        self.tokens = 0
        self.minibatches = 0
        self.decay = False
        self.reference = LearningRateScheduler(
            {"training": settings}, range(batches_per_epoch)
        )
        self.scheduler = (
            None
            if settings["use_cosine_decay"]
            else self.reference.make_warmup_scheduler(optimizer)
        )

    def after_step(self, targets: Tensor) -> None:
        """Advance after optimizer.step; upstream counts nonnegative target elements."""
        if self.scheduler is not None:
            self.scheduler.step()
        self.tokens += int((targets >= 0).sum())
        if self.settings["use_cosine_decay"]:
            self.reference.update_learning_rate(self.optimizer, self.tokens)

    def end_minibatch(self, epoch_end: bool) -> None:
        self.minibatches += 1
        if (
            epoch_end
            and not self.settings["use_cosine_decay"]
            and not self.decay
            and self.minibatches >= self.settings["warmup_steps"]
        ):
            self.scheduler = self.reference.make_decay_scheduler(self.optimizer)
            self.decay = True

    def state_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "minibatches": self.minibatches,
            "decay": self.decay,
            "scheduler": None
            if self.scheduler is None
            else self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.tokens, self.minibatches, self.decay = (
            state[k] for k in ("tokens", "minibatches", "decay")
        )
        if self.decay:
            # Constructing LinearLR touches LR; preserve the restored optimizer LR.
            rates = [g["lr"] for g in self.optimizer.param_groups]
            self.scheduler = self.reference.make_decay_scheduler(self.optimizer)
            for group, rate in zip(self.optimizer.param_groups, rates, strict=True):
                group["lr"] = rate
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])


def make_optimizer(
    reader: FixedLRUReader, settings: dict
) -> tuple[torch.optim.Optimizer, ReaderScheduler]:
    """All upstream parameters, including every layer's writer MoE, are members."""
    optimizer = torch.optim.AdamW(
        reader.parameters(),
        lr=settings["learning_rate"],
        betas=(settings["beta_1"], settings["beta_2"]),
        weight_decay=settings["weight_decay"],
    )
    scheduler = ReaderScheduler(optimizer, settings, settings["batches_per_epoch"])
    return optimizer, scheduler


def action_loss(
    logits: Tensor, targets: Tensor, valid: Tensor, task: str, smoothing: float
) -> Tensor:
    """Masked CE on PLAY only or MSE on raw 8D expert controls; zero if no labels."""
    if not valid.any():
        return logits.sum() * 0
    if task == "remember_color":
        return F.mse_loss(logits[valid], targets[valid])
    return F.cross_entropy(
        logits[valid], targets[valid].long(), label_smoothing=smoothing
    )


def section_update(
    reader: FixedLRUReader,
    optimizer: torch.optim.Optimizer,
    scheduler: ReaderScheduler,
    section: ReaderBatch,
    memories: list[MemoryState],
    offset: int,
    settings: dict,
    progress: dict,
) -> tuple[list[MemoryState], dict]:
    """One independent graph/update; outgoing memory is detached before reuse."""
    reader.train().requires_grad_(True)
    output = reader(section.observations, detach_memories(memories), offset)
    logits = output["logits"]
    targets = section.targets.to(logits.device)
    mask = (section.valid & section.action_valid).to(logits.device)
    prediction = action_loss(
        logits, targets, mask, reader.task_name, reader.model_config_smoothing
    )
    auxiliary = output["aux_loss"]
    loss = prediction if auxiliary is None else prediction + auxiliary
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    writer_ids = {
        id(p) for layer in reader.model.layers for p in layer.memory_ffn.parameters()
    }
    optimizer_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    if not writer_ids <= optimizer_ids:
        raise AssertionError("Writer MoE missing from optimizer")
    writer_grad = (
        sum(
            float(p.grad.detach().square().sum())
            for layer in reader.model.layers
            for p in layer.memory_ffn.parameters()
            if p.grad is not None
        )
        ** 0.5
    )
    norm = torch.nn.utils.clip_grad_norm_(
        reader.parameters(), settings["grad_norm_clip"], error_if_nonfinite=True
    )
    used_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    progress["completed_optimizer_steps"] += 1
    scheduler.after_step(targets)
    if not all(torch.isfinite(p).all() for p in reader.parameters()):
        raise FloatingPointError("Nonfinite reader after update")
    return detach_memories(output["memory_states"]), {
        "action_loss": float(prediction.detach()),
        "moe_aux_loss": 0.0 if auxiliary is None else float(auxiliary.detach()),
        "loss": float(loss.detach()),
        "gradient_norm": float(norm),
        "writer_gradient_norm": writer_grad,
        "writer_optimizer_members": True,
        "learning_rate_used": used_lr,
        "next_learning_rate": optimizer.param_groups[0]["lr"],
        "valid_action_targets": int(mask.sum()),
        "valid_transitions": int(section.valid.sum()),
        "sample_token_presentations": int(section.valid.numel()),
        "model_forward_passes": 1,
        "sequence_forward_presentations": len(targets),
    }


def effective_epochs(steps: int, train_size: int, batch_size: int) -> float:
    return steps / (math.ceil(train_size / batch_size) * 3)
