"""Single-update KLPO and on-policy RTG-REINFORCE, with resumable state."""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from elmur_klpo.klpo import backward_surrogate
from elmur_klpo.manager import CategoricalManager
from elmur_klpo.rollout import Rollout


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def update(
    manager: CategoricalManager,
    optimizer: torch.optim.Optimizer,
    batch: Rollout,
    algorithm: str,
    beta: float = 0.1,
    clip: float = 5.0,
) -> dict:
    """Average episode sums; exactly one optimizer step on a fresh rollout.

    REINFORCE requires p=q before the update. KLPO's scalar is a gradient
    surrogate for frozen-sampler regression, not an expected-return objective.
    """
    start = time.perf_counter()
    if batch.method != "learned":
        raise ValueError("Optimizer requires a learned-policy rollout")
    if not batch.valid.any():
        raise ValueError("Rollout has no valid manager decisions")
    log_p = manager.log_probs(batch.states.detach())
    if not torch.isfinite(log_p).all() or not torch.isfinite(batch.returns).all():
        raise FloatingPointError("Nonfinite rollout or policy")
    returns = torch.where(batch.valid, batch.returns, 0).detach()
    if algorithm == "klpo":
        # Zero both feedback and ratio on invalid timesteps before the verified estimator.
        log_q = torch.where(batch.valid[..., None], batch.log_q, log_p.detach())
        loss = backward_surrogate(log_p, log_q, batch.actions, returns, beta).mean()
    elif algorithm == "reinforce":
        if not torch.allclose(
            log_p.detach()[batch.valid], batch.log_q[batch.valid], atol=1e-6, rtol=1e-6
        ):
            raise ValueError("REINFORCE requires fresh on-policy trajectories")
        chosen = log_p.gather(-1, batch.actions[..., None]).squeeze(-1)
        loss = -(returns * chosen).sum(-1).mean()
    else:
        raise ValueError("Expected klpo or reinforce")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        manager.parameters(), clip, error_if_nonfinite=True
    )
    optimizer.step()
    with torch.no_grad():
        if not all(
            torch.isfinite(parameter).all() for parameter in manager.parameters()
        ):
            raise FloatingPointError(
                "Nonfinite manager parameters after optimizer step"
            )
        new_log_p = manager.log_probs(batch.states)
        q = batch.q_probs
        forward = (q * (batch.log_q - new_log_p)).sum(-1)
        reverse = (new_log_p.exp() * (new_log_p - batch.log_q)).sum(-1)
    return {
        "surrogate_loss": float(loss.detach()),
        "gradient_norm": float(norm),
        "kl_q_p": float(forward[batch.valid].mean()),
        "kl_p_q": float(reverse[batch.valid].mean()),
        "update_seconds": time.perf_counter() - start,
        "numerical_failures": 0,
    }


def save_checkpoint(
    path: Path,
    manager: CategoricalManager,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    extra: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "manager": manager.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "extra": extra,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path, manager: CategoricalManager, optimizer: torch.optim.Optimizer
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    manager.load_state_dict(checkpoint["manager"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    if checkpoint["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return checkpoint
