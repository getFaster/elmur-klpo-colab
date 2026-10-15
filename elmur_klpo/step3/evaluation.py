"""Deterministic held-out evaluation, readiness gates, and controlled baselines."""

import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

from elmur_klpo.step3.diagnostics import paired_difference, summarize
from elmur_klpo.step3.policies import FIXED_GRID, BlendingManager
from elmur_klpo.step3.reader import TaskReader
from elmur_klpo.step3.rollout import collect_native
from elmur_klpo.trainer import seed_everything


@contextmanager
def evaluation_rng(seed: int) -> Iterator[None]:
    """Evaluation must not perturb training streams, including CUDA."""
    state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    seed_everything(seed)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


def evaluate(
    reader: TaskReader,
    task: dict,
    seeds: list[int],
    manager: BlendingManager,
    *,
    policy_seed: int = 10000,
    **kwargs: object,
) -> dict:
    """Regenerate complete trajectories with the selected manager's own bank."""
    with evaluation_rng(policy_seed):
        rollout = collect_native(reader, task, seeds, manager, **kwargs)
    return summarize(rollout, reader.model.memory_size)


def readiness(
    reader: TaskReader,
    task: dict,
    seeds: list[int],
    initial: BlendingManager,
    target_exposed: bool,
    margin: float = 0.0,
) -> dict:
    """Use paired held-out native returns, never imitation likelihood, for gating."""
    genuine = evaluate(reader, task, seeds, initial)
    ablated = evaluate(reader, task, seeds, initial, ablate=True)
    difference = paired_difference(
        genuine["episode_returns"], ablated["episode_returns"]
    )
    informative = evaluate(reader, task, seeds, initial, informative=True)
    robustness = {
        str(alpha): evaluate(
            reader, task, seeds, initial, method="fixed", fixed_alpha=alpha
        )
        for alpha in (0.0, 0.95)
    }
    passed = len(seeds) >= 16 and difference["ci95"][0] > margin and target_exposed
    return {
        "passed": passed,
        "memory_contribution": difference,
        "genuine": genuine,
        "ablated": ablated,
        "decodability": {
            "measurement": informative,
            "method": "Retrieve earlier actual tok2mem write snapshots using privileged task labels; diagnostic only",
        },
        "target_exposed": target_exposed,
        "management_robustness": robustness,
        "gate": {
            "metric": "native episodic return",
            "minimum_episodes": 16,
            "lower_paired_ci_must_exceed": margin,
        },
        "failure": None
        if passed
        else "Memory contribution is not supported by a positive held-out paired interval, insufficient validation episodes, or missing target exposure",
    }


def fixed_comparison(
    reader: TaskReader, task: dict, seeds: list[int], initial: BlendingManager
) -> dict:
    """Common-reader diagnostics; slot selection remains uniform for the sweep."""
    return {
        "initial": evaluate(reader, task, seeds, initial),
        "native_lru": evaluate(reader, task, seeds, initial, method="lru"),
        "noop": evaluate(reader, task, seeds, initial, method="noop"),
        "fixed": {
            str(alpha): evaluate(
                reader, task, seeds, initial, method="fixed", fixed_alpha=alpha
            )
            for alpha in FIXED_GRID
        },
    }
