"""Fixed held-out imitation protocol for upstream-style reader checkpoint selection."""

import numpy as np
import torch
from torch.nn import functional as F

from elmur_klpo.step3.evaluation import evaluation_rng
from elmur_klpo.step3.reader_data import ReaderData
from elmur_klpo.step3.upstream_reader import FixedLRUReader, detach_memories
from elmur_klpo.vendor.elmur.layers.memory import MemoryState


@torch.no_grad()
def validation(
    reader: FixedLRUReader, data: ReaderData, config: dict, memory_mode: str = "genuine"
) -> dict:
    """Validation split only; episode units, fixed seed, no training RNG changes."""
    if memory_mode not in ("genuine", "zero", "shuffled"):
        raise ValueError("Unknown memory intervention")
    previous_mode = reader.training
    reader.eval()
    settings = config["preparation"]
    batch_size = config["evaluation"]["batch_size"]
    length = settings["context_length"]
    forwards = 0
    errors, phases, accuracies, positions = [], {0: [], 1: [], 2: []}, [], []
    try:
        with evaluation_rng(config["evaluation"]["seed"]):
            for start in range(0, data.validation_size(), batch_size):
                batch = data.load(
                    list(range(start, min(start + batch_size, data.validation_size()))),
                    "validation",
                )
                memories = reader.initial(len(batch.identities))
                donor_batch = None
                donor_memories = None
                if memory_mode == "shuffled":
                    if data.validation_size() < 2:
                        raise ValueError(
                            "Shuffled-memory diagnostic requires at least two independent episodes"
                        )
                    donors = [
                        (i + 1) % data.validation_size()
                        for i in range(
                            start, min(start + batch_size, data.validation_size())
                        )
                    ]
                    donor_batch = data.load(donors, "validation")
                    donor_memories = reader.initial(len(batch.identities))
                predictions = []
                for section in range(3):
                    chunk = batch.section(section * length, length)
                    read_memory = memories
                    if memory_mode == "zero":
                        read_memory = [
                            MemoryState(torch.zeros_like(m.vec), m.pos)
                            for m in memories
                        ]
                    elif memory_mode == "shuffled":
                        # Complete layer banks from a different held-out episode;
                        # paired slot permutation alone would be attention-invariant.
                        read_memory = donor_memories
                    out = reader(chunk.observations, read_memory, section * length)
                    forwards += 1
                    if donor_batch is not None:
                        donor_out = reader(
                            donor_batch.section(section * length, length).observations,
                            donor_memories,
                            section * length,
                        )
                        donor_memories = detach_memories(donor_out["memory_states"])
                        forwards += 1
                    predictions.append(out["logits"].cpu())
                    memories = detach_memories(out["memory_states"])
                logits = torch.cat(predictions, 1)
                mask = batch.valid & batch.action_valid
                if reader.task_name == "remember_color":
                    loss = (logits - batch.targets).square().mean(-1)
                else:
                    # Report unsmoothed recall CE; training uses upstream label smoothing.
                    loss = F.cross_entropy(
                        logits.flatten(0, 1),
                        batch.targets.clamp_min(0).flatten(),
                        reduction="none",
                    ).reshape_as(mask)
                    correct = (logits.argmax(-1) == batch.targets).float()
                    accuracies.extend(((correct * mask).sum(1) / mask.sum(1)).tolist())
                    positions.extend(
                        [correct[i][mask[i]].tolist() for i in range(len(mask))]
                    )
                errors.extend(((loss * mask).sum(1) / mask.sum(1)).tolist())
                for phase in phases:
                    phase_mask = mask & (batch.phase == phase)
                    if phase_mask.any():
                        values = (loss * phase_mask).sum(1) / phase_mask.sum(
                            1
                        ).clamp_min(1)
                        phases[phase].extend(values[phase_mask.any(1)].tolist())
    finally:
        reader.train(previous_mode)
    metric = "action_mse" if reader.task_name == "remember_color" else "recall_ce"
    # Episode bootstrap uncertainty, not token-level pseudo-replication.
    draws = (
        np.random.default_rng(771)
        .choice(errors, (1000, len(errors)), replace=True)
        .mean(1)
    )
    result = {
        "selection_loss": float(np.mean(errors)),
        "metric": metric,
        "episode_losses": errors,
        "episode_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "episodes": len(errors),
        "memory_mode": memory_mode,
        "phase_loss": {
            str(k): float(np.mean(v)) if v else None for k, v in phases.items()
        },
        "split": "validation",
        "model_forward_passes": forwards,
    }
    if accuracies:
        result.update(
            full_length_accuracy=float(np.mean(accuracies)),
            episode_accuracy=accuracies,
            play_position_accuracy=np.mean(positions, axis=0).tolist(),
            native_reward_equivalent=float(2 * np.mean(accuracies) - 1),
            reward_measurement="Computed from all 52 native PLAY correctness terms; no renderer/closed-loop robotics claim",
        )
    return result
