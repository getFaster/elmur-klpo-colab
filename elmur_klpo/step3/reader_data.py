"""Observation-only BC batches: native oracle suits or audited recorded RGB."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from elmur_klpo.step3.curriculum import environment_seeds
from elmur_klpo.step3.offline_preparation import RememberColorDemonstrations
from elmur_klpo.step3.tasks import AutoencodeTask


@dataclass
class ReaderBatch:
    """CPU observations [B,T,...], targets [B,T,(8)], boolean masks [B,T]."""

    observations: Tensor
    targets: Tensor
    valid: Tensor
    action_valid: Tensor
    phase: Tensor  # cue/watch=0, delay=1, control/play=2; padding=-1
    identities: list[str]

    def section(self, start: int, length: int) -> "ReaderBatch":
        end = start + length
        return ReaderBatch(
            *(
                v[:, start:end]
                for v in (
                    self.observations,
                    self.targets,
                    self.valid,
                    self.action_valid,
                    self.phase,
                )
            ),
            self.identities,
        )


def autoencode_batch(seeds: list[int]) -> ReaderBatch:
    """103 native decisions: 51 WATCH then 52 PLAY, padded to 105 tokens."""
    observations, labels, masks = [], [], []
    task = AutoencodeTask(52)
    try:
        for seed in seeds:
            obs = task.reset(seed)
            episode, targets, scored = [], [], []
            done = False
            while not done:
                label, play = task.label()
                episode.append(torch.from_numpy(obs.copy()))
                # WATCH controls are irrelevant; never supervise arbitrary actions.
                targets.append(label if play else 0)
                scored.append(play)
                obs, _, done, _ = task.step(label if play else 0)
            observations.append(
                torch.stack(episode + [episode[-1]] * (105 - len(episode)))
            )
            labels.append(torch.tensor(targets + [-10] * (105 - len(targets))))
            masks.append(torch.tensor(scored + [False] * (105 - len(scored))))
    finally:
        task.close()
    valid = torch.arange(105)[None].expand(len(seeds), -1) < 103
    action_valid = torch.stack(masks)
    phase = torch.where(action_valid, 2, 0).masked_fill(~valid, -1)
    return ReaderBatch(
        torch.stack(observations),
        torch.stack(labels),
        valid,
        action_valid,
        phase,
        [f"native_seed:{s}" for s in seeds],
    )


class ReaderData:
    """Deterministic batch materialization; recorded data never requires a renderer."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.native_environment_interactions = 0
        self.materialized_episodes = 0
        self.task = config["task"]["name"]
        self.dataset = None
        if self.task == "remember_color":
            self.dataset = RememberColorDemonstrations(
                config["preparation"]["demonstrations"], config["data_seed"]
            )
            self.train_size = len(self.dataset.splits["train"])
            self.signature = self.dataset.split_manifest()
        else:
            self.train_size = 3000  # nominal epoch; every new epoch generates new seeds
            self.signature = {
                "source": "fresh native POPGym 1.0.7, oracle PLAY labels",
                "split": "disjoint deterministic seed blocks",
                "data_seed": config["data_seed"],
                "cards": 52,
            }

    def load(self, indices: list[int], split: str, epoch: int = 0) -> ReaderBatch:
        """Recreate a pending batch from IDs without consuming training RNG."""
        if self.dataset is None:
            offset = epoch * self.train_size if split == "train" else 0
            seeds = [
                environment_seeds(split, offset + i, 1, self.config["data_seed"])[0]
                for i in indices
            ]
            batch = autoencode_batch(seeds)
            self.native_environment_interactions += len(seeds) * 103
            self.materialized_episodes += len(seeds)
            return batch
        paths = [self.dataset.splits[split][i] for i in indices]
        self.materialized_episodes += len(paths)
        obs, targets, _ = self.dataset.batch(paths, torch.device("cpu"))
        length = self.config["preparation"]["context_length"] * 3
        if obs.shape[1] != length:
            raise ValueError(
                "RememberColor recordings must have exactly three complete sections"
            )
        valid = torch.ones(targets.shape[:2], dtype=torch.bool)
        t = torch.arange(length)[None].expand(len(paths), -1)
        phase = torch.where(t < 5, 0, torch.where(t < 10, 1, 2))
        return ReaderBatch(
            obs,
            targets,
            valid,
            valid.clone(),
            phase,
            [str(Path(p).name) for p in paths],
        )

    def validation_size(self) -> int:
        return (
            len(self.dataset.splits["validation"])
            if self.dataset
            else self.config["evaluation"]["episodes"]
        )
