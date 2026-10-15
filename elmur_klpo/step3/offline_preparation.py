"""Offline RememberColor expert-action preparation; no simulator or rendering."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class RememberColorDemonstrations:
    """Read immutable uint8 RGB [T,128,128,6] and expert targets [T,8]."""

    def __init__(self, config: dict, seed: int) -> None:
        self.root = Path(config["directory"]).resolve()
        files = sorted(self.root.glob("train_data_*.npz"))
        counts = [
            config[key]
            for key in ("train_episodes", "validation_episodes", "test_episodes")
        ]
        if any(count < 1 for count in counts) or sum(counts) != len(files):
            raise ValueError(
                "Disjoint split counts must cover the entire demonstration dataset"
            )
        permutation = np.random.default_rng(seed).permutation(len(files))
        boundaries = np.cumsum([0, *counts])
        self.splits = {
            name: [files[i] for i in permutation[boundaries[j] : boundaries[j + 1]]]
            for j, name in enumerate(("train", "validation", "test"))
        }
        self.action_target = config["action_target"]
        if self.action_target != "raw":
            raise ValueError("Upstream repair requires raw expert action targets")
        self.provenance = json.loads(Path(config["manifest"]).read_text())
        if (
            Path(self.provenance["data_directory"]).resolve() != self.root
            or self.provenance["task"] != "RememberColor3-v0"
        ):
            raise ValueError("Dataset manifest path/task mismatch")
        digest = hashlib.sha256()
        for path in files:
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        self.signature = {
            "archive_sha256": self.provenance["sha256"],
            "files_metadata_sha256": digest.hexdigest(),
        }

    @lru_cache(maxsize=32)
    def episode(self, path: Path) -> tuple[Tensor, Tensor, int]:
        """Return RGB, configured targets, and count outside controller bounds."""
        with np.load(path, allow_pickle=False) as data:
            rgb, actions, done = data["rgb"], data["action"], data["done"]
            length = len(rgb)
            if (
                rgb.dtype != np.uint8
                or rgb.shape != (length, 128, 128, 6)
                or actions.shape != (length, 8)
            ):
                raise ValueError(f"Unexpected demonstration shapes: {path}")
            if (
                done.shape != (length,)
                or not done[-1]
                or done[:-1].any()
                or not np.isfinite(actions).all()
            ):
                raise ValueError(f"Invalid episode boundaries/actions: {path}")
            outside_bounds = int((np.abs(actions) > 1).sum())
            targets = actions
            return (
                torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy()),
                torch.from_numpy(targets.astype(np.float32)),
                outside_bounds,
            )

    def batch(
        self, paths: list[Path], device: torch.device
    ) -> tuple[Tensor, Tensor, int]:
        """Stack complete equal-length episodes, retaining compact uint8 on device."""
        episodes = [self.episode(path) for path in paths]
        observations = torch.stack([episode[0] for episode in episodes]).to(device)
        actions = torch.stack([episode[1] for episode in episodes]).to(device)
        return observations, actions, sum(episode[2] for episode in episodes)

    def split_manifest(self) -> dict:
        return {
            "source": self.signature,
            "splits": {
                key: [p.name for p in paths] for key, paths in self.splits.items()
            },
            "seed": "Filename-disjoint splits only; source episodes do not record collection seeds",
            "action_target": self.action_target,
        }
