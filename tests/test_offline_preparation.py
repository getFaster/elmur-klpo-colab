"""Audited dataset splits and immutable raw targets, independent of rendering."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from elmur_klpo.step3.offline_preparation import RememberColorDemonstrations
from elmur_klpo.step3.upstream_smoke import synthetic_robot_config


@pytest.fixture
def offline_config(tmp_path):
    config = json.loads(Path("configs/step3_smoke.json").read_text())["tasks"][1]
    return synthetic_robot_config(config, tmp_path)


def test_dataset_splits_and_raw_targets(offline_config):
    dataset = RememberColorDemonstrations(
        offline_config["preparation"]["demonstrations"], 1000
    )
    groups = [set(paths) for paths in dataset.splits.values()]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
    path = dataset.root / "train_data_0.npz"
    rgb, action, outside = dataset.episode(path)
    assert rgb.shape == (60, 6, 128, 128) and rgb.dtype == torch.uint8
    with np.load(path) as source:
        torch.testing.assert_close(
            action, torch.from_numpy(source["action"]), atol=0, rtol=0
        )
    assert action.min() < -1 and action.max() > 1
    assert outside > 0


def test_clipped_targets_are_no_longer_a_preparation_option(offline_config):
    config = offline_config["preparation"]["demonstrations"]
    config["action_target"] = "clip_to_controller_bounds"
    with pytest.raises(ValueError, match="raw expert action targets"):
        RememberColorDemonstrations(config, 1000)


def test_dataset_rejects_changed_manifest_path(offline_config, tmp_path):
    config = offline_config["preparation"]["demonstrations"]
    manifest = Path(config["manifest"])
    data = json.loads(manifest.read_text())
    data["data_directory"] = str(tmp_path / "wrong")
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest path/task mismatch"):
        RememberColorDemonstrations(config, 1000)
