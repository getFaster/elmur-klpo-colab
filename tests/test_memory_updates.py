"""Validation coverage for the shared slot-conditioned memory transition."""

import pytest
import torch

from elmur_klpo.training_memory import write_vectors


@pytest.mark.parametrize("action", [-1, 11, 0.5])
def test_transition_rejects_invalid_actions(action):
    memory = torch.zeros(1, 2, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        write_vectors(memory, torch.ones_like(memory), torch.tensor([action]))


def test_transition_rejects_wrong_memory_shape():
    memory = torch.zeros(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        write_vectors(memory, torch.ones_like(memory), torch.tensor([0, 0]))
