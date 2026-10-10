import pytest
import torch

from elmur_klpo.memory import ActionSpace, update_memory


@pytest.mark.parametrize("slot", [0, 1])
@pytest.mark.parametrize("index,alpha", list(enumerate([0.0, 0.25, 0.5, 0.8, 0.95])))
def test_merge_coefficients_leave_other_slot_unchanged(slot, index, alpha):
    memory = torch.tensor([2.0, -3.0], dtype=torch.float64, requires_grad=True)
    candidate = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    result = update_memory(memory, candidate, 5 * slot + index, ActionSpace())
    expected = memory.detach().clone()
    expected[slot] = alpha * memory.detach()[slot] + (1 - alpha) * 5.0
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(
        memory.detach(), torch.tensor([2.0, -3.0], dtype=torch.float64)
    )
    assert result.data_ptr() != memory.data_ptr()
    assert not result.requires_grad
    assert result.grad_fn is None


def test_global_noop_and_batched_actions():
    memory = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    result = update_memory(memory, -2.0, torch.tensor([10, 0, 7]), ActionSpace())
    expected = torch.tensor([[1.0, 2.0], [-2.0, 4.0], [5.0, 2.0]], dtype=torch.float64)
    torch.testing.assert_close(result, expected)
    assert result.data_ptr() != memory.data_ptr()


@pytest.mark.parametrize("action", [-1, 11, 0.5])
def test_transition_rejects_invalid_actions(action):
    with pytest.raises(ValueError):
        update_memory(torch.zeros(2, dtype=torch.float64), 1.0, action, ActionSpace())


def test_transition_rejects_wrong_memory_shape():
    with pytest.raises(ValueError):
        update_memory(torch.zeros(3, dtype=torch.float64), 1.0, 0, ActionSpace())
