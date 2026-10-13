"""Step 2 integration and regression tests, reusing the verified estimator."""

from dataclasses import replace

import pytest
import torch

from elmur_klpo.demonstrations import tmaze_demonstrations
from elmur_klpo.evaluation import closed_loop
from elmur_klpo.klpo import backward_surrogate
from elmur_klpo.manager import CategoricalManager
from elmur_klpo.memory import ActionSpace
from elmur_klpo.reader import ELMURReader
from elmur_klpo.rollout import collect, features, manager_dimension
from elmur_klpo.trainer import load_checkpoint, save_checkpoint, seed_everything, update
from elmur_klpo.training_memory import write_vectors
from elmur_klpo.vendor.elmur.layers.memory import MemoryState


@pytest.fixture
def pipeline():
    seed_everything(19)
    reader = ELMURReader(slots=2, dimension=8).freeze()
    data = tmaze_demonstrations(8, 7, 3)
    manager = CategoricalManager(
        manager_dimension(reader), ActionSpace(2), dtype=torch.float32
    )
    return reader, data, manager


def test_action_encoding_roundtrip():
    for slots in (1, 2, 4):
        space = ActionSpace(slots)
        assert len(space) == slots * 5 + 1
        for index in range(len(space)):
            assert space.encode(*space.decode(index)) == index
    with pytest.raises(ValueError):
        ActionSpace().encode(0, 1.0)


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_batched_vector_writes_match_reference(slots):
    torch.manual_seed(17)
    space = ActionSpace(slots)
    memory = torch.randn(len(space), slots, 3, dtype=torch.float64, requires_grad=True)
    candidates = torch.randn_like(memory, requires_grad=True)
    original_memory = memory.detach().clone()
    original_candidates = candidates.detach().clone()
    ids = torch.arange(len(space))
    result = write_vectors(memory, candidates, ids)
    assert not result.requires_grad
    for row, action in enumerate(ids.tolist()):
        expected = memory[row].detach().clone()
        slot, alpha = space.decode(action)
        if slot is not None:
            expected[slot] = (
                alpha * expected[slot] + (1 - alpha) * candidates[row, slot].detach()
            )
        torch.testing.assert_close(result[row], expected, atol=1e-14, rtol=0)
    torch.testing.assert_close(result[-1], memory[-1], atol=0, rtol=0)
    assert result.data_ptr() != memory.data_ptr()
    torch.testing.assert_close(memory, original_memory, atol=0, rtol=0)
    torch.testing.assert_close(candidates, original_candidates, atol=0, rtol=0)


def test_vector_writes_reject_shared_candidate():
    with pytest.raises(ValueError, match="slot-conditioned"):
        write_vectors(torch.zeros(2, 3, 4), torch.ones(2, 4), torch.zeros(2).long())


def test_reader_exposes_original_slot_conditioned_candidates():
    seed_everything(19)
    reader = ELMURReader(slots=2, dimension=8).eval()
    memory = reader.initial(3)
    memory = MemoryState(torch.randn_like(memory.vec), memory.pos)
    captured = []
    hook = reader.model.layers[0].memory_ffn.register_forward_hook(
        lambda module, inputs, output: captured.append(output[0])
    )
    try:
        _, _, candidates, _ = reader.step(torch.randn(3, 4), memory, 0)
    finally:
        hook.remove()
    assert candidates.shape == (3, 2, 8)
    assert captured[0].requires_grad
    assert not candidates.requires_grad
    torch.testing.assert_close(candidates, captured[0], atol=0, rtol=0)
    assert not torch.equal(candidates[:, 0], candidates[:, 1])


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_manager_features_preserve_all_slot_candidates(slots):
    seed_everything(19)
    reader = ELMURReader(slots=slots, dimension=8).freeze()
    memory = reader.initial(3)
    context = torch.randn(3, 8, requires_grad=True)
    candidates = torch.arange(3 * slots * 8, dtype=torch.float32).reshape(3, slots, 8)
    candidates.requires_grad_()
    state = features(context, memory, candidates, time=7)
    assert state.shape == (3, manager_dimension(reader))
    start = 1 + 8 + slots * 8
    torch.testing.assert_close(
        state[:, start : start + slots * 8], candidates.flatten(1), atol=0, rtol=0
    )
    assert not state.requires_grad


def test_initialization_and_batched_reader(pipeline):
    reader, data, _ = pipeline
    memory = reader.initial(8)
    assert torch.count_nonzero(memory.vec) == 0
    assert (memory.pos == -1).all()
    with torch.no_grad():
        batched = reader.step(data.observations[:, 0], memory, 0)
        for row in range(8):
            single = reader.step(
                data.observations[row : row + 1, 0], reader.initial(1), 0
            )
            for index in (0, 1, 2):
                torch.testing.assert_close(
                    batched[index][row : row + 1], single[index], atol=1e-6, rtol=1e-5
                )


def test_variable_length_rtg_and_frozen_states(pipeline):
    reader, data, manager = pipeline
    batch = collect(reader, data, manager.frozen_copy())
    assert data.valid.sum(1).unique().numel() > 1
    for row, length in enumerate(data.valid.sum(1).tolist()):
        for time in range(length - 1):
            torch.testing.assert_close(
                batch.returns[row, time], batch.rewards[row, time + 1 : length].sum()
            )
        assert batch.valid[row].sum() == length - 1
        assert batch.returns[row, length - 1] == 0
        assert batch.query_mask[row, length - 1]
    assert not batch.states.requires_grad
    assert not batch.returns.requires_grad
    assert not batch.log_q.requires_grad
    torch.testing.assert_close(batch.q_probs.sum(-1), torch.ones_like(batch.returns))
    torch.testing.assert_close(
        batch.chosen_log_q, batch.log_q.gather(-1, batch.actions[..., None]).squeeze(-1)
    )
    assert not any(parameter.requires_grad for parameter in reader.parameters())


@pytest.mark.parametrize("algorithm", ["klpo", "reinforce"])
def test_updates_only_manager_and_preserves_behavior(pipeline, algorithm):
    reader, data, manager = pipeline
    behavior = manager.frozen_copy()
    batch = collect(reader, data, behavior)
    old_q = batch.log_q.clone()
    old_states = batch.states.clone()
    old_reader = {name: value.clone() for name, value in reader.state_dict().items()}
    old_weight = manager.weight.detach().clone()
    optimizer = torch.optim.Adam(manager.parameters(), lr=0.01)
    result = update(manager, optimizer, batch, algorithm)
    assert result["gradient_norm"] > 0
    assert result["kl_q_p"] > 0
    assert result["kl_p_q"] > 0
    assert not torch.equal(manager.weight, old_weight)
    torch.testing.assert_close(batch.log_q, old_q, atol=0, rtol=0)
    torch.testing.assert_close(batch.states, old_states, atol=0, rtol=0)
    torch.testing.assert_close(behavior.log_probs(batch.states), old_q, atol=0, rtol=0)
    for name, value in reader.state_dict().items():
        torch.testing.assert_close(value, old_reader[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in reader.parameters())


def test_klpo_masked_gradient_matches_step1_reference(pipeline):
    reader, data, manager = pipeline
    batch = collect(reader, data, manager.frozen_copy())
    manager.double()
    batch = replace(
        batch,
        states=batch.states.double(),
        returns=batch.returns.double(),
        log_q=batch.log_q.double(),
        q_probs=batch.q_probs.double(),
    )
    # Move off behavior to exercise the beta ratio and behavior weighting.
    with torch.no_grad():
        manager.weight.add_(torch.randn_like(manager.weight) * 0.03)
    log_p = manager.log_probs(batch.states)
    expected = 0
    for row in range(len(data.actions)):
        valid = batch.valid[row]
        expected = expected + backward_surrogate(
            log_p[row, valid],
            batch.log_q[row, valid],
            batch.actions[row, valid],
            batch.returns[row, valid],
            0.2,
        )
    gradient = torch.autograd.grad(expected / len(data.actions), manager.weight)[0]
    batch = replace(
        batch,
        states=batch.states.requires_grad_(),
        returns=batch.returns.requires_grad_(),
        log_q=batch.log_q.requires_grad_(),
    )
    optimizer = torch.optim.SGD(manager.parameters(), lr=0)
    update(manager, optimizer, batch, "klpo", beta=0.2, clip=1e6)
    torch.testing.assert_close(manager.weight.grad, gradient, atol=1e-12, rtol=1e-12)
    assert batch.states.grad is None
    assert batch.returns.grad is None
    assert batch.log_q.grad is None


def test_reinforce_rejects_stale_batch(pipeline):
    reader, data, manager = pipeline
    batch = collect(reader, data, manager)
    optimizer = torch.optim.Adam(manager.parameters(), lr=0.01)
    update(manager, optimizer, batch, "reinforce")
    with pytest.raises(ValueError, match="on-policy"):
        update(manager, optimizer, batch, "reinforce")


def test_resume_matches_uninterrupted_next_update(pipeline, tmp_path):
    reader, data, manager = pipeline
    optimizer = torch.optim.Adam(manager.parameters(), lr=0.01)
    update(manager, optimizer, collect(reader, data, manager), "klpo")
    save_checkpoint(
        tmp_path / "checkpoint.pt",
        manager,
        optimizer,
        1,
        {"reader": reader.state_dict()},
    )
    first = collect(reader, data, manager)
    update(manager, optimizer, first, "klpo")
    expected = manager.weight.detach().clone()
    restored = CategoricalManager(
        manager_dimension(reader), ActionSpace(), dtype=torch.float32
    )
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
    checkpoint = load_checkpoint(
        tmp_path / "checkpoint.pt", restored, restored_optimizer
    )
    assert checkpoint["iteration"] == 1
    second = collect(reader, data, restored)
    torch.testing.assert_close(second.actions, first.actions, atol=0, rtol=0)
    torch.testing.assert_close(second.states, first.states, atol=0, rtol=0)
    update(restored, restored_optimizer, second, "klpo")
    torch.testing.assert_close(restored.weight, expected, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["lru", "random", "noop"])
def test_baseline_full_rollout(pipeline, method):
    reader, data, _ = pipeline
    batch = collect(reader, data, method=method)
    assert torch.isfinite(batch.rewards).all()
    if method == "noop":
        assert (batch.actions == ActionSpace().noop_id).all()
        assert (batch.retention == 0).all()


def test_nonfinite_rollout_rejected(pipeline):
    reader, data, manager = pipeline
    batch = collect(reader, data, manager)
    batch.returns[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        update(manager, torch.optim.Adam(manager.parameters()), batch, "klpo")


@pytest.mark.parametrize("method", ["learned", "lru", "random", "noop"])
def test_closed_loop_reproducible_and_terminates(pipeline, method):
    reader, data, manager = pipeline
    seed_everything(91)
    first = closed_loop(reader, data, manager, method, seed=91)
    seed_everything(91)
    second = closed_loop(reader, data, manager, method, seed=91)
    assert first == second
    assert 0 <= first["success_rate"] <= 1
    assert first["environment_return"] == first["success_rate"]
    assert first["mean_episode_steps"] <= data.valid.shape[1] + 1
