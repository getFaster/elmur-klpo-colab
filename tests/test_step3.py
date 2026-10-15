"""Native-task invariants, controlled policies, gradient boundaries and resume."""

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from popgym.envs.autoencode import AutoencodeEasy

from elmur_klpo.rollout import features, manager_dimension
from elmur_klpo.step3.curriculum import Curriculum, environment_seeds
from elmur_klpo.step3.evaluation import evaluate, evaluation_rng
from elmur_klpo.step3.experiment import run, run_task
from elmur_klpo.step3.policies import (
    BlendingManager,
    initial_probabilities,
    retention_from_native,
    transition,
)
from elmur_klpo.step3.reader import TaskReader
from elmur_klpo.step3.rollout import collect_native, discounted_future
from elmur_klpo.step3.tasks import AutoencodeTask, TaskUnavailable, make_task
from elmur_klpo.step3.training import train_manager
from elmur_klpo.trainer import load_checkpoint, save_checkpoint, seed_everything, update
from elmur_klpo.vendor.elmur.layers.memory import MemoryState


@pytest.fixture
def native_pipeline():
    seed_everything(19)
    reader = TaskReader({"name": "autoencode"}, {"slots": 2, "dimension": 8})
    manager = BlendingManager(manager_dimension(reader), 2)
    return reader, manager, {"name": "autoencode", "target": 4}


@pytest.mark.parametrize("slots", [1, 2, 4])
@pytest.mark.parametrize("variant", ["full", "coefficient_only"])
def test_initial_probabilities_are_task_independent(slots, variant):
    manager = BlendingManager(7, slots, variant)
    states = torch.randn(5, 7)
    states[:, 0] = 1
    expected = initial_probabilities(slots).float().expand(5, -1)
    torch.testing.assert_close(manager.probs(states), expected, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(
        initial_probabilities(slots).sum(),
        torch.tensor(1.0, dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )
    assert torch.count_nonzero(manager.weight[:, 1:]) == 0


def test_initial_empirical_sampling():
    seed_everything(4)
    manager = BlendingManager(2, 2)
    samples = manager.distribution(
        torch.tensor([[1.0, 3.0]]).expand(100_000, -1)
    ).sample()
    frequency = torch.bincount(samples, minlength=11).float() / len(samples)
    torch.testing.assert_close(
        frequency, initial_probabilities(2).float(), atol=0.004, rtol=0
    )
    assert abs(float(frequency[:5].sum() - frequency[5:10].sum())) < 0.006
    assert float(frequency[2]) == pytest.approx(0.294, abs=0.004)


@pytest.mark.parametrize("variant", ["slot_only", "coefficient_only"])
def test_constraints_survive_optimization(variant):
    manager = BlendingManager(3, 3, variant)
    states = torch.randn(9, 3)
    states[:, 0] = 1
    optimizer = torch.optim.Adam(manager.parameters(), lr=0.1)
    for _ in range(3):
        objective = -manager.log_probs(states)[:, 2].mean()
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
    probs = manager.probs(states)
    joint = probs[:, :-1].reshape(9, 3, 5)
    if variant == "slot_only":
        assert torch.count_nonzero(joint[..., [0, 1, 3, 4]]) == 0
        assert torch.count_nonzero(probs[:, -1]) == 0
    else:
        torch.testing.assert_close(joint[:, 0], joint[:, 1], atol=0, rtol=0)
        torch.testing.assert_close(joint[:, 1], joint[:, 2], atol=0, rtol=0)
        torch.testing.assert_close(
            joint.sum(-1), ((1 - probs[:, -1]) / 3)[:, None].expand(-1, 3)
        )


def test_coefficient_convention_and_independent_sweep():
    assert retention_from_native(0.4) == pytest.approx(0.6)
    assert retention_from_native(0.8) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        retention_from_native(1.1)
    memory = MemoryState(
        torch.tensor([[[2.0], [3.0]]], requires_grad=True), torch.tensor([[1, 2]])
    )
    candidate = torch.tensor([[[7.0], [11.0]]], requires_grad=True)
    result = transition(memory, candidate, torch.tensor([5]), 8, fixed_alpha=0.2)
    torch.testing.assert_close(result.vec, torch.tensor([[[2.0], [9.4]]]))
    assert result.pos.tolist() == [[1, 8]]
    assert not result.vec.requires_grad
    noop = transition(memory, candidate, torch.tensor([10]), 8)
    keep = transition(memory, candidate, torch.tensor([5]), 8, fixed_alpha=1.0)
    torch.testing.assert_close(noop.vec, memory.vec, atol=0, rtol=0)
    torch.testing.assert_close(keep.vec, memory.vec, atol=0, rtol=0)
    assert noop.pos.tolist() == [[1, 2]]
    assert keep.pos.tolist() == [[1, 8]]


@pytest.mark.parametrize("length", [2, 4, 52, 78, 104])
def test_native_objective_reward_and_reset(length):
    task = AutoencodeTask(length)
    first = task.reset(18)
    assert first.shape == (6,)
    assert first.sum() == 2
    reward_sum = 0
    steps = 0
    done = False
    while not done:
        action, _ = task.label()
        observation, reward, done, _ = task.step(action)
        assert observation.shape == task.observation_shape
        reward_sum += reward
        steps += 1
    assert reward_sum == pytest.approx(1.0)
    assert steps == 2 * length - 1
    with pytest.raises(RuntimeError, match="Reset"):
        task.step(0)
    np.testing.assert_array_equal(first, task.reset(18))
    with pytest.raises(ValueError):
        task.step(4)
    task.close()


def test_target_adapter_matches_unmodified_upstream():
    task = AutoencodeTask(52)
    native = AutoencodeEasy()
    task.reset(13)
    obs, _ = native.reset(seed=13)
    assert task.observation == obs
    for index in range(103):
        _, reward, done, _ = task.step(index % 4)
        obs, expected, terminated, truncated, _ = native.step(index % 4)
        assert reward == expected
        assert done == (terminated or truncated)
        assert task.observation == obs


def test_short_deck_does_not_remove_suits():
    suits = set()
    task = AutoencodeTask(2)
    for seed in range(100):
        task.reset(seed)
        suits.add(task.observation[1])
    assert suits == {0, 1, 2, 3}


def test_native_robot_requires_explicit_runtime():
    with pytest.raises(TaskUnavailable, match="native_python"):
        make_task({"name": "remember_color", "target": 5})
    with pytest.raises(ValueError):
        make_task({"name": "invented_task", "target": 4})


def test_native_rollout_indexing_and_replay(native_pipeline):
    reader, manager, task = native_pipeline
    rollout = collect_native(reader, task, [1, 2, 3], manager)
    batch = rollout.batch
    assert batch.states.shape == (3, 7, manager_dimension(reader))
    assert batch.valid[:, :-1].all()
    assert not batch.valid[:, -1].any()
    assert (batch.actions[:, -1] == manager.action_space.noop_id).all()
    assert not batch.states.requires_grad
    assert not batch.returns.requires_grad
    assert len(rollout.traces) == 18
    torch.testing.assert_close(batch.returns[:, -2], batch.rewards[:, -1])
    torch.testing.assert_close(batch.returns[:, 0], batch.rewards[:, 1:].sum(1))
    torch.testing.assert_close(manager.log_probs(batch.states), batch.log_q)


def test_discount_and_padding_do_not_cross_episodes():
    rewards = torch.tensor(
        [[100.0, 2.0, 4.0, 8.0], [9.0, 6.0, 0.0, 0.0]], requires_grad=True
    )
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    expected = torch.tensor([[6.0, 8.0, 8.0, 0.0], [6.0, 0.0, 0.0, 0.0]])
    actual = discounted_future(rewards, mask, 0.5)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


@pytest.mark.parametrize("variant", ["full", "slot_only", "coefficient_only"])
def test_manager_update_freezes_reader_and_requires_freshness(native_pipeline, variant):
    reader, _, task = native_pipeline
    manager = BlendingManager(manager_dimension(reader), 2, variant)
    before_reader = copy.deepcopy(reader.state_dict())
    before_manager = manager.weight.detach().clone()
    optimizer = torch.optim.Adam(manager.parameters(), lr=0.001)
    batch = collect_native(reader, task, [1, 2, 3, 4], manager.frozen_copy()).batch
    diagnostic = update(manager, optimizer, batch, "reinforce")
    assert diagnostic["gradient_norm"] > 0
    assert not torch.equal(before_manager, manager.weight)
    for name, value in reader.state_dict().items():
        torch.testing.assert_close(value, before_reader[name], atol=0, rtol=0)
    assert all(p.grad is None and not p.requires_grad for p in reader.parameters())
    with pytest.raises(ValueError, match="fresh on-policy"):
        update(manager, optimizer, batch, "reinforce")


def test_slot_candidates_and_features(native_pipeline):
    reader, _, _ = native_pipeline
    memory = MemoryState(torch.randn(2, 2, 8), torch.tensor([[0, 2], [3, 4]]))
    logits, context, candidate, _ = reader.step(torch.randn(2, 6), memory, 5)
    assert logits.shape == (2, 4)
    assert candidate.shape == (2, 2, 8)
    assert candidate.requires_grad
    assert not torch.equal(candidate[:, 0], candidate[:, 1])
    state = features(context, memory, candidate, 5)
    torch.testing.assert_close(state[:, 25:41], candidate.detach().flatten(1))
    assert not state.requires_grad


def test_continuous_native_reader_shapes_and_bounds():
    reader = TaskReader({"name": "remember_color"}, {"slots": 2, "dimension": 8})
    logits, _, candidate, _ = reader.step(
        torch.zeros(2, 6, 128, 128), reader.initial(2), 0
    )
    assert logits.shape == (2, 8)
    assert candidate.shape == (2, 2, 8)
    action, score = reader.task_action(logits, greedy=False)
    assert action.shape == (2, 8) and score.shape == (2,)
    assert (action.abs() <= 1).all()
    assert torch.isfinite(score).all()


def test_deterministic_evaluation_preserves_training_rng(native_pipeline):
    reader, manager, task = native_pipeline
    before = torch.get_rng_state().clone()
    first = evaluate(reader, task, [11, 12], manager, policy_seed=91)
    assert torch.equal(before, torch.get_rng_state())
    second = evaluate(reader, task, [11, 12], manager, policy_seed=91)
    assert first["episode_returns"] == second["episode_returns"]
    assert first["behavior"] == second["behavior"]
    assert first["return"] == second["return"]


def test_rng_scope_restores_all_streams_on_exception():
    seed_everything(19)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with evaluation_rng(5):
            raise RuntimeError("interrupted evaluation")
    assert random.getstate() == before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_checkpoint_reproduces_next_native_update(native_pipeline, tmp_path):
    reader, manager, task = native_pipeline
    optimizer = torch.optim.Adam(manager.parameters(), lr=0.001)
    first = collect_native(reader, task, [1, 2], manager)
    update(manager, optimizer, first.batch, "reinforce")
    checkpoint = tmp_path / "manager.pt"
    save_checkpoint(checkpoint, manager, optimizer, 1, {"reader": reader.state_dict()})
    second = collect_native(reader, task, [3, 4], manager)
    update(manager, optimizer, second.batch, "reinforce")
    expected = manager.weight.detach().clone()
    load_checkpoint(checkpoint, manager, optimizer)
    replay = collect_native(reader, task, [3, 4], manager)
    torch.testing.assert_close(
        replay.batch.actions, second.batch.actions, atol=0, rtol=0
    )
    torch.testing.assert_close(replay.batch.states, second.batch.states, atol=0, rtol=0)
    update(manager, optimizer, replay.batch, "reinforce")
    torch.testing.assert_close(manager.weight, expected, atol=0, rtol=0)


def test_curriculum_and_seed_separation():
    schedule = Curriculum(
        [{"updates": 2, "difficulty": 2}, {"updates": 3, "difficulty": 52}], 52
    )
    assert schedule.at(1).difficulty == 2 and schedule.at(2).difficulty == 52
    with pytest.raises(ValueError):
        Curriculum([{"updates": 2, "difficulty": 2}], 52)
    splits = [
        set(environment_seeds(split, 0, 100))
        for split in ("train", "validation", "test", "robustness")
    ]
    for i, split in enumerate(splits):
        assert all(not split & other for other in splits[i + 1 :])
    with pytest.raises(ValueError):
        environment_seeds("train", 99_999_999, 2)


def test_failed_gate_never_trains_manager(native_pipeline, tmp_path):
    reader, _, _ = native_pipeline
    config = json.loads(
        Path(
            "experiments/step3/historical_pre_repair/configs/step3_smoke.json"
        ).read_text()
    )["tasks"][0]
    with pytest.raises(RuntimeError, match="readiness gate failed"):
        train_manager(config, tmp_path, reader, {"passed": False}, 1, "full")
    assert not list(tmp_path.glob("manager*"))


def test_manager_partial_resume_matches_uninterrupted(
    native_pipeline, tmp_path, monkeypatch
):
    import elmur_klpo.step3.training as training

    reader, _, task = native_pipeline
    config = json.loads(
        Path(
            "experiments/step3/historical_pre_repair/configs/step3_smoke.json"
        ).read_text()
    )["tasks"][0]
    config["task"] = task
    config["reader"]["dimension"] = 8
    config["manager"].update(updates=2, batch_size=2, eval_every=1)
    config["evaluation"]["episodes"] = 2
    expected, _ = train_manager(
        config, tmp_path / "full", reader, {"passed": True}, 3, "full"
    )
    original = training.update
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "update", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_manager(config, tmp_path / "resumed", reader, {"passed": True}, 3, "full")
    monkeypatch.setattr(training, "update", original)
    actual, result = train_manager(
        config, tmp_path / "resumed", reader, {"passed": True}, 3, "full", resume=True
    )
    torch.testing.assert_close(actual.weight, expected.weight, atol=0, rtol=0)
    assert len(result["history"]) == 2
    incompatible = copy.deepcopy(config)
    incompatible["manager"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="configuration or reader mismatch"):
        train_manager(
            incompatible,
            tmp_path / "resumed",
            reader,
            {"passed": True},
            3,
            "full",
            resume=True,
        )


def test_variable_termination_masks_padding(native_pipeline, monkeypatch):
    import elmur_klpo.step3.rollout as rollout_module

    reader, manager, task = native_pipeline
    lengths = iter([2, 4])
    monkeypatch.setattr(
        rollout_module, "make_task", lambda _: AutoencodeTask(next(lengths))
    )
    batch = collect_native(reader, task, [1, 2], manager).batch
    assert batch.valid[0].tolist() == [True, True, False, False, False, False, False]
    assert batch.valid[1].tolist() == [True, True, True, True, True, True, False]
    assert (batch.rewards[0, 3:] == 0).all()
    assert (batch.returns[0, 2:] == 0).all()
    torch.testing.assert_close(batch.returns[0, 1], batch.rewards[0, 2])


@pytest.mark.skipif(
    os.environ.get("ELMUR_TEST_NATIVE_ROBOT") != "1",
    reason="Native robot renderer unavailable; opt in with ELMUR_TEST_NATIVE_ROBOT=1",
)
def test_native_robot_complete_episode():
    task = make_task(
        {
            "name": "remember_color",
            "target": 5,
            "native_python": os.environ.get("ELMUR_NATIVE_PYTHON"),
        }
    )
    try:
        observation = task.reset(123)
        assert observation.shape == (6, 128, 128)
        done = False
        for _ in range(60):
            observation, reward, done, info = task.step(np.zeros(8, dtype=np.float32))
            assert np.isfinite(observation).all() and np.isfinite(reward)
            assert isinstance(info["success"], bool)
            if done:
                break
        assert done
    finally:
        task.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_native_pipeline():
    reader = TaskReader({"name": "autoencode"}, {"slots": 2, "dimension": 8}).cuda()
    manager = BlendingManager(manager_dimension(reader), 2).cuda()
    batch = collect_native(
        reader, {"name": "autoencode", "target": 4}, [1, 2], manager
    ).batch
    assert batch.states.is_cuda
    assert (
        update(
            manager,
            torch.optim.Adam(manager.parameters(), lr=0.001),
            batch,
            "reinforce",
        )["gradient_norm"]
        > 0
    )


def test_legacy_preparation_config_rejected(tmp_path):
    config = json.loads(
        Path(
            "experiments/step3/historical_pre_repair/configs/step3_smoke.json"
        ).read_text()
    )
    with pytest.raises(ValueError, match="Legacy Step 3 configuration"):
        run(config, tmp_path)


def test_repaired_manager_stages_remain_blocked(tmp_path):
    config = json.loads(Path("configs/step3_manager_training.json").read_text())[
        "tasks"
    ][0]
    with pytest.raises(RuntimeError, match="Learned-manager stages blocked"):
        run_task(config, tmp_path, "train", False, "cpu")
    assert not list(tmp_path.glob("manager*"))
