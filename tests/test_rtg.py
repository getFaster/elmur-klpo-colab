import pytest
import torch

from elmur_klpo.returns import post_write_rtg


def test_rtg_excludes_current_reward_and_has_no_terminal_action():
    rewards = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    torch.testing.assert_close(
        post_write_rtg(rewards), rewards.new_tensor([14.0, 12.0, 8.0])
    )
    torch.testing.assert_close(post_write_rtg(rewards[:2]), rewards.new_tensor([2.0]))


def test_rtg_batches_and_reward_timing():
    rewards = torch.tensor([[1.0, 2.0, 4.0], [-1.0, -2.0, -4.0]], dtype=torch.float64)
    torch.testing.assert_close(
        post_write_rtg(rewards), rewards.new_tensor([[6.0, 4.0], [-6.0, -4.0]])
    )
    changed = rewards.clone()
    changed[:, 1] += 7
    difference = post_write_rtg(changed) - post_write_rtg(rewards)
    torch.testing.assert_close(difference, rewards.new_tensor([[7.0, 0.0], [7.0, 0.0]]))


@pytest.mark.parametrize("shape", [(), (0,), (1,)])
def test_rtg_rejects_missing_future_rewards(shape):
    with pytest.raises(ValueError):
        post_write_rtg(torch.zeros(shape))


def test_rollout_reads_before_writing_and_excludes_terminal_action(env, batch):
    assert batch.actions.shape == (11**3, 3)
    assert batch.rewards.shape == (11**3, 4)
    assert batch.memories.shape == (11**3, 4, 2)
    assert batch.states.shape == (11**3, 3, 5)
    assert batch.returns.shape == batch.actions.shape
    for time in range(env.decisions):
        torch.testing.assert_close(batch.states[:, time, 3:], batch.memories[:, time])
        torch.testing.assert_close(
            batch.rewards[:, time],
            env.reader_log_likelihood(time, batch.memories[:, time]),
        )
        torch.testing.assert_close(
            batch.memories[:, time + 1],
            env.transition(time, batch.memories[:, time], batch.actions[:, time]),
        )
    torch.testing.assert_close(batch.returns[:, -1], batch.rewards[:, -1])
    assert not any(
        t.requires_grad for t in (batch.states, batch.rewards, batch.memories)
    )
    with pytest.raises(ValueError):
        env.transition(env.decisions, env.initial_memory(), 0)
