"""Closed-loop evaluation in the authors' T-Maze environment."""

import numpy as np
import torch

from elmur_klpo.demonstrations import Demonstrations
from elmur_klpo.manager import CategoricalManager
from elmur_klpo.memory import ActionSpace
from elmur_klpo.reader import ELMURReader
from elmur_klpo.rollout import features
from elmur_klpo.training_memory import write_vectors
from elmur_klpo.vendor.elmur.layers.memory import MemoryState
from elmur_klpo.vendor.tmaze import TMazeClassicPassive


@torch.no_grad()
def closed_loop(
    reader: ELMURReader,
    data: Demonstrations,
    policy: CategoricalManager | None = None,
    method: str = "learned",
    seed: int = 0,
) -> dict:
    """Use greedy reader actions and sampled learned/random memory actions.

    Match held-out corridor lengths and goals, with fresh independent noise.
    The preprocessing and time-based junction flag follow generate_trajectory.
    Environment stepping is CPU-side; all reader inference stays batched.
    """
    reader.freeze()
    if method not in {"learned", "lru", "random", "noop"}:
        raise ValueError("Unknown management method")
    rng = np.random.default_rng(seed)
    device = data.observations.device
    corridors = (data.valid.sum(1) - 1).cpu().tolist()
    cues = data.observations[:, 0, 1].cpu().tolist()
    environments = []
    for length, cue in zip(corridors, cues, strict=True):
        env = TMazeClassicPassive(episode_length=length + 2, corridor_length=length)
        env.reset()
        env.goal_y = int(cue)
        environments.append(env)
    observations = data.observations[:, 0].clone()
    active = torch.ones(len(environments), dtype=torch.bool, device=device)
    memory = reader.initial(len(environments))
    returns = np.zeros(len(environments))
    steps = np.zeros(len(environments))
    space = ActionSpace(reader.model.memory_size)
    for time in range(data.valid.shape[1] + 1):
        logits, context, candidate, native = reader.step(observations, memory, time)
        if method == "learned":
            if policy is None:
                raise ValueError("Learned management requires a policy")
            ids = torch.multinomial(
                policy.probs(features(context, memory, candidate, time)), 1
            ).squeeze(-1)
        else:
            ids = torch.randint(len(space), (len(environments),), device=device)
            if method == "noop":
                ids.fill_(space.noop_id)
        vec = write_vectors(memory.vec, candidate, ids)
        selected = torch.arange(space.slots, device=device)[None] == (ids // 5)[:, None]
        pos = torch.where(selected, time, memory.pos)
        if method == "lru":
            vec, pos = native.vec.detach(), native.pos
        memory = MemoryState(
            torch.where(active[:, None, None], vec, memory.vec),
            torch.where(active[:, None], pos, memory.pos),
        )
        actions = logits.argmax(-1).cpu().tolist()
        active_rows = active.cpu().tolist()
        next_obs = observations.cpu().numpy().copy()
        for row, (env, action, running) in enumerate(
            zip(environments, actions, active_rows, strict=True)
        ):
            if running:
                observation, reward, done, _ = env.step(action)
                returns[row] += reward
                steps[row] += 1
                next_obs[row] = [
                    observation[1],
                    observation[2],
                    int(time == corridors[row] - 1),
                    rng.integers(-1, 2),
                ]
                if done:
                    active[row] = False
        observations = torch.as_tensor(next_obs, device=device)
        if not active.any():
            break
    for env in environments:
        env.close()
    return {
        "success_rate": float((returns == 1).mean()),
        "environment_return": float(returns.mean()),
        "mean_episode_steps": float(steps.mean()),
    }
