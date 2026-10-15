"""Fresh native-reward trajectories for the shared Step 2 REINFORCE update."""

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from elmur_klpo.rollout import Rollout, features
from elmur_klpo.step3.policies import BlendingManager, transition
from elmur_klpo.step3.reader import TaskReader
from elmur_klpo.step3.tasks import make_task
from elmur_klpo.vendor.elmur.layers.memory import MemoryState


def discounted_future(rewards: Tensor, valid: Tensor, gamma: float) -> Tensor:
    """Detached [N,T] RTG: exclude r_t, stop at episode end, mask padding."""
    if rewards.shape != valid.shape or not 0 <= gamma <= 1:
        raise ValueError("Invalid rewards/mask/discount")
    returns = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[:, 0])
    for index in range(rewards.shape[1] - 1, -1, -1):
        returns[:, index] = carry
        carry = torch.where(valid[:, index], rewards[:, index] + gamma * carry, 0)
    return returns.detach()


@dataclass
class NativeRollout:
    batch: Rollout
    episode_returns: list[float]
    successes: list[bool] | None
    accuracies: list[float] | None
    traces: list[dict]
    compute: dict


@torch.no_grad()
def collect_native(
    reader: TaskReader,
    task_config: dict,
    seeds: list[int],
    manager: BlendingManager,
    *,
    method: str = "learned",
    fixed_alpha: float | None = None,
    greedy: bool = False,
    ablate: bool = False,
    informative: bool = False,
    gamma: float = 1.0,
) -> NativeRollout:
    """Closed-loop memory and native task reward; frozen reader, fixed score states.

    Native termination is checked before committing a write. Candidate/state
    computation uses only the pre-action observation and memory. No manager
    action is recorded at terminal or padded timesteps. Informative memory is
    an explicitly privileged decoder diagnostic using earlier encoder writes.
    """
    if method not in {"learned", "fixed", "lru", "noop"}:
        raise ValueError("Unknown management method")
    reader.freeze()
    device = next(reader.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    tasks = []
    start = time.perf_counter()
    try:
        for _ in seeds:
            tasks.append(make_task(task_config))
        observations = np.stack(
            [task.reset(seed) for task, seed in zip(tasks, seeds, strict=True)]
        )
        memory = reader.initial(len(tasks))
        active = torch.ones(len(tasks), dtype=torch.bool, device=device)
        records = []
        traces = []
        correct = [[] for _ in tasks]
        success = [False] * len(tasks)
        success_defined = False
        snapshots: list[list[MemoryState]] = [[] for _ in tasks]
        steps = forwards = 0
        for time_index in range(max(task.max_steps for task in tasks)):
            observation = torch.as_tensor(
                observations, device=device, dtype=torch.float32
            )
            read_memory = memory
            if informative:
                vec, pos = memory.vec.clone(), memory.pos.clone()
                for row, task in enumerate(tasks):
                    if active[row] and task.event() != "cue":
                        # At recall the remaining native hand identifies the
                        # earlier observed card, without constructing new vectors.
                        index = (
                            len(task.env.deck["system"]) - 1
                            if task_config["name"] == "autoencode"
                            else len(snapshots[row]) - 1
                        )
                        if 0 <= index < len(snapshots[row]):
                            vec[row], pos[row] = snapshots[row][index]
                read_memory = MemoryState(vec, pos)
            if ablate:
                read_memory = MemoryState(torch.zeros_like(memory.vec), memory.pos)
            logits, context, candidate, native = reader.step(
                observation, read_memory, time_index
            )
            forwards += len(tasks)
            state = features(context, memory, candidate, time_index)
            log_q = manager.log_probs(state)
            actions, _ = reader.task_action(logits, greedy=True)
            rewards = state.new_zeros(len(tasks))
            accuracy = torch.zeros_like(active)
            ended = ~active.clone()
            events = [task.event() for task in tasks]
            for row, task in enumerate(tasks):
                if not active[row]:
                    continue
                obs, reward, done, info = task.step(actions[row].cpu().numpy())
                observations[row] = obs
                rewards[row] = reward
                ended[row] = done
                steps += 1
                if info.get("accuracy") is not None:
                    correct[row].append(info["accuracy"])
                    accuracy[row] = bool(info["accuracy"])
                if info.get("success") is not None:
                    success_defined = True
                    success[row] |= info["success"]
            decision = active & ~ended
            ids = torch.full(
                (len(tasks),),
                manager.action_space.noop_id,
                device=device,
                dtype=torch.long,
            )
            if decision.any():
                if method == "learned":
                    if greedy and manager.variant == "coefficient_only":
                        # Keep slots uniform when evaluating coefficient learning.
                        coeff = torch.cat(
                            (
                                log_q[decision, :5]
                                + np.log(manager.action_space.slots),
                                log_q[decision, -1:],
                            ),
                            -1,
                        ).argmax(-1)
                        slots = torch.randint(
                            manager.action_space.slots, coeff.shape, device=device
                        )
                        ids[decision] = torch.where(
                            coeff == 5, manager.action_space.noop_id, slots * 5 + coeff
                        )
                    else:
                        ids[decision] = (
                            log_q[decision].argmax(-1)
                            if greedy
                            else torch.multinomial(log_q[decision].exp(), 1).squeeze(-1)
                        )
                elif method == "fixed":
                    ids[decision] = (
                        torch.randint(
                            manager.action_space.slots,
                            (int(decision.sum()),),
                            device=device,
                        )
                        * 5
                    )
            new_memory = transition(
                memory,
                candidate,
                ids,
                time_index,
                fixed_alpha=fixed_alpha if method == "fixed" else None,
            )
            if method == "lru":
                new_memory = MemoryState(native.vec.detach(), native.pos)
            # Snapshots are real candidate writes into a valid bank. The decoder
            # diagnostic selects a past snapshot rather than injecting vectors.
            if informative:
                full_write = transition(
                    memory,
                    candidate,
                    torch.zeros_like(ids),
                    time_index,
                    fixed_alpha=0.0,
                )
                for row, event in enumerate(events):
                    if active[row] and event == "cue":
                        snapshots[row].append(
                            MemoryState(
                                full_write.vec[row].clone(), full_write.pos[row].clone()
                            )
                        )
            change = torch.linalg.vector_norm(new_memory.vec - memory.vec, dim=(-1, -2))
            for row in decision.nonzero().flatten().tolist():
                action_id = int(ids[row])
                slot, alpha = manager.action_space.decode(action_id)
                if method == "lru":
                    changed = (native.pos[row] != memory.pos[row]).nonzero().flatten()
                    slot = int(changed[0]) if len(changed) else None
                    alpha = None  # upstream empty-slot behavior is separate
                elif method == "fixed":
                    alpha = fixed_alpha
                traces.append(
                    {
                        "episode": row,
                        "time": time_index,
                        "progress": time_index / tasks[row].max_steps,
                        "difficulty": tasks[row].difficulty,
                        "event": events[row],
                        "slot": slot,
                        "alpha": alpha,
                        "age": None
                        if slot is None
                        else int(time_index - memory.pos[row, slot]),
                        "overwrite": slot is not None
                        and int(memory.pos[row, slot]) >= 0,
                        "memory_change_l2": float(change[row]),
                        "noop": method == "noop"
                        or (method == "learned" and slot is None),
                        "coefficient_probs": log_q[row, :-1]
                        .exp()
                        .reshape(manager.action_space.slots, 5)
                        .sum(0)
                        .tolist()
                        if method == "learned"
                        else None,
                    }
                )
            records.append(
                (state, ids, log_q, rewards, accuracy, change, active.clone(), decision)
            )
            memory = MemoryState(
                torch.where(decision[:, None, None], new_memory.vec, memory.vec),
                torch.where(decision[:, None], new_memory.pos, memory.pos),
            )
            active = ~ended
            if not active.any():
                break
        if active.any():
            raise RuntimeError(
                "Native environment did not terminate within its declared horizon"
            )
        states, ids, log_q, rewards, accuracy, changes, valid, decisions = (
            torch.stack(items, 1) for items in zip(*records, strict=True)
        )
        batch = Rollout(
            states,
            ids,
            log_q,
            log_q.exp(),
            log_q.gather(-1, ids[..., None]).squeeze(-1),
            discounted_future(rewards, valid, gamma),
            decisions,
            torch.tensor(seeds, device=device),
            rewards,
            accuracy,
            changes,
            valid & ~decisions,
            "learned" if method == "learned" else method,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        return NativeRollout(
            batch,
            rewards.sum(1).tolist(),
            success if success_defined else None,
            [float(np.mean(values)) for values in correct] if all(correct) else None,
            traces,
            {
                "rollout_seconds": seconds,
                "environment_steps": steps,
                "environment_steps_per_second": steps / seconds,
                "reader_forward_passes": forwards,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
            },
        )
    finally:
        for task in tasks:
            task.close()
