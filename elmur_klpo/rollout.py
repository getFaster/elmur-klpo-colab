"""Batched frozen-reader trajectories with fixed recorded policy states."""

from dataclasses import dataclass

import torch
from torch import Tensor

from elmur_klpo.demonstrations import Demonstrations
from elmur_klpo.manager import CategoricalManager
from elmur_klpo.memory import ActionSpace
from elmur_klpo.reader import ELMURReader
from elmur_klpo.training_memory import write_vectors
from elmur_klpo.vendor.elmur.layers.memory import MemoryState


@dataclass
class Rollout:
    states: Tensor
    actions: Tensor
    log_q: Tensor
    q_probs: Tensor
    chosen_log_q: Tensor
    returns: Tensor
    valid: Tensor
    episode_ids: Tensor
    rewards: Tensor
    correct: Tensor
    retention: Tensor
    query_mask: Tensor
    method: str


def features(
    context: Tensor, memory: MemoryState, candidate: Tensor, time: int
) -> Tensor:
    """Flatten all [N, slots, width] candidates into detached memory mangement policy inputs."""
    return torch.cat(
        (
            context.new_ones(context.shape[0], 1),
            context,
            memory.vec.flatten(1),
            candidate.flatten(1),
            (time - memory.pos).to(context.dtype) / 100,
            context.new_full((context.shape[0], 1), time / 100),
        ),
        -1,
    ).detach()


def manager_dimension(reader: ELMURReader) -> int:
    return (
        (2 * reader.model.memory_size + 1) * reader.model.d_model
        + reader.model.memory_size
        + 2
    )


@torch.no_grad()
def collect(
    reader: ELMURReader,
    data: Demonstrations,
    policy: CategoricalManager | None = None,
    method: str = "learned",
    ablate: bool = False,
) -> Rollout:
    """Read before write; final valid reward has no manager decision.

    Padding stays within its episode row. The reverse sum excludes the current
    reward, and the decision mask excludes terminal actions and padding.
    """
    reader.freeze()
    if method not in {"learned", "lru", "random", "noop"}:
        raise ValueError("Unknown management method")
    memory = reader.initial(len(data.actions))
    space = ActionSpace(reader.model.memory_size)
    records = []
    for time in range(data.actions.shape[1]):
        read_memory = (
            MemoryState(torch.zeros_like(memory.vec), memory.pos) if ablate else memory
        )
        logits, context, candidate, native = reader.step(
            data.observations[:, time], read_memory, time
        )
        state = features(context, memory, candidate, time)
        if method == "learned":
            if policy is None:
                raise ValueError("Learned management requires a policy")
            log_q = policy.log_probs(state)
        else:
            log_q = state.new_full(
                (len(state), len(space)),
                -torch.log(state.new_tensor(float(len(space)))),
            )
        ids = torch.multinomial(log_q.exp(), 1).squeeze(-1)
        if method == "noop":
            ids.fill_(space.noop_id)
        new_vec = write_vectors(memory.vec, candidate, ids)
        new_pos = memory.pos.clone()
        selected = (
            torch.arange(space.slots, device=ids.device)[None] == (ids // 5)[:, None]
        )
        new_pos = torch.where(selected, time, new_pos)
        if method == "lru":
            new_vec, new_pos = native.vec.detach(), native.pos
            ids = (
                torch.where(
                    new_pos != memory.pos,
                    torch.arange(space.slots, device=ids.device)[None],
                    -1,
                )
                .amax(-1)
                .clamp_min(0)
                * 5
            )
        active = data.valid[:, time]
        decision = active & (
            data.valid[:, time + 1]
            if time + 1 < data.valid.shape[1]
            else torch.zeros_like(active)
        )
        change = torch.linalg.vector_norm(new_vec - memory.vec, dim=(-1, -2))
        reward = (
            logits.log_softmax(-1).gather(-1, data.actions[:, time, None]).squeeze(-1)
        )
        records.append(
            (
                state,
                ids,
                log_q,
                reward * active,
                (logits.argmax(-1) == data.actions[:, time]) & active,
                change,
            )
        )
        memory = MemoryState(
            torch.where(decision[:, None, None], new_vec, memory.vec).detach(),
            torch.where(decision[:, None], new_pos, memory.pos),
        )
    states, ids, log_q, rewards, correct, retention = (
        torch.stack(values, 1) for values in zip(*records, strict=True)
    )
    future = torch.cat((rewards[:, 1:], torch.zeros_like(rewards[:, :1])), 1)
    returns = future.flip(1).cumsum(1).flip(1)
    decisions = data.valid & torch.cat(
        (data.valid[:, 1:], torch.zeros_like(data.valid[:, :1])), 1
    )
    query_mask = data.valid & ~decisions
    return Rollout(
        states,
        ids,
        log_q,
        log_q.exp(),
        log_q.gather(-1, ids[..., None]).squeeze(-1),
        returns,
        decisions,
        data.episode_ids,
        rewards,
        correct,
        retention,
        query_mask,
        method,
    )
