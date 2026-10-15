"""Validated task schedules and disjoint environment seed streams."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    updates: int
    difficulty: int


class Curriculum:
    """A finite ordered schedule; the final phase must expose target difficulty."""

    def __init__(self, phases: list[dict], target: int) -> None:
        self.phases = tuple(Phase(**phase) for phase in phases)
        if not self.phases or any(p.updates < 1 for p in self.phases):
            raise ValueError("Curriculum requires positive phase budgets")
        if self.phases[-1].difficulty != target:
            raise ValueError("Final curriculum phase must match target difficulty")
        if any(
            a.difficulty > b.difficulty for a, b in zip(self.phases, self.phases[1:])
        ):
            raise ValueError("Difficulty must not decrease")

    @property
    def updates(self) -> int:
        return sum(p.updates for p in self.phases)

    def at(self, iteration: int) -> Phase:
        if not 0 <= iteration < self.updates:
            raise ValueError("Iteration outside curriculum")
        for phase in self.phases:
            if iteration < phase.updates:
                return phase
            iteration -= phase.updates
        raise AssertionError("Unreachable")


def environment_seeds(split: str, start: int, count: int, base: int = 0) -> list[int]:
    """Disjoint 100-million-seed blocks, independent of manager sampling seeds."""
    offsets = {
        "train": 0,
        "validation": 100_000_000,
        "test": 200_000_000,
        "robustness": 300_000_000,
    }
    if (
        split not in offsets
        or start < 0
        or count < 1
        or start + count + base > 100_000_000
        or base < 0
    ):
        raise ValueError("Invalid split or seed range")
    return list(
        range(offsets[split] + base + start, offsets[split] + base + start + count)
    )
