"""Stable categorical action IDs and memory merge coefficients."""

from dataclasses import dataclass

MERGE_COEFFICIENTS = (0.0, 0.25, 0.5, 0.8, 0.95)


@dataclass(frozen=True)
class ActionSpace:
    """Slot-major IDs: 5*k+i -> (k, alpha[i]); final ID -> global no-op."""

    slots: int = 2

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError("Memory must have at least one slot")

    def __len__(self) -> int:
        return len(MERGE_COEFFICIENTS) * self.slots + 1

    @property
    def noop_id(self) -> int:
        return len(self) - 1

    def decode(self, action_id: int) -> tuple[int | None, float]:
        if not 0 <= action_id < len(self):
            raise ValueError(f"Invalid action ID: {action_id}")
        if action_id == self.noop_id:
            return None, 1.0
        slot, coefficient = divmod(action_id, len(MERGE_COEFFICIENTS))
        return slot, MERGE_COEFFICIENTS[coefficient]

    def actions(self) -> tuple[tuple[int | None, float], ...]:
        return tuple(self.decode(i) for i in range(len(self)))

    def encode(self, slot: int | None, alpha: float) -> int:
        """Inverse of decode; alpha=1 is only valid for the global no-op."""
        if slot is None and alpha == 1:
            return self.noop_id
        if slot is None or not 0 <= slot < self.slots:
            raise ValueError("Invalid slot")
        if alpha not in MERGE_COEFFICIENTS:
            raise ValueError("Invalid merge coefficient")
        return slot * len(MERGE_COEFFICIENTS) + MERGE_COEFFICIENTS.index(alpha)
