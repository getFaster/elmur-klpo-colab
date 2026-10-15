"""Atomic JSON artifacts shared by the independently runnable stages."""

import json
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    """Publish a completed artifact atomically, rejecting nonfinite measurements."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
