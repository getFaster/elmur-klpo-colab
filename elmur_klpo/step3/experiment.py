"""Upstream Step 3 reproduction entry point; later manager stages remain gated."""

import argparse
import json
from pathlib import Path

import torch

from elmur_klpo.step3.artifacts import write_json
from elmur_klpo.step3.preparation import prepare_reader, validate_config


def run_task(
    config: dict,
    output: Path,
    stage: str,
    resume: bool,
    device: str,
    reader_root: Path | None = None,
) -> dict:
    """Only fixed-LRU preparation and engineering smoke are authorized here."""
    validate_config(config)
    if stage not in ("prepare", "smoke"):
        raise RuntimeError(
            "Learned-manager stages blocked: upstream reproduction/readiness gates and event-level integration remain unfinished (execution items 4–6)"
        )
    if stage == "smoke":
        from elmur_klpo.step3.upstream_smoke import smoke_task

        return smoke_task(config, output, resume, device)
    if config.get("verification_only"):
        raise ValueError(
            "Engineering verification configuration cannot run full reader preparation"
        )
    _, _, report = prepare_reader(config, output, resume, device)
    result = {"status": "reader_preparation_complete_gates_pending", "reader": report}
    write_json(output / "prepare_results.json", result)
    return result


def run(
    config: dict,
    output: Path,
    resume: bool = False,
    device: str = "cpu",
    *,
    stage: str | None = None,
    task: str | None = None,
    reader_root: Path | None = None,
) -> dict:
    """Persist per-task results without launching manager comparisons."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch environment")
    torch.set_num_threads(config.get("cpu_threads", 1))
    selected = [
        entry
        for entry in config["tasks"]
        if task is None or entry["task"]["name"] == task
    ]
    if not selected:
        raise ValueError("No matching task configuration")
    results = {}
    for entry in selected:
        name = entry["task"]["name"]
        results[name] = run_task(
            entry, output / name, stage or config["stage"], resume, device, reader_root
        )
        print(json.dumps({"task": name, "status": results[name]["status"]}), flush=True)
    result = {
        "tasks": results,
        "note": "Engineering smoke does not establish reproduction learning or reader readiness; no manager conclusions",
    }
    write_json(output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "prepare",
            "train",
            "evaluate",
            "compare",
            "fixed",
            "robustness",
        ),
    )
    parser.add_argument("--task", choices=("autoencode", "remember_color"))
    parser.add_argument("--reader-root", type=Path)
    args = parser.parse_args()
    run(
        json.loads(args.config.read_text()),
        args.output,
        args.resume,
        args.device,
        stage=args.stage,
        task=args.task,
        reader_root=args.reader_root,
    )


if __name__ == "__main__":
    main()
