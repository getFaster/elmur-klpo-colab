"""Resumable upstream fixed-LRU imitation preparation; legacy checkpoints rejected."""

import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from elmur_klpo.step3.artifacts import write_json
from elmur_klpo.step3.reader_data import ReaderData
from elmur_klpo.step3.reader_evaluation import validation
from elmur_klpo.step3.reader_training import (
    effective_epochs,
    make_optimizer,
    section_update,
)
from elmur_klpo.step3.reference import (
    BUDGETS,
    COMMIT,
    configuration_diff,
    reference_settings,
    verify_sources,
    write_event_contract,
)
from elmur_klpo.step3.upstream_reader import FixedLRUReader
from elmur_klpo.trainer import seed_everything
from elmur_klpo.vendor.elmur.layers.memory import MemoryState

SCHEMA = "step3-upstream-fixed-lru-v1"


class VerificationInterruption(RuntimeError):
    """Deliberate section-boundary interruption for engineering resume checks."""


def reader_signature(config: dict) -> dict:
    return {"schema": SCHEMA, "commit": COMMIT, "config": config}


def validate_config(config: dict) -> None:
    """Full runs must match the pinned model/optimizer and exact requested budget."""
    verify_sources()
    task = config["task"]["name"]
    settings = config["preparation"]
    if config.get("protocol") != SCHEMA:
        raise ValueError(
            "Legacy Step 3 configuration incompatible; use repaired upstream configs and a fresh output directory"
        )
    if settings["sections"] != 3 or settings["optimizer_updates"] < 1:
        raise ValueError(
            "Requires exactly three sections and a positive optimizer update budget"
        )
    if (
        settings["batch_size"] < 1
        or settings["eval_every"] < 1
        or settings["checkpoint_every"] < 1
    ):
        raise ValueError(
            "Batch size and checkpoint/validation intervals must be positive"
        )
    if settings["source"] != (
        "offline_demonstrations" if task == "remember_color" else "fresh_native_oracle"
    ):
        raise ValueError("Task-specific offline imitation source required")
    if config["task"]["target"] != (52 if task == "autoencode" else 5):
        raise ValueError(
            "Main preparation must use the full native target distribution"
        )
    expected = reference_settings(task)
    if config["reader"] != expected["model"]:
        raise ValueError(
            "Reader architecture must match the pinned native model settings"
        )
    for key, value in expected["training"].items():
        if key == "online_inference" and settings.get(key) is False:
            continue  # Offline BC validation is deliberate; renderer is optional.
        if settings.get(key) != value and not (
            config.get("verification_only") and key == "batch_size"
        ):
            raise ValueError(f"Training setting differs from pinned reference: {key}")
    if (
        not config.get("verification_only")
        and settings["optimizer_updates"] != BUDGETS[task]
    ):
        raise ValueError(
            "Full preparation requires the exact upstream optimizer update budget"
        )
    if settings["optimizer_updates"] > BUDGETS[task]:
        raise ValueError("Optimizer budget exceeds upstream reproduction budget")
    if (
        task == "remember_color"
        and settings["demonstrations"]["action_target"] != "raw"
    ):
        raise ValueError("Raw expert actions required; do not clip training targets")


def _rng() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save(
    path: Path,
    reader: FixedLRUReader,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    state: dict,
    signature: dict,
    dataset_signature: dict,
) -> None:
    snapshot = copy.deepcopy(state)
    if snapshot["memories"] is not None:
        snapshot["memories"] = [
            (m.vec.detach().cpu(), m.pos.detach().cpu()) for m in snapshot["memories"]
        ]
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema": SCHEMA,
            "signature": signature,
            "dataset_signature": dataset_signature,
            "reader": reader.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": snapshot,
            "rng": _rng(),
        },
        temporary,
    )
    temporary.replace(path)


def _load(
    path: Path,
    reader: FixedLRUReader,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    signature: dict,
    dataset_signature: dict,
) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") != SCHEMA:
        raise ValueError("Legacy reader checkpoint incompatible with upstream repair")
    if (
        saved["signature"] != signature
        or saved["dataset_signature"] != dataset_signature
    ):
        raise ValueError("Reader checkpoint configuration or dataset mismatch")
    reader.load_state_dict(saved["reader"])
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    state = saved["state"]
    if state["memories"] is not None:
        device = next(reader.parameters()).device
        state["memories"] = [
            MemoryState(v.to(device), p.to(device)) for v, p in state["memories"]
        ]
    _restore_rng(saved["rng"])
    return state


def prepare_reader(
    config: dict,
    output: Path,
    resume: bool = False,
    device: str = "cpu",
    *,
    method: str = "lru",
    fixed_alpha: float | None = None,
    verification_interrupt_after: int | None = None,
) -> tuple[FixedLRUReader, None, dict]:
    """Exact section updates; persist pending batch, layer memories, scheduler/RNG."""
    if method != "lru" or fixed_alpha is not None:
        raise RuntimeError(
            "Learned-manager reader preparation is deferred until reproduction and readiness gates"
        )
    validate_config(config)
    if verification_interrupt_after is not None and not config.get("verification_only"):
        raise ValueError("Simulated interruption allowed only in verification smoke")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "reader_last.pt"
    if not resume and (path.exists() or (output / "reader.pt").exists()):
        raise FileExistsError("Use --resume or a fresh output directory")
    if resume and not path.exists():
        if (output / "reader.pt").exists():
            raise ValueError(
                "Legacy reader checkpoint incompatible with upstream repair"
            )
        raise FileNotFoundError(path)
    data = ReaderData(config)
    settings = {
        **config["preparation"],
        "batches_per_epoch": math.ceil(
            data.train_size / config["preparation"]["batch_size"]
        ),
    }
    seed_everything(config["reader_seed"])
    reader = FixedLRUReader(config["task"], config["reader"]).to(device)
    optimizer, scheduler = make_optimizer(reader, settings)
    signature = reader_signature(config)
    if resume:
        state = _load(path, reader, optimizer, scheduler, signature, data.signature)
    else:
        tqdm.write(f"Reader {config['task']['name']}: initial validation...")
        initial_start = time.perf_counter()
        initial_validation = validation(reader, data, config)
        initial_seconds = time.perf_counter() - initial_start
        state = {
            "completed_optimizer_steps": 0,
            "epoch": 0,
            "order": torch.randperm(data.train_size).tolist(),
            "cursor": 0,
            "pending": None,
            "next_section": 0,
            "memories": None,
            "history": [],
            "validation_history": [{"optimizer_steps": 0, **initial_validation}],
            "best_validation": initial_validation["selection_loss"],
            "best_step": 0,
            "unique_episodes": [],
            "sample_presentations": 0,
            "native_environment_interactions": data.native_environment_interactions,
            "materialized_episodes": data.materialized_episodes,
            "reader_seconds": 0.0,
            "gpu_step_seconds": 0.0,
            "validation_seconds": initial_seconds,
            "validation_forward_passes": initial_validation["model_forward_passes"],
            "training_forward_passes": 0,
            "sequence_forward_presentations": 0,
            "transition_presentations": 0,
            "token_presentations_including_padding": 0,
            "peak_gpu_bytes": 0,
        }
        _save(path, reader, optimizer, scheduler, state, signature, data.signature)
        _save(
            output / "reader_best.pt",
            reader,
            optimizer,
            scheduler,
            state,
            signature,
            data.signature,
        )
    write_json(output / "configuration_match.json", configuration_diff(config))
    write_json(
        output / "write_event_contract.json",
        write_event_contract(config["reader"], settings["context_length"]),
    )
    write_json(output / "dataset_splits.json", data.signature)
    target = settings["optimizer_updates"]
    accounted_interactions = data.native_environment_interactions
    accounted_materializations = data.materialized_episodes
    batch = None
    unique = set(state["unique_episodes"])
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with tqdm(
        total=target,
        initial=state["completed_optimizer_steps"],
        desc=f"Reader {config['task']['name']}",
        unit="update",
        dynamic_ncols=True,
    ) as progress:
        while state["completed_optimizer_steps"] < target:
            if state["pending"] is None:
                end = min(state["cursor"] + settings["batch_size"], data.train_size)
                state["pending"] = state["order"][state["cursor"] : end]
                state["cursor"] = end
                state["next_section"] = 0
                batch = data.load(state["pending"], "train", state["epoch"])
                state["memories"] = reader.initial(len(batch.identities))
                if data.dataset is not None:
                    unique.update(batch.identities)
                    state["unique_episodes"] = sorted(unique)
                state["sample_presentations"] += len(batch.identities)
            elif batch is None:
                batch = data.load(state["pending"], "train", state["epoch"])
            section = state["next_section"]
            chunk = batch.section(
                section * settings["context_length"], settings["context_length"]
            )
            start = time.perf_counter()
            cuda_start = cuda_end = None
            if torch.device(device).type == "cuda":
                cuda_start, cuda_end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                cuda_start.record()
            memory, row = section_update(
                reader,
                optimizer,
                scheduler,
                chunk,
                state["memories"],
                section * settings["context_length"],
                settings,
                state,
            )
            if torch.device(device).type == "cuda":
                cuda_end.record()
                torch.cuda.synchronize(device)
                state["gpu_step_seconds"] = (
                    state.get("gpu_step_seconds", 0.0)
                    + cuda_start.elapsed_time(cuda_end) / 1000
                )
                state["peak_gpu_bytes"] = max(
                    state["peak_gpu_bytes"], torch.cuda.max_memory_allocated(device)
                )
            state["reader_seconds"] += time.perf_counter() - start
            state["memories"] = memory
            state["next_section"] += 1
            steps = state["completed_optimizer_steps"]
            progress.set_postfix(
                loss=f"{row['action_loss']:.4f}",
                lr=f"{row['learning_rate_used']:.2e}",
                section=f"{section + 1}/3",
                refresh=False,
            )
            progress.update(steps - progress.n)
            row.update(
                optimizer_steps=steps,
                section=section,
                epoch=state["epoch"],
                batch_id=scheduler.minibatches,
                trajectories=batch.identities if section == 0 else [],
            )
            state["training_forward_passes"] += 1
            state["sequence_forward_presentations"] += len(batch.identities)
            state["transition_presentations"] += row["valid_transitions"]
            state["token_presentations_including_padding"] += row[
                "sample_token_presentations"
            ]
            state["history"].append(row)
            if state["next_section"] == 3:
                epoch_end = state["cursor"] == data.train_size
                scheduler.end_minibatch(epoch_end)
                state["pending"], state["memories"], batch = None, None, None
                state["next_section"] = 0
                if epoch_end:
                    state["epoch"] += 1
                    state["cursor"] = 0
                    state["order"] = torch.randperm(data.train_size).tolist()
            improved = False
            if steps % settings["eval_every"] == 0 or steps == target:
                start = time.perf_counter()
                result = validation(reader, data, config)
                state["validation_seconds"] += time.perf_counter() - start
                state["validation_forward_passes"] += result["model_forward_passes"]
                state["validation_history"].append({"optimizer_steps": steps, **result})
                if result["selection_loss"] < state["best_validation"]:
                    state["best_validation"], state["best_step"] = (
                        result["selection_loss"],
                        steps,
                    )
                    improved = True
            state["native_environment_interactions"] = (
                state.get("native_environment_interactions", 0)
                + data.native_environment_interactions
                - accounted_interactions
            )
            state["materialized_episodes"] = (
                state.get("materialized_episodes", 0)
                + data.materialized_episodes
                - accounted_materializations
            )
            accounted_interactions = data.native_environment_interactions
            accounted_materializations = data.materialized_episodes
            if (
                steps % settings["checkpoint_every"] == 0
                or steps % settings["eval_every"] == 0
                or steps == target
            ):
                _save(
                    path, reader, optimizer, scheduler, state, signature, data.signature
                )
                write_json(output / "history.json", state["history"])
                write_json(
                    output / "progress.json",
                    {
                        "completed_optimizer_steps": steps,
                        "target_optimizer_steps": target,
                        "next_section": state["next_section"],
                        "best_step": state["best_step"],
                    },
                )
            if improved:
                _save(
                    output / "reader_best.pt",
                    reader,
                    optimizer,
                    scheduler,
                    state,
                    signature,
                    data.signature,
                )
            if steps == verification_interrupt_after:
                _save(
                    path, reader, optimizer, scheduler, state, signature, data.signature
                )
                raise VerificationInterruption(
                    "Simulated interruption after saved optimizer step"
                )
            if steps % settings["eval_every"] == 0 or steps == target:
                tqdm.write(
                    f"{config['task']['name']}: completed optimizer steps {steps}/{target}, validation {result['metric']}={result['selection_loss']:.4f}"
                )
    report = {
        "schema": SCHEMA,
        "commit": COMMIT,
        "method": "independent_upstream_fixed_lru",
        "verification_only": config.get("verification_only", False),
        "completed_optimizer_steps": state["completed_optimizer_steps"],
        "target_optimizer_steps": target,
        "best_step": state["best_step"],
        "best_validation_loss": state["best_validation"],
        "last_validation": state["validation_history"][-1],
        "validation_history": state["validation_history"],
        "history": state["history"],
        "unique_trajectories_seen": len(unique)
        if data.dataset is not None
        else state["sample_presentations"],
        "sample_presentations": state["sample_presentations"],
        "effective_epochs": effective_epochs(
            target, data.train_size, settings["batch_size"]
        ),
        "train_split_trajectories": data.train_size,
        "reader_seconds": state["reader_seconds"],
        "gpu_step_seconds": state.get("gpu_step_seconds", 0.0),
        "drop_last": False,
        "best_validation": next(
            row
            for row in state["validation_history"]
            if row["optimizer_steps"] == state["best_step"]
        ),
        "validation_seconds": state["validation_seconds"],
        "training_forward_passes": state["training_forward_passes"],
        "validation_forward_passes": state["validation_forward_passes"],
        "sequence_forward_presentations": state["sequence_forward_presentations"],
        "transition_presentations": state["transition_presentations"],
        "token_presentations_including_padding": state[
            "token_presentations_including_padding"
        ],
        "peak_gpu_bytes": state["peak_gpu_bytes"],
        "device": device,
        "environment_interactions": state.get("native_environment_interactions", 0),
        "materialized_episodes_including_validation_and_resume": state.get(
            "materialized_episodes", 0
        ),
        "readiness": {
            "passed": False,
            "failure": "Separate bootstrap reader memory-dependence gate not executed; upstream reproduction learning gate also requires full-budget evaluation",
        },
        "checkpoint_last": str(path),
        "checkpoint_best": str(output / "reader_best.pt"),
        "test_split_used": False,
    }
    write_json(output / "reader_report.json", report)
    return reader.freeze(), None, report


def restore_reader(
    config: dict, path: Path, device: str = "cpu"
) -> tuple[FixedLRUReader, None, dict]:
    """Restore only schema-compatible fixed-LRU inference weights and saved report."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") != SCHEMA or saved.get("signature") != reader_signature(
        config
    ):
        raise ValueError("Legacy or incompatible reader checkpoint")
    reader = FixedLRUReader(config["task"], config["reader"]).to(device)
    reader.load_state_dict(saved["reader"])
    return (
        reader.freeze(),
        None,
        {
            "readiness": {
                "passed": False,
                "failure": "Manager event integration deferred",
            }
        },
    )
