"""Runnable controlled Step 2 experiment using the authors' T-Maze and ELMUR."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from elmur_klpo.demonstrations import Demonstrations, tmaze_demonstrations
from elmur_klpo.evaluation import closed_loop
from elmur_klpo.manager import CategoricalManager
from elmur_klpo.memory import ActionSpace
from elmur_klpo.reader import ELMURReader
from elmur_klpo.rollout import Rollout, collect, manager_dimension
from elmur_klpo.trainer import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    update,
)
from elmur_klpo.vendor.elmur.layers.memory import MemoryState


def metrics(batch: Rollout, data: Demonstrations, slots: int) -> dict:
    """Report offline imitation metrics, including the memory-sensitive junction."""
    ids = batch.actions[batch.valid]
    histogram = torch.bincount(ids, minlength=5 * slots + 1).float()
    histogram /= histogram.sum().clamp_min(1)
    query = batch.query_mask
    retention = batch.retention[batch.valid]
    result = {
        "nll": float(-batch.rewards.sum() / data.valid.sum()),
        "accuracy": float(batch.correct.sum() / data.valid.sum()),
        "imitation_return": float(batch.rewards.sum(1).mean()),
        "junction_nll": float(-batch.rewards[query].mean()),
        "junction_accuracy": float(batch.correct[query].float().mean()),
        "action_distribution": histogram.tolist(),
        "alpha_frequencies": histogram[:-1].reshape(slots, 5).sum(0).tolist(),
        "slot_frequencies": histogram[:-1].reshape(slots, 5).sum(1).tolist(),
        "noop_frequency": float(histogram[-1]),
        "mean_memory_change_l2": float(retention.mean()),
        "unchanged_memory_fraction": float((retention == 0).float().mean()),
    }
    if batch.method == "lru":
        result.pop("action_distribution")
        result["alpha_frequencies"] = {"native_new_candidate_weight": 0.79}
    return result


def pretrain(
    reader: ELMURReader, train: Demonstrations, steps: int, batch_size: int
) -> None:
    """Standard LRU behavior cloning, with detached memory between windows.

    Reader and representation are prepared together, then both are frozen for
    manager optimization. No reader loss flows through a previous memory write.
    """
    reader.train().requires_grad_(True)
    optimizer = torch.optim.Adam(reader.parameters(), lr=0.002)
    with tqdm(
        total=steps, desc="Reader tmaze", unit="update", dynamic_ncols=True
    ) as progress:
        for iteration in range(steps):
            indices = torch.randint(
                len(train.actions), (batch_size,), device=train.actions.device
            )
            batch = train.select(indices)
            memory = reader.initial(batch_size)
            total = 0
            for time_index in range(batch.actions.shape[1]):
                logits, _, _, native = reader.step(
                    batch.observations[:, time_index], memory, time_index
                )
                loss = torch.nn.functional.cross_entropy(
                    logits, batch.actions[:, time_index], reduction="none"
                )
                total = (
                    total
                    + (loss * batch.valid[:, time_index]).sum() / batch.valid.sum()
                )
                memory = MemoryState(native.vec.detach(), native.pos)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                reader.parameters(), 5, error_if_nonfinite=True
            )
            optimizer.step()
            if (iteration + 1) % 100 == 0:
                tqdm.write(
                    json.dumps(
                        {"reader_step": iteration + 1, "bc_nll": float(total.detach())}
                    )
                )
            progress.set_postfix(loss=f"{float(total.detach()):.4f}", refresh=False)
            progress.update(1)
    reader.freeze()


def run(config: dict, output: Path, resume: bool = False, device: str = "cpu") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.get("cpu_threads", 1))
    seed_everything(config["data_seed"])
    train = tmaze_demonstrations(
        config["train_episodes"], config["horizon"], config["data_seed"], device
    )
    validation = tmaze_demonstrations(
        config["validation_episodes"],
        config["horizon"],
        config["data_seed"] + 1,
        device,
    )
    test = tmaze_demonstrations(
        config["test_episodes"], config["horizon"], config["data_seed"] + 2, device
    )
    reader = ELMURReader(config["slots"], config["dimension"]).to(device)
    reader_path = output / "reader.pt"
    reader_config = {
        key: config[key]
        for key in (
            "data_seed",
            "train_episodes",
            "horizon",
            "slots",
            "dimension",
            "reader_steps",
            "batch_size",
        )
    }
    start = time.perf_counter()
    if resume and reader_path.exists():
        saved_reader = torch.load(reader_path, map_location=device, weights_only=True)
        if saved_reader["config"] != reader_config:
            raise ValueError("Reader checkpoint configuration mismatch")
        reader.load_state_dict(saved_reader["state"])
        reader.freeze()
    else:
        pretrain(reader, train, config["reader_steps"], config["batch_size"])
        torch.save({"state": reader.state_dict(), "config": reader_config}, reader_path)
    reader_seconds = time.perf_counter() - start
    if resume and (output / "results.json").exists():
        reader_seconds = json.loads((output / "results.json").read_text())[
            "reader_seconds"
        ]
    # Reset before each paired condition to preserve initialization and random draws.
    seed_everything(config["data_seed"] + 100)
    genuine = metrics(
        collect(reader, validation, method="lru"), validation, config["slots"]
    )
    seed_everything(config["data_seed"] + 100)
    ablated = metrics(
        collect(reader, validation, method="lru", ablate=True),
        validation,
        config["slots"],
    )
    dependence = {
        "genuine": genuine,
        "ablated": ablated,
        "junction_nll_improvement": ablated["junction_nll"] - genuine["junction_nll"],
        "passed": genuine["junction_nll"] + config.get("memory_gate_margin", 0.02)
        < ablated["junction_nll"]
        and genuine["junction_accuracy"] > 0.6,
    }
    (output / "memory_dependence.json").write_text(
        json.dumps(dependence, indent=2) + "\n"
    )
    if not dependence["passed"]:
        raise RuntimeError(
            "Reader failed memory-dependence validation; see memory_dependence.json. Increase reader preparation; manager training was not started."
        )
    results = {
        "config": config,
        "reader_seconds": reader_seconds,
        "device": device,
        "torch_version": torch.__version__,
        "memory_dependence": dependence,
        "seeds": [],
    }
    for seed in config["seeds"]:
        entry = {"seed": seed, "methods": {}}
        for method in ("lru", "random", "noop"):
            seed_everything(seed + 10000)
            start = time.perf_counter()
            entry["methods"][method] = metrics(
                collect(reader, test, method=method), test, config["slots"]
            )
            entry["methods"][method]["evaluation_seconds"] = time.perf_counter() - start
            seed_everything(seed + 10000)
            entry["methods"][method]["closed_loop"] = closed_loop(
                reader, test, method=method, seed=seed + 10000
            )
        # LRU uses continuous upstream blending and no global no-op; do not
        # mislabel its update counts as discrete alpha action utilization.
        for algorithm in ("reinforce", "klpo"):
            seed_everything(seed)
            manager = CategoricalManager(
                manager_dimension(reader),
                ActionSpace(config["slots"]),
                seed=seed,
                scale=0.01,
                dtype=torch.float32,
            ).to(device)
            optimizer = torch.optim.Adam(
                manager.parameters(), lr=config["learning_rate"]
            )
            checkpoint_path = output / f"{algorithm}_{seed}.pt"
            begin = 0
            history = []
            seed_everything(seed + 10000)
            before = metrics(collect(reader, test, manager), test, config["slots"])
            seed_everything(seed)
            if resume and checkpoint_path.exists():
                saved = load_checkpoint(checkpoint_path, manager, optimizer)
                if saved["extra"]["config"] != config:
                    raise ValueError("Manager checkpoint configuration mismatch")
                begin = saved["iteration"]
                history = saved["extra"]["history"]
                before = saved["extra"]["before"]
            with tqdm(
                total=config["manager_steps"],
                initial=begin,
                desc=f"Manager tmaze {algorithm} seed {seed}",
                unit="update",
                dynamic_ncols=True,
            ) as progress:
                for iteration in range(begin, config["manager_steps"]):
                    indices = torch.randint(
                        len(train.actions), (config["batch_size"],), device=device
                    )
                    batch_data = train.select(indices)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    batch = collect(reader, batch_data, manager.frozen_copy())
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    rollout_seconds = time.perf_counter() - start
                    diagnostic = update(
                        manager, optimizer, batch, algorithm, config["beta"]
                    )
                    diagnostic.update(
                        {
                            "iteration": iteration + 1,
                            "rollout_seconds": rollout_seconds,
                            "decision_throughput": float(batch.valid.sum())
                            / rollout_seconds,
                            "peak_gpu_bytes": torch.cuda.max_memory_allocated()
                            if device.startswith("cuda")
                            else 0,
                            "rollout_metrics": metrics(
                                batch, batch_data, config["slots"]
                            ),
                        }
                    )
                    if (iteration + 1) % config[
                        "eval_every"
                    ] == 0 or iteration + 1 == config["manager_steps"]:
                        # Evaluation must not alter the next training sample stream.
                        rng = torch.get_rng_state()
                        numpy_rng = np.random.get_state()
                        python_rng = random.getstate()
                        cuda_rng = (
                            torch.cuda.get_rng_state_all()
                            if device.startswith("cuda")
                            else None
                        )
                        seed_everything(seed + 10000)
                        diagnostic["held_out"] = metrics(
                            collect(reader, validation, manager),
                            validation,
                            config["slots"],
                        )
                        torch.set_rng_state(rng)
                        np.random.set_state(numpy_rng)
                        random.setstate(python_rng)
                        if cuda_rng is not None:
                            torch.cuda.set_rng_state_all(cuda_rng)
                    history.append(diagnostic)
                    save_checkpoint(
                        checkpoint_path,
                        manager,
                        optimizer,
                        iteration + 1,
                        {
                            "config": config,
                            "history": history,
                            "before": before,
                            "reader_state": reader.state_dict(),
                        },
                    )
                    progress.set_postfix(
                        gradient=f"{diagnostic['gradient_norm']:.3g}", refresh=False
                    )
                    progress.update(1)
            seed_everything(seed + 10000)
            after = metrics(collect(reader, test, manager), test, config["slots"])
            seed_everything(seed + 10000)
            memory_ablated = metrics(
                collect(reader, test, manager, ablate=True), test, config["slots"]
            )
            entry["methods"][algorithm] = {
                "before": before,
                "after": after,
                "memory_ablated": memory_ablated,
                "history": history,
                "training_trajectories": config["manager_steps"] * config["batch_size"],
                "beta": config["beta"] if algorithm == "klpo" else None,
            }
            seed_everything(seed + 10000)
            entry["methods"][algorithm]["closed_loop"] = closed_loop(
                reader, test, manager, seed=seed + 10000
            )
        results["seeds"].append(entry)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "seed": seed,
                    "junction_accuracy": {
                        name: value.get("after", value)["junction_accuracy"]
                        for name, value in entry["methods"].items()
                    },
                }
            ),
            flush=True,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/step2_smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/step2_smoke"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("step") == 3:
        from elmur_klpo.step3.experiment import run as run_step3

        result = run_step3(config, args.output, args.resume, args.device)
        if any(
            task["status"] in {"reader_gate_failed", "native_task_unavailable"}
            for task in result["tasks"].values()
        ):
            parser.exit(
                2,
                "Step 3 stopped at a validation or native-runtime gate; inspect the saved results.\n",
            )
    else:
        run(config, args.output, args.resume, args.device)


if __name__ == "__main__":
    main()
