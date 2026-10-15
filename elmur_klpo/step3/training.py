"""Stage C frozen-reader manager optimization with the verified REINFORCE core."""

import hashlib
from pathlib import Path

import torch
from tqdm import tqdm

from elmur_klpo.rollout import manager_dimension
from elmur_klpo.step3.curriculum import environment_seeds
from elmur_klpo.step3.diagnostics import summarize
from elmur_klpo.step3.evaluation import evaluate
from elmur_klpo.step3.policies import BlendingManager
from elmur_klpo.step3.reader import TaskReader
from elmur_klpo.step3.rollout import collect_native
from elmur_klpo.trainer import load_checkpoint, save_checkpoint, seed_everything, update


def reader_digest(reader: TaskReader) -> str:
    """Fingerprint all frozen reader/encoder tensors for checkpoint compatibility."""
    digest = hashlib.sha256()
    for name, tensor in reader.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_manager(
    config: dict,
    output: Path,
    reader: TaskReader,
    readiness_report: dict,
    seed: int,
    variant: str,
    resume: bool = False,
) -> tuple[BlendingManager, dict]:
    """One Adam step per fresh batch; refuse failed reader readiness gates."""
    if not readiness_report["passed"]:
        raise RuntimeError(
            "Reader readiness gate failed; manager training was not started"
        )
    reader.freeze()
    seed_everything(seed)
    device = next(reader.parameters()).device
    manager = BlendingManager(
        manager_dimension(reader), config["reader"]["slots"], variant
    ).to(device)
    optimizer = torch.optim.Adam(
        manager.parameters(), lr=config["manager"]["learning_rate"]
    )
    fingerprint = reader_digest(reader)
    signature = {
        "reader_sha256": fingerprint,
        "config": config,
        "seed": seed,
        "variant": variant,
    }
    path = output / f"manager_{variant}_{seed}.pt"
    if path.exists() and not resume:
        raise FileExistsError(f"Use --resume or a fresh output directory: {path}")
    history = []
    begin = 0
    if resume:
        if not path.exists():
            raise FileNotFoundError(f"Missing manager checkpoint: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["extra"]["signature"] != signature:
            raise ValueError("Manager checkpoint configuration or reader mismatch")
        saved = load_checkpoint(path, manager, optimizer)
        begin, history = saved["iteration"], saved["extra"]["history"]
    validation = environment_seeds(
        "validation", 0, config["evaluation"]["episodes"], config["data_seed"]
    )
    with tqdm(
        total=config["manager"]["updates"],
        initial=begin,
        desc=f"Manager {config['task']['name']} {variant} seed {seed}",
        unit="update",
        dynamic_ncols=True,
    ) as progress:
        for iteration in range(begin, config["manager"]["updates"]):
            batch_size = config["manager"]["batch_size"]
            seeds = environment_seeds(
                "train",
                1_000_000 + seed * 100_000 + iteration * batch_size,
                batch_size,
                config["data_seed"],
            )
            rollout = collect_native(
                reader,
                config["task"],
                seeds,
                manager.frozen_copy(),
                gamma=config["manager"]["gamma"],
            )
            diagnostic = update(
                manager,
                optimizer,
                rollout.batch,
                "reinforce",
                clip=config["manager"]["gradient_clip"],
            )
            diagnostic.update(
                iteration=iteration + 1,
                environment_seeds=seeds,
                compute=rollout.compute,
                training=summarize(rollout, config["reader"]["slots"]),
            )
            if any(p.grad is not None or p.requires_grad for p in reader.parameters()):
                raise AssertionError("Reader gradient isolation violated")
            if (iteration + 1) % config["manager"][
                "eval_every"
            ] == 0 or iteration + 1 == config["manager"]["updates"]:
                diagnostic["validation"] = evaluate(
                    reader,
                    config["task"],
                    validation,
                    manager,
                    policy_seed=seed + 10000,
                )
            history.append(diagnostic)
            save_checkpoint(
                path,
                manager,
                optimizer,
                iteration + 1,
                {
                    "signature": signature,
                    "history": history,
                    "reader_state": reader.state_dict(),
                    "readiness": readiness_report,
                },
            )
            if (iteration + 1) % config["manager"]["eval_every"] == 0:
                tqdm.write(
                    str(
                        {
                            "stage": "manager",
                            "variant": variant,
                            "seed": seed,
                            "iteration": iteration + 1,
                            "return": diagnostic["training"]["return"]["mean"],
                        }
                    )
                )
            progress.set_postfix(
                return_mean=f"{diagnostic['training']['return']['mean']:.4f}",
                gradient=f"{diagnostic['gradient_norm']:.3g}",
                refresh=False,
            )
            progress.update(1)
    if reader_digest(reader) != fingerprint:
        raise AssertionError("Frozen reader changed during manager training")
    return manager, {
        "seed": seed,
        "variant": variant,
        "history": history,
        "reader_sha256": fingerprint,
        "training_trajectories": len(history) * config["manager"]["batch_size"],
        "environment_interactions": sum(
            row["compute"]["environment_steps"] for row in history
        ),
        "rollout_seconds": sum(row["compute"]["rollout_seconds"] for row in history),
        "update_seconds": sum(row["update_seconds"] for row in history),
        "checkpoint_bytes": path.stat().st_size,
    }


def restore_manager(
    path: Path, config: dict, reader: TaskReader, seed: int, variant: str
) -> BlendingManager:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "reader_sha256": reader_digest(reader),
        "config": config,
        "seed": seed,
        "variant": variant,
    }
    if saved["extra"]["signature"] != expected:
        raise ValueError("Manager checkpoint configuration or reader mismatch")
    if saved["iteration"] != config["manager"]["updates"]:
        raise ValueError("Manager training incomplete; resume before final evaluation")
    manager = BlendingManager(
        manager_dimension(reader), config["reader"]["slots"], variant
    ).to(next(reader.parameters()).device)
    manager.load_state_dict(saved["manager"])
    return manager.eval().requires_grad_(False)
