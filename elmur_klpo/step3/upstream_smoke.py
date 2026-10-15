"""Deterministic pinned-source parity and checkpoint resume; no preparation run."""

import ast
import copy
import importlib
import json
import math
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from elmur_klpo.step3.artifacts import write_json
from elmur_klpo.step3.reader_data import ReaderData
from elmur_klpo.step3.reader_training import action_loss, make_optimizer
from elmur_klpo.step3.reference import (
    configuration_diff,
    read_reference,
    reference_package,
    verify_sources,
    write_event_contract,
)
from elmur_klpo.step3.upstream_reader import FixedLRUReader, detach_memories
from elmur_klpo.trainer import seed_everything


def reference_model_class() -> type:
    """Load byte-exact raw model with independent raw layers and encoder modules."""
    source = read_reference("offline_rl_baselines/ELMUR/model.py")
    package = reference_package()
    source = source.replace(
        "from offline_rl_baselines.ELMUR.layers",
        f"from {package}.offline_rl_baselines.ELMUR.layers",
    )
    source = source.replace(
        "from RATE.env_encoders",
        f"from {package}.RATE.env_encoders",
    )
    module = types.ModuleType("pinned_rate_elmur_reference")
    exec(compile(source, "pinned_RATE/model.py", "exec"), module.__dict__)
    return module.ELMURModel


def reference_loss_function() -> object:
    """Execute actual trainer loss code without simulator/WandB dependencies."""
    tree = ast.parse(read_reference("src/trainer.py"))
    trainer = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer"
    )
    loss = next(
        n
        for n in trainer.body
        if isinstance(n, ast.FunctionDef) and n.name == "calculate_losses"
    )
    module = ast.Module(body=[loss], type_ignores=[])
    namespace = {"torch": torch, "nn": torch.nn, "F": F}
    exec(compile(module, "pinned_RATE/calculate_losses", "exec"), namespace)
    return namespace["calculate_losses"]


def synthetic_robot_config(config: dict, root: Path) -> dict:
    """Create tiny synthetic 60-token RGB fixtures, explicitly not expert data."""
    config = copy.deepcopy(config)
    data = root / "synthetic_rgb_fixture"
    data.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        path = data / f"train_data_{index}.npz"
        if not path.exists():
            rng = np.random.default_rng(800 + index)
            np.savez_compressed(
                path,
                rgb=rng.integers(0, 256, (60, 128, 128, 6), dtype=np.uint8),
                action=np.tile(np.linspace(-1.5, 1.6, 8, dtype=np.float32), (60, 1)),
                done=np.array([0] * 59 + [1]),
                success=np.zeros(60),
            )
    manifest = root / "synthetic_rgb_manifest.json"
    write_json(
        manifest,
        {
            "data_directory": str(data.resolve()),
            "task": "RememberColor3-v0",
            "sha256": "synthetic-engineering-fixture-not-published-data",
            "source": "synthetic smoke fixture only",
        },
    )
    config["preparation"]["demonstrations"] = {
        "directory": str(data.resolve()),
        "manifest": str(manifest.resolve()),
        "train_episodes": 1,
        "validation_episodes": 1,
        "test_episodes": 1,
        "action_target": "raw",
    }
    return config


def parity_minibatch(config: dict, device: str = "cpu", batch_size: int = 2) -> dict:
    """Compare full architecture, three updates, all outputs/memories/gradients."""
    data = ReaderData(config)
    batch = data.load([i % data.train_size for i in range(batch_size)], "train")
    settings = {
        **config["preparation"],
        "batches_per_epoch": math.ceil(
            data.train_size / config["preparation"]["batch_size"]
        ),
    }
    seed_everything(42)
    local = FixedLRUReader(config["task"], config["reader"]).to(device)
    seed_everything(42)
    reference = reference_model_class()(**config["reader"]).to(device)
    for key, value in local.model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], atol=0, rtol=0)
    optimizer, scheduler = make_optimizer(local, settings)
    ref_optimizer = torch.optim.AdamW(
        reference.parameters(),
        lr=settings["learning_rate"],
        betas=(settings["beta_1"], settings["beta_2"]),
        weight_decay=settings["weight_decay"],
    )
    raw_scheduler = importlib.import_module(
        f"{reference_package()}.src.utils.lr_scheduler"
    ).LearningRateScheduler
    ref_schedule = raw_scheduler(
        {"training": settings}, range(settings["batches_per_epoch"])
    )
    ref_warmup = (
        None
        if settings["use_cosine_decay"]
        else ref_schedule.make_warmup_scheduler(ref_optimizer)
    )
    seed_everything(43)
    memory = local.initial(batch_size)
    seed_everything(43)
    ref_memory = reference.init_memory(batch_size, torch.device(device))
    assert len({m.vec.data_ptr() for m in memory}) == len(memory)
    initial_std = torch.cat([m.vec.flatten() for m in memory]).std().item()
    rows, tokens = [], 0
    task_losses = reference_loss_function()
    loss_self = types.SimpleNamespace(
        env_name=config["reader"]["env_name"], config={"model": config["reader"]}
    )
    max_gradient_error = 0.0
    for section in range(3):
        width = settings["context_length"]
        chunk = batch.section(section * width, width)
        memory, ref_memory = detach_memories(memory), detach_memories(ref_memory)
        assert all(not m.vec.requires_grad and m.vec.grad_fn is None for m in memory)
        for a, b in zip(memory, ref_memory, strict=True):
            torch.testing.assert_close(a.vec, b.vec, atol=0, rtol=0)
            torch.testing.assert_close(a.pos, b.pos, atol=0, rtol=0)
        seed_everything(100 + section)
        output = local(chunk.observations, memory, section * width)
        states = (
            chunk.observations.to(device=device, dtype=torch.float32)
            / local.observation_scale
        )
        seed_everything(100 + section)
        ref_output = reference(
            states,
            None,
            states.new_zeros(batch_size, width, 1),
            None,
            None,
            memory_states=ref_memory,
            pos_offset=section * width,
        )
        torch.testing.assert_close(
            output["logits"], ref_output["logits"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            output["aux_loss"], ref_output["aux_loss"], atol=0, rtol=0
        )
        mask = (chunk.valid & chunk.action_valid).to(device)
        targets = chunk.targets.to(device)
        prediction = action_loss(
            output["logits"],
            targets,
            mask,
            local.task_name,
            local.model_config_smoothing,
        )
        if mask.any():
            # Filter to the specified valid labels before the untouched upstream loss.
            ref_loss, _ = task_losses(
                loss_self, ref_output["logits"][mask], targets[mask], None, 0
            )
        else:
            ref_loss = ref_output["logits"].sum() * 0
        loss = prediction + output["aux_loss"]
        ref_loss = ref_loss + ref_output["aux_loss"]
        torch.testing.assert_close(loss, ref_loss, atol=0, rtol=0)
        writer = [
            p for layer in local.model.layers for p in layer.memory_ffn.parameters()
        ]
        task_gradients = torch.autograd.grad(
            prediction, writer, allow_unused=True, retain_graph=True
        )
        assert all(g is None for g in task_gradients)
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        ref_loss.backward()
        for (name, a), (_, b) in zip(
            local.model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert (a.grad is None) == (b.grad is None), name
            if a.grad is not None:
                assert torch.isfinite(a.grad).all(), name
                max_gradient_error = max(
                    max_gradient_error, float((a.grad - b.grad).abs().max())
                )
                torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
        writer_norm = (
            sum(float(p.grad.square().sum()) for p in writer if p.grad is not None)
            ** 0.5
        )
        writer_names = [
            name
            for name, p in local.model.named_parameters()
            if ".memory_ffn." in name and p.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(local.parameters(), 5, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(
            reference.parameters(), 5, error_if_nonfinite=True
        )
        optimizer.step()
        ref_optimizer.step()
        scheduler.after_step(targets)
        tokens += int((targets >= 0).sum())
        if ref_warmup is not None:
            ref_warmup.step()
        else:
            ref_schedule.update_learning_rate(ref_optimizer, tokens)
        assert optimizer.param_groups[0]["lr"] == ref_optimizer.param_groups[0]["lr"]
        for name, p in local.model.state_dict().items():
            torch.testing.assert_close(p, reference.state_dict()[name], atol=0, rtol=0)
        memory, ref_memory = output["memory_states"], ref_output["memory_states"]
        for a, b in zip(memory, ref_memory, strict=True):
            torch.testing.assert_close(a.vec, b.vec, atol=0, rtol=0)
            torch.testing.assert_close(a.pos, b.pos, atol=0, rtol=0)
            assert a.vec.shape == (
                batch_size,
                config["reader"]["memory_size"],
                config["reader"]["d_model"],
            )
        rows.append(
            {
                "section": section,
                "logits_shape": list(output["logits"].shape),
                "memory_shapes": [list(m.vec.shape) for m in memory],
                "timestamps_first_slots": memory[0].pos[0, :3].tolist(),
                "valid_targets": int(mask.sum()),
                "action_loss": float(prediction.detach()),
                "moe_aux_loss": float(output["aux_loss"].detach()),
                "writer_aux_gradient_norm": writer_norm,
                "writer_task_gradient_parameters": 0,
                "writer_parameters_with_aux_grad": writer_names,
                "next_lr": optimizer.param_groups[0]["lr"],
            }
        )
    return {
        "passed": True,
        "source_integrity": verify_sources(),
        "task": local.task_name,
        "batch_size": batch_size,
        "completed_optimizer_steps_per_implementation": 3,
        "initial_memory_std": initial_std,
        "max_gradient_absolute_error": max_gradient_error,
        "sections": rows,
        "gradient_audit": "No prediction gradients reach any layer's memory_ffn under detached section history. Writer MoE auxiliary gradients retained exactly; normalized top-1 routing can yield zero/nearly-zero gradient. No added supervision or BPTT.",
        "device": device,
    }


def smoke_task(
    config: dict, output: Path, resume: bool = False, device: str = "cpu"
) -> dict:
    """Run engineering verification only, including interruption after section one."""
    from elmur_klpo.step3 import preparation

    if (
        not config.get("verification_only")
        or config["preparation"]["optimizer_updates"] != 3
        or config["preparation"]["batch_size"] != 1
    ):
        raise ValueError(
            "Use the dedicated three-section, batch-one smoke configuration"
        )
    smoke_start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if config["task"]["name"] == "remember_color":
        config = synthetic_robot_config(config, output)
    write_json(output / "configuration_match.json", configuration_diff(config))
    write_json(
        output / "write_event_contract.json",
        write_event_contract(config["reader"], config["preparation"]["context_length"]),
    )
    if resume and (output / "smoke_results.json").exists():
        result = json.loads((output / "smoke_results.json").read_text())
        preparation.prepare_reader(config, output / "resumed", True, device)
        return result
    parity = parity_minibatch(config, device)
    expected, _, full = preparation.prepare_reader(
        config, output / "uninterrupted", False, device
    )
    try:
        preparation.prepare_reader(
            config, output / "resumed", False, device, verification_interrupt_after=1
        )
    except preparation.VerificationInterruption:
        pass
    else:
        raise AssertionError("Expected smoke interruption after section one")
    checkpoint = torch.load(
        output / "resumed/reader_last.pt", weights_only=False, map_location="cpu"
    )
    assert checkpoint["state"]["completed_optimizer_steps"] == 1
    assert checkpoint["state"]["next_section"] == 1
    actual, _, restored = preparation.prepare_reader(
        config, output / "resumed", True, device
    )
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], atol=0, rtol=0)
    assert full["history"] == restored["history"]
    assert full["validation_history"] == restored["validation_history"]
    assert restored["completed_optimizer_steps"] == 3
    assert not restored["readiness"]["passed"]
    result = {
        "status": "upstream_parity_smoke_passed",
        "parity": parity,
        "checkpoint_resume": {
            "passed": True,
            "interrupted_after_optimizer_steps": 1,
            "resumed_at_section": 1,
            "final_optimizer_steps": 3,
            "parameters_bitwise_equal": True,
            "history_equal": True,
            "validation_equal": True,
        },
        "scope": "Full architecture and complete three-section minibatch; parity batch two, resume batch one; synthetic RememberColor RGB/actions, native full-length Autoencode episodes. No full-budget reader preparation, reproduction learning/readiness gate, or manager experiment.",
        "full_budget_reader_preparation_started": False,
        "compute": {
            "total_wall_seconds_including_checkpoint_io": time.perf_counter()
            - smoke_start,
            "model_forward_passes": 6
            + full["training_forward_passes"]
            + full["validation_forward_passes"]
            + restored["training_forward_passes"]
            + restored["validation_forward_passes"],
            "optimizer_steps_local_and_reference_parity": 6,
            "optimizer_steps_resume_verification": 6,
            "gpu_step_seconds": full["gpu_step_seconds"] + restored["gpu_step_seconds"],
            "peak_gpu_bytes": max(full["peak_gpu_bytes"], restored["peak_gpu_bytes"]),
            "validation_seconds": full["validation_seconds"]
            + restored["validation_seconds"],
            "unique_training_episodes_in_preparation_fixture": full[
                "unique_trajectories_seen"
            ],
            "training_sample_presentations_in_preparation_fixtures": full[
                "sample_presentations"
            ]
            + restored["sample_presentations"],
        },
    }
    write_json(output / "smoke_results.json", result)
    return result
