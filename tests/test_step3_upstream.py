"""Pinned-source reader parity, section updates, masking, and native write audit."""

import copy
import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from elmur_klpo.memory import MERGE_COEFFICIENTS, ActionSpace
from elmur_klpo.step3.preparation import SCHEMA, prepare_reader, validate_config
from elmur_klpo.step3.reader_data import autoencode_batch
from elmur_klpo.step3.reader_training import ReaderScheduler, action_loss
from elmur_klpo.step3.reference import (
    BUDGETS,
    COMMIT,
    reference_settings,
    verify_sources,
    write_event_contract,
)
from elmur_klpo.step3.upstream_reader import FixedLRUReader
from elmur_klpo.step3.upstream_smoke import reference_model_class, smoke_task


def config_for(name):
    return next(
        entry
        for entry in json.loads(Path("configs/step3_smoke.json").read_text())["tasks"]
        if entry["task"]["name"] == name
    )


@pytest.fixture(scope="module", params=["autoencode", "remember_color"])
def verified_smoke(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"upstream-{request.param}")
    config = config_for(request.param)
    return smoke_task(config, root), root, config


def test_complete_three_section_parity_and_resume(verified_smoke):
    result, root, config = verified_smoke
    parity = result["parity"]
    assert parity["passed"] and parity["max_gradient_absolute_error"] == 0
    assert parity["completed_optimizer_steps_per_implementation"] == 3
    assert result["checkpoint_resume"]["parameters_bitwise_equal"]
    assert not result["full_budget_reader_preparation_started"]
    expected_std = config["reader"]["memory_init_std"]
    assert parity["initial_memory_std"] == pytest.approx(expected_std, abs=0.004)
    assert [s["valid_targets"] for s in parity["sections"]] == (
        [0, 38, 66] if config["task"]["name"] == "autoencode" else [40, 40, 40]
    )
    for section, row in enumerate(parity["sections"]):
        assert row["logits_shape"] == [
            2,
            config["preparation"]["context_length"],
            config["reader"]["act_dim"],
        ]
        assert len(row["memory_shapes"]) == config["reader"]["n_layer"]
        assert row["moe_aux_loss"] > 0
        assert row["writer_task_gradient_parameters"] == 0
        assert row["timestamps_first_slots"][: section + 1] == [
            (i + 1) * config["preparation"]["context_length"] - 1
            for i in range(section + 1)
        ]
    saved = torch.load(
        root / "resumed/reader_last.pt", map_location="cpu", weights_only=False
    )
    assert saved["state"]["completed_optimizer_steps"] == 3
    assert saved["state"]["pending"] is None
    assert saved["scheduler"]["minibatches"] == 1
    assert saved["schema"] == SCHEMA
    assert not any("reconstruct" in name for name in saved["reader"])
    assert len(saved["state"]["validation_history"]) == 2
    assert (root / "resumed/reader_best.pt").exists()
    actual_config = saved["signature"]["config"]
    _, manager, report = prepare_reader(actual_config, root / "resumed", resume=True)
    assert manager is None and not report["readiness"]["passed"]
    assert report["test_split_used"] is False
    altered = copy.deepcopy(actual_config)
    altered["reader_seed"] += 1
    with pytest.raises(ValueError, match="configuration or dataset mismatch"):
        prepare_reader(altered, root / "resumed", resume=True)


def test_pinned_sources_and_full_configs():
    assert verify_sources()["commit"] == COMMIT
    config = json.loads(Path("configs/step3_reader_preparation.json").read_text())
    for entry in config["tasks"]:
        validate_config(entry)
        name = entry["task"]["name"]
        ref = reference_settings(name)
        assert entry["reader"] == ref["model"]
        assert entry["preparation"]["optimizer_updates"] == BUDGETS[name]
        assert entry["reader"]["use_tok2mem"] is True
        assert entry["reader"]["use_mem2tok"] is True
        nominal_n = 3000 if name == "autoencode" else 1000
        assert (
            BUDGETS[name]
            == ref["training"]["epochs"]
            * math.ceil(nominal_n / ref["training"]["batch_size"])
            * 3
        )
        broken = copy.deepcopy(entry)
        broken["preparation"]["optimizer_updates"] = 500
        with pytest.raises(ValueError, match="exact upstream optimizer update budget"):
            validate_config(broken)


def test_canonical_manager_grid_is_unchanged():
    assert MERGE_COEFFICIENTS == (0, 0.25, 0.5, 0.8, 0.95)
    for slots in (8, 256):
        space = ActionSpace(slots)
        assert len(space) == 5 * slots + 1
        for slot in range(slots):
            for i, alpha in enumerate(MERGE_COEFFICIENTS):
                assert space.decode(5 * slot + i) == (slot, alpha)
                assert space.encode(slot, alpha) == 5 * slot + i
        assert space.decode(5 * slots) == (None, 1)


def test_play_padding_and_mse_loss_masks():
    logits = torch.randn(2, 5, 4, requires_grad=True)
    target = torch.tensor([[0, 1, 2, -10, -10], [1, 2, 3, 0, -10]])
    mask = torch.tensor(
        [[False, True, True, False, False], [False, False, True, True, False]]
    )
    loss = action_loss(logits, target, mask, "autoencode", 0.22)
    torch.testing.assert_close(
        loss, F.cross_entropy(logits[mask], target[mask], label_smoothing=0.22)
    )
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    empty = action_loss(logits, target, torch.zeros_like(mask), "autoencode", 0.22)
    assert empty.item() == 0 and torch.isfinite(empty)
    continuous = torch.zeros(2, 5, 8, requires_grad=True)
    raw = torch.full_like(continuous, 1.6)
    mse = action_loss(continuous, raw, mask, "remember_color", 0.20901542191561256)
    assert mse.item() == pytest.approx(1.6**2)
    mse.backward()
    assert torch.count_nonzero(continuous.grad[~mask]) == 0


def test_native_autoencode_reverse_boundary_and_no_wait():
    batch = autoencode_batch([13, 14])
    assert batch.observations.shape == (2, 105, 6)
    assert batch.valid.sum(1).tolist() == [103, 103]
    assert batch.action_valid.sum(1).tolist() == [52, 52]
    assert not batch.action_valid[:, :51].any()
    assert batch.action_valid[:, 51:103].all()
    for episode in range(2):
        watched_suits = batch.observations[episode, :52, 2:].argmax(-1)
        torch.testing.assert_close(
            batch.targets[episode, 51:103], watched_suits.flip(0)
        )
        assert (
            batch.targets[episode, 51] == batch.observations[episode, 51, 2:].argmax()
        )
    assert batch.targets[:, :103].min() >= 0
    assert batch.targets[:, :103].max() < 4
    assert (batch.targets[:, 103:] == -10).all()


def test_no_action_return_or_future_observation_leakage():
    config = config_for("autoencode")
    reader = FixedLRUReader(config["task"], config["reader"]).eval()
    batch = autoencode_batch([13])
    observations = batch.observations[:, :35]
    memory = reader.initial(1)
    output = reader(observations, memory)
    changed = observations.clone()
    changed[:, 18:] = torch.randn_like(changed[:, 18:])
    altered = reader(changed, memory)
    torch.testing.assert_close(
        output["logits"][:, :18], altered["logits"][:, :18], atol=0, rtol=0
    )
    # Even the raw pinned API ignores arbitrary expert-action/RTG values under 's'.
    raw = reader.model(
        observations,
        torch.full((1, 35, 1), 999),
        torch.full((1, 35, 1), 1234.0),
        batch.targets[:, :35],
        None,
        memory_states=memory,
    )
    torch.testing.assert_close(raw["logits"], output["logits"], atol=0, rtol=0)
    for a, b in zip(raw["memory_states"], output["memory_states"], strict=True):
        torch.testing.assert_close(a.vec, b.vec, atol=0, rtol=0)
    assert reader.model.norm_type == "rmsnorm"


@pytest.mark.parametrize("name", ["autoencode", "remember_color"])
@pytest.mark.parametrize("bank", ["empty", "full", "noncontiguous"])
def test_native_empty_full_write_semantics_match_raw_source(name, bank):
    config = config_for(name)
    reader = FixedLRUReader(config["task"], config["reader"]).eval()
    reference = reference_model_class()(**config["reader"]).eval()
    reference.load_state_dict(reader.model.state_dict())
    width, slots = config["reader"]["d_model"], config["reader"]["memory_size"]
    memory = reader.initial(1)
    # Use nonzero empty vectors to expose the pinned empty-slot blend quirk.
    for m in memory:
        m.vec.fill_(0.3)
        if bank == "full":
            m.pos.copy_(torch.arange(slots)[None])
            m.pos[0, 2] = -0  # tie with slot 0, native argmin must choose slot 0
        elif bank == "noncontiguous":
            m.pos.copy_(torch.arange(slots)[None])
            m.pos[0, 1] = -1
            m.pos[0, 3] = -1
    obs = (
        torch.zeros(1, 3, 6) if name == "autoencode" else torch.zeros(1, 3, 6, 128, 128)
    )
    captured = []
    hook = reader.model.layers[0].memory_ffn.register_forward_hook(
        lambda module, args, out: captured.append(out[0].detach())
    )
    with torch.no_grad():
        result = reader(obs, memory, 7)
        raw = reference(
            obs / reader.observation_scale,
            None,
            torch.zeros(1, 3, 1),
            None,
            None,
            memory_states=memory,
            pos_offset=7,
        )
    hook.remove()
    for a, b in zip(result["memory_states"], raw["memory_states"], strict=True):
        torch.testing.assert_close(a.vec, b.vec, atol=0, rtol=0)
        torch.testing.assert_close(a.pos, b.pos, atol=0, rtol=0)
    assert captured[0].shape == (1, slots, width)
    incoming, outgoing = memory[0], result["memory_states"][0]
    target = 1 if bank == "noncontiguous" else 0
    lam = config["reader"]["lru_blend_alpha"]
    torch.testing.assert_close(
        outgoing.vec[:, target],
        lam * captured[0][:, target] + (1 - lam) * incoming.vec[:, target],
    )
    assert outgoing.pos[0, target] == 9
    if bank != "noncontiguous":
        torch.testing.assert_close(
            outgoing.vec[:, 1:], incoming.vec[:, 1:], atol=0, rtol=0
        )
    else:
        # Explicitly preserve the upstream cumsum mask behavior on unusual holes.
        assert outgoing.pos[0, 2] == 9
        assert outgoing.pos[0, 3] == -1
    contract = write_event_contract(
        config["reader"], config["preparation"]["context_length"]
    )
    assert contract["events_per_forward"] == config["reader"]["n_layer"]
    assert contract["candidate_shape"] == ["batch", slots, width]


def test_upstream_scheduler_units_and_epoch_switch_resume():
    parameter = torch.nn.Parameter(torch.ones(()))
    settings = {**reference_settings("autoencode")["training"], "warmup_steps": 2}
    optimizer = torch.optim.AdamW([parameter], lr=settings["learning_rate"])
    schedule = ReaderScheduler(optimizer, settings, 1)
    assert optimizer.param_groups[0]["lr"] == settings["learning_rate"] / 2
    for _ in range(3):
        optimizer.step()
        schedule.after_step(torch.tensor([0, 1, -10]))
    schedule.end_minibatch(epoch_end=True)
    assert (
        schedule.tokens == 6 and not schedule.decay
    )  # three section steps != two batches
    schedule.end_minibatch(epoch_end=True)
    assert schedule.decay
    state = schedule.state_dict()
    next_parameter = torch.nn.Parameter(torch.ones(()))
    next_optimizer = torch.optim.AdamW([next_parameter], lr=settings["learning_rate"])
    restored = ReaderScheduler(next_optimizer, settings, 1)
    next_optimizer.load_state_dict(optimizer.state_dict())
    restored.load_state_dict(state)
    optimizer.step()
    schedule.after_step(torch.tensor([1]))
    next_optimizer.step()
    restored.after_step(torch.tensor([1]))
    assert optimizer.param_groups[0]["lr"] == next_optimizer.param_groups[0]["lr"]
    assert schedule.state_dict() == restored.state_dict()
    cosine = {**reference_settings("remember_color")["training"]}
    cosine_optimizer = torch.optim.AdamW([parameter], lr=cosine["learning_rate"])
    cos_schedule = ReaderScheduler(cosine_optimizer, cosine, 1)
    assert (
        cosine_optimizer.param_groups[0]["lr"] == cosine["learning_rate"]
    )  # full LR for first update
    cosine_optimizer.step()
    cos_schedule.after_step(torch.tensor([-1.6, 0.0, 1.2]))
    assert (
        cos_schedule.tokens == 2
    )  # nonnegative action elements, not timesteps or updates
    assert cosine_optimizer.param_groups[0]["lr"] == cosine["learning_rate"] * 2 / 30000


def test_legacy_checkpoint_is_explicitly_incompatible(tmp_path):
    config = config_for("autoencode")
    torch.save(
        {"extra": {"reader_state": {"reconstruct.weight": torch.ones(1)}}},
        tmp_path / "reader.pt",
    )
    with pytest.raises(ValueError, match="Legacy reader checkpoint incompatible"):
        prepare_reader(config, tmp_path, resume=True)
    with pytest.raises(FileExistsError, match="fresh output directory"):
        prepare_reader(config, tmp_path)
