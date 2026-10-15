# Step 3 repair implementation — execution items 1–3

Implemented on 2026-10-15. Scope follows the user's stopping instruction:
complete execution items **1, 2, and 3** in [step_3_fix.md](step_3_fix.md),
then stop before full-budget reader preparation. No Step 4 work was started.
This is a new report. The previous report is preserved, unchanged, as
[step_3_implementation_legacy.md](step_3_implementation_legacy.md).
[step_3_results.md](step_3_results.md) remains historical and unchanged.

## Delivered scope

1. Snapshotted historical code/configurations/reports and recorded hashes of
   47 existing experiment artifacts/checkpoints. Pinned and independently
   preserved the upstream sources; emitted configuration, gradient, and
   write-event audits.
2. Removed the custom reconstruction head, supervision, weight defaults,
   preparation/evaluation code, and diagnostic export. Replaced active reader
   preparation with the task-specific upstream model and fixed-LRU section
   training. Old reader checkpoints are explicitly incompatible.
3. Passed deterministic parity tests for both complete architectures and three
   chronological sections, plus interruption/resume tests and Step 1/2 regression.

**Full preparation has not started:** neither a 57,600-update Autoencode reader
nor a 9,600-update RememberColor reader has been trained by this repair.
Three-update engineering fixtures do not establish stable learning,
reproduction performance, or reader readiness.

## Preservation and source reference

The reference is RATE commit
[`8abce2dce9ceacd411d46d4a99c287f7975892f5`](https://github.com/CognitiveAISystems/RATE/tree/8abce2dce9ceacd411d46d4a99c287f7975892f5),
verified from the existing clean `/tmp/elmur-rate` checkout. Current `main`
was not substituted. Byte-exact model/layer/encoder, trainer, scheduler,
dataloader, CLI, and command sources are in
`elmur_klpo/vendor/rate_reference/sources.zip`, with per-file SHA-256 hashes in
[its manifest](elmur_klpo/vendor/rate_reference/manifest.json). Tests extract
these sources to temporary storage and load independent classes with import
prefixes adjusted in memory. The original active vendored model/layers/encoders
were not modified. Attribution and the MIT license remain in `vendor/`.

[Historical snapshot manifest](experiments/step3/historical_pre_repair/manifest.json)
records preserved artifact paths, sizes, and hashes. Configurations and old
reports are also copied under that directory. The old Step 3 code and tests
are in ignored `runs/step3_pre_repair_snapshot/source.tar.gz`. Existing runs
and checkpoints remain at their historical paths; new artifacts use fresh roots.

## Configuration match and deliberate differences

The actual pinned RememberColor command is `run_experiments/MIKASA_Robo/ELMUR_1/rc3.sh`;
Autoencode uses the command in `offline_rl_baselines/ELMUR/README.md`.
The executable reference parser resolves command values and model defaults
from the archived sources, rather than duplicating a hand-written table.

Every model and training field is compared in:

- [Autoencode reference/local diff](experiments/step3/repair/autoencode_configuration_match.json).
- [RememberColor reference/local diff](experiments/step3/repair/remember_color_configuration_match.json).

| Setting | AutoencodeEasy | RememberColor3 |
| --- | ---: | ---: |
| Model width / layers / heads | 64 / 12 / 4 | 128 / 4 / 16 |
| Slots **per layer** | 8 | 256 |
| FFN / routed FFN / shared FFN | 128 / 128 / 256 | 128 / 128 / 128 |
| Context / sections / sample tokens | 35 / 3 / 105 | 20 / 3 / 60 |
| Initial memory standard deviation | 0 | 0.1 |
| Native new-candidate coefficient | 0.79 | 0.40683708068133906 |
| Norm / pre-layer norm | RMSNorm / false | LayerNorm (`None`) / false |
| Routed experts / configured top-k / shared experts | 1 / 1 / 2 | 16 / 2 / 1 |
| Model / attention / memory dropout | 0.14 / 0.26 / 0.17 | 0.1266135150715325 / 0.29604715763026573 / 0.05486068387428239 |
| Configured label smoothing | 0.22 | 0.20901542191561256 (unused by MSE) |
| MoE auxiliary coefficient | 0.1 | 0.1 |
| Output | Linear, four suit logits | Linear + Tanh, eight controls |

Both tasks use relative positions, a causal self-attention mask, observation-only
`sequence_format="s"`, separate layer-local memory, and detached section history.
Shared experts are enabled and `use_swiglu=False` is passed exactly as configured.
The pinned MoE's experts nevertheless instantiate SwiGLU internally. Its routed
count is `max(1, top_k - n_shared)`, which is one for both commands; shared
branches remain additional always-active branches. These semantics are preserved.

| Training setting | AutoencodeEasy | RememberColor3 |
| --- | ---: | ---: |
| Completed optimizer updates required per full run | **57,600** | **9,600** |
| Batch size (`drop_last=False`) | 128 | 64 |
| Reference epochs / reference dataset size | 800 / 3,000 | 200 / 1,000 |
| AdamW learning rate | 0.00012 | 0.00020459570733760105 |
| AdamW betas | (0.99, 0.99) | (0.99, 0.99) |
| Weight decay / gradient clip | 0.1 / 5 | 0.001 / 5 |
| Configured warmup | 50,000 | 30,000 |
| Cosine decay | false | true |
| Final tokens / LR end factor | 10,000,000 / 0.01 | 10,000,000 / 0.1 |
| Validation interval, optimizer updates | 600 | 300 |
| Last-checkpoint interval, optimizer updates | 300 | 300 |

Discrepancies and intentional choices are explicit:

- The pinned native coefficients are 0.79 and 0.40683708068133906, rather than
  rounded paper values 0.80 and 0.40. RememberColor's exact command LR/dropouts
  also supersede rounded examples in the README.
- The pinned `src/train.py` declares omitted optional read/write/bias flags as
  `None` and passes them through to the model. Literal CLI execution therefore
  can disable these paths. The repaired configs explicitly enable model defaults
  `use_tok2mem=True`, `use_mem2tok=True`, and `use_relative_bias=True`, and specify
  `use_shared_memory=False`. This is a disclosed repair of CLI defaults, not
  a claim of byte-for-byte reproduction of that CLI invocation.
- Autoencode's command placeholder `state_dim=-1` resolves to six in the upstream
  trainer. Local inputs use that resolved six-feature one-hot representation.
- Autoencode intentionally uses fresh native POPGym 1.0.7 episodes and oracle
  PLAY labels, rather than the published fixed PPO-GRU trajectory dataset.
  The full 52-card distribution replaces the historical short-length curriculum.
- RememberColor retains the audited 800/100/100 filename-disjoint split, RGB
  scaling by 255, and **raw unclipped** actions. Its upstream Tanh output remains
  bounded, even though some expert targets exceed those bounds. No target
  clipping, output-head replacement, or symlog transform compensates for this.
- Validation is fixed offline imitation on the validation split; native online
  inference is explicitly false. `ckpt_epoch` is retained as reference metadata;
  explicit update intervals govern local checkpoint/validation timing. The
  held-out test set does not select checkpoints. Published headline results
  are not a valid comparison for these data/evaluation choices.

## Section training, scheduler, and resume

`reader_data.py` materializes observation-only batches. `upstream_reader.py`
wraps the unmodified model, initializing one independent Gaussian bank and -1
position tensor per layer for every batch. Every sample has three chronological
sections; incoming vectors detach before each section. `reader_training.py`
performs **zero_grad, backward, global clipping, and optimizer.step per section**.
Action loss and the already-weighted upstream MoE auxiliary loss are summed
without a second auxiliary multiplier. Every layer's writer MoE is an optimizer
member. There is no cross-section memory BPTT or substitute writer supervision.

Autoencode has 103 actual decisions: 51 WATCH, then 52 PLAY. The last card is
visible at the first PLAY decision. Reverse-suit labels are tested against the
observed deck order. The two padding observations repeat the final observation,
matching the loader convention, with invalid targets. WATCH and padding receive
no suit CE; the WATCH-only first section still executes its auxiliary update.
RememberColor minimizes MSE across raw eight-dimensional expert controls. Phase
metrics use cue steps 0–4, delay steps 5–9, and control steps 10–59. No renderer
or inferred oracle actions are used for recorded RGB preparation.

Scheduler behavior is intentionally literal:

- Autoencode uses upstream `LambdaLR`: initialization sets LR to `base/50000`;
  it advances after each section update. The transition to linear decay is
  checked **at epoch end against completed minibatches**, while warmup itself
  counts section scheduler steps. At the nominal 800 epochs there are only
  19,200 minibatches, so the 50,000-minibatch transition is never reached;
  the section-step warmup still saturates. No unit reinterpretation is applied.
- RememberColor's first update uses the base LR. After each update, upstream
  counts `(targets >= 0).sum()` and uses this counter for warmup/cosine scheduling.
  For continuous targets that counts nonnegative **action elements**, not timesteps
  or optimizer updates. The original cosine formula, including its unclamped
  progress and LR floor, is retained.

`completed_optimizer_steps` increments immediately after successful
`optimizer.step()`. That counter, not epoch count, stops preparation. With 800
RememberColor training files, 9,600 updates represent about **246.154 effective
passes**, rather than 200. Its 3,200 minibatches present 196,928 training
trajectories, repeatedly exposing the 800 unique files. Autoencode uses nominal
3,000-episode epochs with fresh seed blocks, and generates 2,400,000 unique
training episodes across its full budget. These are **planned counts**, not
completed measurements.

`reader_last.pt` and `reader_best.pt` save schema, full config/source/data
signatures, weights, optimizer, scheduler, RNG, counters, epoch permutation,
pending batch IDs, next section, and detached per-layer memories. A pending batch
is reconstructed from immutable paths/native seeds on resume. Validation uses a
fixed RNG scope and restores training RNG. Baseline validation at zero steps,
periodic validation, and final validation use the same protocol; best/last
metrics are reported separately. Legacy `reader.pt` files are rejected.

## Write-event and gradient audits

Machine-readable contracts:
[Autoencode](experiments/step3/repair/autoencode_write_event_contract.json),
[RememberColor](experiments/step3/repair/remember_color_write_event_contract.json).

Each layer produces one bank of slot-conditioned candidates `[B, slots, width]`
per forward section, from `memory_ffn`, followed by memory dropout in training.
Layers execute in index order. `use_tok2mem` enables **reading**, where tokens
query incoming memory; `use_mem2tok` enables **writing**, where memory queries all
section tokens. The names are preserved. Outgoing memory does not affect any
prediction in the current section; its earliest effect is in the next section.

Relative token positions remain newest-first while observations remain
chronological, exactly as in the source. Written timestamps are `tok_pos[0]`,
so the first three writes are 34/69/104 or 19/39/59. Event index is
`section*n_layer + layer`; all layers of one section share its physical time
interval. After three sections, only three slots per layer have been selected;
resetting memory per batch means training does not fill either complete bank.

The native code differs from its own empty-slot comment: `lru_one_hot` receives
weight lambda even on the first empty slot. It blends with the initialized
vector there. The `cumsum==1` empty mask can additionally select intervening
occupied slots in artificial noncontiguous-hole states. Both behaviors are
covered by deterministic raw-source parity tests and preserved for this fixed-LRU
reference. The specification's future learned-manager empty-slot hard replacement
will be a separate, documented integration rule; it is not implemented here.

The gradient audit independently differentiates task loss against each layer's
writer FFN parameters: **none receive a task-loss gradient** with detached
section history. Upstream auxiliary gradients remain present. Actual normalized
top-one routing makes them zero or very small in these fixtures; this is not
replaced by reconstruction or future-task BPTT. Measured audit files:
[Autoencode](experiments/step3/repair/autoencode_gradient_audit.json),
[RememberColor](experiments/step3/repair/remember_color_gradient_audit.json).

The existing manager grid remains exactly `(0, 0.25, 0.5, 0.8, 0.95)`, slot-major
IDs `5*slot+i`, followed by global no-op ID `5*slots`. Tests assert the complete
mapping for both eight and 256 slots. The single-token `TaskReader` and manager
utilities remain only for historical diagnostics/regression tests. Active
preparation uses `FixedLRUReader`; manager CLI stages are blocked until the
reproduction/readiness gates and event-level implementation exist.

## Executed verification and resources

Full commands were run from the pinned Python 3.12 environment:

```bash
export UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache
uv sync --offline
uv run pytest -q --basetemp=runs/step3_repair_pytest
uv run python scripts/verify_gradients.py
uv run ruff check .
uv run ruff format --check .
uv build --offline
uv run python -m elmur_klpo.experiment --config configs/step3_smoke.json --output runs/step3_repair_final_smoke
uv run python -m elmur_klpo.experiment --config configs/step3_smoke.json --output runs/step3_repair_final_smoke --resume
uv run python -m elmur_klpo.step3.aggregation --inputs runs/step3_repair_final_smoke --output experiments/step3/repair/smoke.json
```

The suite passed **133 tests**, with two expected skips: CUDA and native robot
rendering. The raw source emits one existing docstring escape SyntaxWarning.
Step 1 checks A–G and all-parameter finite differences pass; maximum finite-
difference absolute error is `3.38644e-11`. Ruff and package build pass. Wheel
inspection confirms inclusion of the pinned source archive and manifest.

The independent parity minibatches use **batch two**, all 12 or four layers,
and complete 35×3 or 20×3 sections. Resume fixtures use batch one. Full production
batch sizes 128/64 and CUDA capacity have not been verified. Autoencode uses
native full-length generated episodes. RememberColor smoke uses three explicitly
synthetic RGB/action files, with one file in each split; these are not expert data
or new dataset integrity evidence. No released dataset audit was reopened.

| Final smoke measurement | Autoencode | RememberColor synthetic fixture |
| --- | ---: | ---: |
| Maximum local/reference gradient absolute difference | 0 | 0 |
| Initial memory empirical std | 0 | 0.0999799 |
| Optimizer updates in each compared trajectory of training | 3 | 3 |
| Local + raw reference parity optimizer events | 6 | 6 |
| Uninterrupted + interrupted/resumed optimizer events | 6 | 6 |
| Model forward invocations, parity plus resume verification | 24 | 24 |
| Training sample presentations, two resume-verification runs | 2 | 2 |
| Valid training transitions, per resume-verification run | 103 | 60 |
| Validation forward invocations, per run | 6 | 6 |
| Total smoke wall time including checkpoint I/O | 10.84 s | 13.75 s |
| Validation wall time, two runs | 0.410 s | 0.538 s |
| GPU time / GPU peak allocation | 0 / 0 | 0 / 0 |

Each interruption occurs after optimizer event one, with next section one and
pending per-layer memories saved. Final weights, training histories, and
validation histories match uninterrupted training **bit for bit**. Completed
CLI resume performs no extra optimizer updates. Best and last happen to be
step three in these fixtures; this is not evidence of meaningful learning.

Full histories/checkpoints are under ignored `runs/step3_repair_final_smoke/`;
[compact verification measurements](experiments/step3/repair/smoke.json) are
checked in. Gradient norms and fixture losses are diagnostics only. The
Autoencode writer auxiliary gradient norm is zero in all parity sections;
RememberColor's is approximately `1.8e-8` to `4.7e-8`.

A repeat suite initially exhausted `/tmp` while retaining duplicate scratch
checkpoints. Only this repair's duplicate smoke/test outputs were removed,
and final pytest temporary files were placed under ignored `runs/` on the
workspace filesystem. The final suite passed. The first online `uv run` also
failed resolving build dependencies through DNS; offline sync/build succeeded.
These are engineering retries, not failed full preparation or simulator runs.
No new native robot integration was attempted; its historical renderer failure
is unchanged. Resource counters distinguish optimizer updates, episode/sample
presentations, valid transitions, padded tokens, forward invocations, validation,
GPU step timing/peak allocation, and actual native materialization interactions,
including deterministic regeneration on resume.

## Manual preparation commands and remaining work

Run full-budget preparation manually with **fresh output directories**:

```bash
export UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache
uv run python -m elmur_klpo.experiment --config configs/step3_reader_preparation.json --output runs/step3_upstream_readers
uv run python -m elmur_klpo.experiment --config configs/step3_reader_preparation.json --output runs/step3_upstream_readers --resume
```

To start one task independently:

```bash
uv run python -m elmur_klpo.step3.experiment --config configs/step3_reader_preparation.json --output runs/step3_upstream_readers --task autoencode
uv run python -m elmur_klpo.experiment --config configs/step3_remembercolor_reader.json --output runs/step3_upstream_rc3
```

Use `--device cuda` with a compatible PyTorch CUDA environment. The previously
provisioned `runs/step3_gpu` environment can be selected with
`UV_PROJECT_ENVIRONMENT=runs/step3_gpu uv run --no-sync ... --device cuda`.
This repair's CUDA behavior and full-batch memory requirements remain untested.
Reader preparation does not automatically start manager training or evaluate the
separate native-return readiness gate.

Outstanding execution items are unchanged:

- **4:** run full budgets, verify saved counters, and retain best/last checkpoints.
- **5:** evaluate fixed-LRU learning, genuine/zero/shuffled memory diagnostics,
  independent reader seeds, and the separate bootstrap readiness gate. The
  validation utility supports memory interventions, but these have not been
  executed as reproduction/readiness evidence here. Shuffled memory uses another
  held-out episode's layer banks, rather than an attention-invariant joint slot
  permutation. Closed-loop robotics success requires a functioning renderer.
- **6:** implement layer/event manager actions, legal masks in all probabilities
  and KL/score-centering paths, hard empty replacement, imitation reward timing,
  frozen reader checks, and matched separately trained systems/ablations. Native
  rewards must not substitute for manager imitation rewards. Existing scalar
  manager utilities do not implement this event interface.
- **7:** write new full results separating fidelity, memory dependence, and manager
  effects after those experiments. No existing results report was overwritten.

Both reproduction learning and reader readiness remain **unestablished**. Their
requirements were not relaxed, and no learned-manager comparison was run.
