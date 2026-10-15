# Step 3 Repair — Upstream-Matched ELMUR Reader and Learned Memory Manager

## Objective

Repair Step 3 so the reader/writer and its imitation-learning procedure closely reproduce upstream ELMUR, while preserving ELMUR-KLPO's *intentional* change: learning the memory-management decision (slot, retention coefficient, and optional no-op). Establish a functioning upstream LRU baseline before running the learned-manager experiment. This document is an implementation order for Codex, not a research proposal.

## Scope and exclusions

1. **Remove the custom candidate/observation reconstruction head and its loss** everywhere in reader preparation and evaluation. Do **not** remove upstream ELMUR's MoE auxiliary/load-balancing loss. No added reconstruction supervision or invented future-task BPTT.
2. **Match upstream's task-specific ELMUR model, input formatting, initialization, and training** (details below), not just slot count/layer count/window. Use upstream's implementation as the behavior reference; do not silently rewrite tensor orientation or reverse names based on intuition.
3. **Keep offline expert imitation.** Autoencode may generate fresh native episodes and use exact correct reverse-suit oracle labels; this is intentionally not byte-for-byte reproduction of upstream's fixed PPO-GRU dataset. RememberColor uses the audited offline expert dataset. Dataset provenance/split auditing is **complete**; do not reopen that task unless a new integrity regression is observed.
4. **No symlog work in this change.** Do not add any symlog transforms to features, actions, losses, or rewards. Do not reintroduce action clipping or change target values as a proxy for normalization. Preserve verified task-specific input preprocessing, native rewards, and output action bounds as applicable.
5. **Training budgets are exactly 57,600 optimizer/backprop steps for AutoencodeEasy and 9,600 for RememberColor3**, for the reader preparation/reproduction runs. Do not replace these with 500/1,500-update diagnostics or treat epoch counts alone as sufficient. The two numbers are per-run budgets, not sums across seeds; count successful `optimizer.step()` events explicitly.
6. Do not change the Step 1/2 KLPO estimator, its numerical conventions, or historical experimental artifacts. Do not begin Step 4.

## Reference sources and source-of-truth policy

- Paper: https://openreview.net/forum?id=bm3rbtEMFj and https://arxiv.org/abs/2510.07151
- Official repo: https://github.com/CognitiveAISystems/RATE
- Model: `offline_rl_baselines/ELMUR/model.py`
- Trainer: `src/trainer.py`; dataloaders: `src/utils/dataloaders.py`
- RememberColor3 native configuration: `run_experiments/MIKASA_Robo/ELMUR_1/rc3.sh`
- AutoencodeEasy native configuration: command in `offline_rl_baselines/ELMUR/README.md`
- Previous pinned RATE revision (if still available): `8abce2dce9ceacd411d46d4a99c287f7975892f5`.

Record the exact repository commit used and provide a field-by-field comparison of settings. Where this specification's rounded numbers differ from the pinned upstream command, use the *actual pinned command* and record the discrepancy. Do not assume that current `main` is identical to the pinned/reference commit.

## A. Upstream architecture — mandatory

| Parameter | RememberColor3-v0 | AutoencodeEasy |
|---|---:|---:|
| `d_model` | 128 | 64 |
| `n_layer` | 4 | 12 |
| `n_head` | 16 | 4 |
| `memory_size` (slots **per layer**) | 256 | 8 |
| `d_ff` | 128 | 128 |
| `training.context_length` | 20 | 35 |
| `training.sections` | 3 | 3 |
| effective sample length | 60 | 105 |
| `memory_init_std` | 0.1 | 0 |
| new-candidate weight `lru_blend_alpha` | 0.40683708068133906 | 0.79 |
| relative positional encoding | Yes | Yes |
| causal self-attention mask | Yes | Yes |
| `sequence_format` | `s` | `s` |
| MoE enabled | Yes | Yes |
| number of experts | 16 | 1 |
| top-k experts | 2 | 1 |
| shared experts | 1 | 2 |
| memory dropout | 0.05486068387428239 | 0.17 |
| model dropout | 0.1266135150715325 | 0.14 |
| attention dropout | 0.29604715763026573 | 0.26 |
| MoE load-balancing coefficient | 0.1 | 0.1 |
| `detach_memory` | True | True |

Also match task-specific `norm_type`, `pre_lnorm`, `routed_d_ff`, `shared_d_ff`, `use_shared_expert`, `use_swiglu`, label smoothing, token embedding, memory read/write ordering, attention implementation, and output heads from the pinned upstream command/model, including values not listed in this table. In particular, Autoencode uses RMSNorm in the upstream command. Inspect and preserve the actual semantics of the flags named `use_tok2mem` and `use_mem2tok`: the names are potentially counterintuitive; confirm read vs write from code.

Initialize a **separate layer-local memory state per ELMUR layer**, using upstream's random normal vectors scaled by `memory_init_std` and positions initialized to -1. Reset memory at each training batch/episode as upstream does. Preserve batch and layer dimensions exactly. Do not collapse the multilayer memory bank into one shared two-slot bank.

Use observation-only `sequence_format="s"`; expert actions are labels, not current-action input tokens. Return-to-go belongs to manager training targets and must not accidentally become a reader input token. Verify causality (the current expert action cannot leak through input preparation).

## B. Reader training — mandatory

| Parameter | RememberColor3-v0 | AutoencodeEasy |
|---|---:|---:|
| total **optimizer updates** | **9,600** | **57,600** |
| reference dataset trajectories (for budget derivation only) | 1,000 | 3,000 |
| batch size | 64 | 128 |
| upstream reference epochs | 200 | 800 |
| optimizer | AdamW | AdamW |
| learning rate | 0.00020459570733760105 | 0.00012 |
| AdamW beta1/beta2 | 0.99 / 0.99 | 0.99 / 0.99 |
| weight decay | 0.001 | 0.1 |
| global grad clip | 5 | 5 |
| configured warmup steps | 30,000 | 50,000 |
| cosine decay | Yes | No |

Use the pinned upstream LR scheduler semantics, including the true meaning/unit of warmup steps; **do not invent warmup reinterpretations** just because Autoencode's warmup setting is large relative to a smaller local data regime. Match remaining scheduler values (`final_tokens`, `lr_end_factor`) from native command.

### Training loop semantics

- Each training sample is split into three chronological sections, 20 or 35 tokens each; memory persists across sections and is detached before each next section, as in upstream.
- Compute behavioral-cloning prediction loss for **every section**, including valid-target masks/padding. Perform `zero_grad`, `backward`, grad clip, and `optimizer.step()` for each section, not one accumulated update for the three sections. Update scheduler as the upstream trainer does.
- Continuous expert actions (RememberColor) use upstream regression/MSE semantics. Discrete suits (Autoencode) use upstream categorical cross-entropy and upstream task-specific label-smoothing setting. Exclude padded targets.
- Retain the upstream MoE auxiliary loss and its coefficients; verify that the writer's relevant MoE parameters are optimizer members. Preserve upstream detach boundaries; do not add gradients through earlier segment writes to compensate for absent writer task gradients.
- Remove the local candidate-reconstruction head, its weight, optimizer states, checkpoint fields, diagnostics, and config defaults. Old checkpoints requiring that head are incompatible with new preparation and should not be silently resumed.
- Count optimizer updates in a persistent integer counter (increment **after** successful `optimizer.step()`). Save counter in checkpoints and stop at exactly the requested target. The fixed update count is authoritative even for freshly generated Autoencode episodes where epoch count need not match the upstream finite dataset's epoch calculation.
- For reference, upstream nominal count is `epochs * ceil(N/B) * 3`: Autoencode `800*ceil(3000/128)*3=57,600`; RememberColor `200*ceil(1000/64)*3=9,600`. These counts assume `drop_last=False`. If using a split smaller than the nominal 1,000 RememberColor files, the target update count stays 9,600; report repeated exposure and effective epochs explicitly.
- Run multiple validation checkpoints, with the **same fixed held-out evaluation protocol**; retain the best validation checkpoint and the last checkpoint, and report both. Do not tune or select on the held-out test set.

### Data, labels, and inputs

**AutoencodeEasy:** retain fresh shuffled native POPGym 1.0.7 episodes with the original 52-card reverse-suit task and oracle-suit label for each PLAY decision. This is an intentional controlled-experiment data choice, not an exact replay of the published PPO-GRU dataset. The action space is exactly `Discrete(4)` (four suits), with **no WAIT action**. Actions during WATCH are ignored by the environment: process WATCH observations to build memory but do **not** compute suit CE on arbitrary WATCH actions. Mask CE to PLAY positions only. Verify the last-card/first-PLAY boundary, correct reverse-label ordering, and that privileged labels do not enter the reader/manager observations or memory candidates. Keep the full-length evaluation distribution 52 cards. Disable the old short-length curriculum in the main upstream-budget reproduction; archive its 500-update results as historical diagnostics.

**RememberColor3:** use the already-audited published offline expert trajectories, RGB-only two-camera observation path, and continuous 8D expert actions. No simulator required for offline BC. Preserve verified preprocessing and target units; do not silently clamp targets or train with inferred oracle data. Report metrics by cue/delay/control phase, in addition to aggregate MSE. Closed-loop native evaluation is optional only when a functioning renderer/runtime is available; do not interpret expert dataset rewards/success flags as reader-policy success.

## C. Upstream reproduction gate — mandatory before learned-manager comparisons

Implement and evaluate an **independent upstream-style fixed-LRU ELMUR system** using the matched architecture and training budget. It must reproduce upstream write order, empty-slot handling, slot timestamps, and fixed candidate blending (the new-candidate coefficient is `lambda`; retention coefficient is `alpha=1-lambda`). Document any intentional data differences (especially Autoencode oracle labels versus PPO-GRU data and any RememberColor subset). Do not claim replication of published headline performance if data/evaluation differ.

Required engineering checks: model forward shapes per layer and section; valid backward/update; finite gradients; no privileged-input leakage; correct 3-segment memory persistence/detach; exact step count; properly masked action loss; MoE aux loss; consistent resume; aligned training/evaluation inputs; genuine-memory versus zero/shuffled-memory diagnostics. Demonstrate nontrivial, stable imitation-learning improvement and an auditable upstream implementation before advancing. Do **not** fabricate an absolute success threshold from the paper. If reproduction fails, stop learned-manager conclusions, report diagnostics, and keep artifacts.

The original Step 3 bootstrap reader memory-dependence readiness gate remains separately defined and must not be silently relaxed. Upstream-LRU reproduction and learned-manager readiness are **distinct** gates. An upstream-LRU-trained reader is not automatically valid under a different stochastic manager's memory distribution.

## D. Learned manager integration — preserve research method, specify granularity

### Write-event audit (must occur before coding manager integration)

Inspect the pinned upstream model and provide a machine-checkable write-event contract describing:

- where each layer produces its new memory candidate(s), their exact tensor shapes, and whether writes are per token, per segment, or some other index;
- layer-local incoming/outgoing memories and timestamp semantics;
- how multiple writers/events are ordered inside one forward pass;
- the relationship between forward/prediction time and each write event;
- whether an event writes to one or multiple slots and how the native LRU chooses its target;
- which event observations/features are causally available to the learned manager.

**Default integration principle:** replace only the upstream slot-selection/blending decision at the *same actual upstream write events*, without changing tok2mem candidate generation, read path, token budget, or update frequency. Make one manager action per native write decision for each relevant layer/event, unless upstream's actual event model forces another mapping. If the existing Step 2 interface cannot express that granularity, adapt the interface and its tests; do not silently collapse, duplicate, or aggregate events. The mapping from manager decision index to physical timestep, segment, layer, and later imitation reward must be explicit and logged.

### Manager action semantics

Use the established categorical manager action `u=(slot, alpha)` plus a global no-op where allowed; `alpha` is **retention**, with occupied-slot update

`m_new = alpha * m_old + (1 - alpha) * candidate`.

Preserve the established Step 2/3 action coefficient grid from the checked-in implementation as the canonical source; assert its exact ordered values and action ID mapping in tests rather than silently inventing or shifting a grid. If a discrepancy with the historical experiment specification is discovered, report it before proceeding.

**Empty slots:** when a selected slot is empty (`pos==-1`), hard-code `m_new=candidate` irrespective of chosen alpha, and set its timestamp according to upstream semantics. Do not blend with the initial Gaussian/zero vector. Keep the chosen action's correct log-probability and logging even when alpha has no content effect.

**Eligibility:** for the main upstream-fidelity condition, enforce the original empty-slot-first constraint until the layer's bank is full: restrict slot choice to the upstream-determined empty target, then allow the manager to select among occupied slots. The global no-op is disallowed during mandatory initial filling. Implement these as explicit **legal-action masks** consistently in behavior sampling, manager logits/probability normalization, exact score-centering, KL computation, and greedy evaluation; never sample an action then force-change it without correcting its policy probability. Log this restriction and its implication: manager decisions before the bank fills are deterministic and cannot establish coefficient learning. Optionally retain an unrestricted-manager ablation under a **separate** named configuration; do not conflate it with the upstream-fidelity treatment.

For full banks, preserve Step 2 categorical management, including global no-op and timestamp behavior: `alpha=1` refreshes the selected timestamp while preserving content; global no-op preserves both. Add test coverage for both.

Manager features must be constructed from the corresponding layer's causal state and candidate, preserving the established `(e(o), M, Z, age, t)` feature definitions where the original manager requires them; if upstream's event representation changes those meanings, document the exact mapping and validate shapes. **Do not hard-code a 2-slot flat feature tensor in a 256-slot system.**

### Optimization and evaluation

- Keep the existing validated Step 2 RTG-KLPO estimator, score-centering and sampler refresh conventions unchanged in mathematics. Reader/writer frozen during manager optimization.
- **Manager reward is imitation quality**, not environment reward used as a substitute. Define event reward using held-out-from-current-action *future* expert-action log likelihood or the established Step 2/ELMUR-KLPO imitation-reward convention, with explicit valid-action masks and precise reward-to-write alignment. Native environment rewards may be reported separately, but may not silently replace the imitation objective.
- Events in a multi-layer/multi-token forward pass may be simultaneous with respect to environment timestep. Do not naively assign `t+1` environment reward to every layer event. Specify the event ordering and ensure no policy action is credited with an action prediction that occurs before the write can influence the reader. Test this on a tiny deterministic example.
- Preserve original no-critic, no-memory-history-BPTT manager learning and Step 2 finite-difference/identity tests. Extend gradient/log-prob correctness tests to legal-action masking and new layer/write-event shapes.
- Use independently trained, equally budgeted fixed-LRU and learned-manager readers for system-level comparisons. For a frozen-reader manager-only diagnostic, label it as such. Do not interpret a memory-distribution shift as a genuine manager advantage.
- Keep the frozen initialization policy, slot-only, coefficient-only, and full manager ablations as separate, explicitly specified variants; maintain matching reader-training/evaluation protocol and management rollout budget across variants. Preserve the previously defined stopping/readiness criteria; on failure report rather than bypass.

## E. Validation, reporting, and acceptance criteria

1. **Configuration match:** emit reference-versus-local configuration diff for both tasks, including every architecturally or optimization-relevant option and pinned commit.
2. **Reproduction tests:** compare output/tensor shapes, memory init stats, 3-section behavior, layer-local detach, masking, LRU empty/full writes, MoE auxiliary, and action-loss calculations to the pinned upstream implementation on deterministic fixtures.
3. **Exact training count:** Autoencode reader has **57,600** completed optimizer steps, RememberColor reader **9,600**; verify after resume and from saved logs/counters. Do not confuse minibatches, trajectories, sections, or epochs with updates.
4. **Datasets:** use previously audited splits; preserve train/validation/test isolation and report actual number of unique episodes seen and repeated sampled presentations. No new dataset audit unless data-integrity evidence changes.
5. **Diagnostic metrics:** Autoencode recall CE, full-length accuracy, native return, phase/position accuracy, genuine-vs-zero-vs-shuffled memory. RememberColor aggregate/phase-wise action MSE and memory ablations, plus native success only if genuinely executed with a supported renderer. Report uncertainty across episode/seed units and independent reader seeds when available.
6. **Baselines/gates:** upstream-style LRU reader passes reproducibility and learning checks before manager stage; the separate reader memory-readiness gate is evaluated honestly and stops manager stage when it fails.
7. **Manager correctness:** every sampled event has a valid legal action, normalized masked probabilities, correct stored log-prob, correct credit-assignment index, and correct empty/full/no-op timestamp update. Frozen reader parameters do not change during manager training.
8. **Regression:** existing Step 1/2 exact objectives and finite-difference tests remain passing. Prior Step 3 artifacts/checkpoints are archived; changed configurations write new artifact roots with explicit provenance.
9. **Resource accounting:** report unique trajectories, sample/transition presentations, optimizer steps, model forward passes, total GPU time, memory peak, validation overhead, and failed environment integrations. If training is blocked by compute/runtime, implement and run verification/smoke gates and report precisely what did not execute; do not claim the full budgets completed.

## Execution order

1. Snapshot old artifacts; pin upstream commit and write configuration/gradient/write-event audits.
2. Remove reconstruction and reproduce the task-specific reader/writer, training loop, and fixed-LRU semantics.
3. Pass deterministic upstream-parity smoke tests on both tasks, including complete 3-segment minibatches and checkpoint resume.
4. Run full-budget reader preparation: **Autoencode 57,600**, **RememberColor 9,600** updates; validate periodically and retain best/last.
5. Evaluate fixed-LRU reproduction and memory-dependence diagnostics. Stop and diagnose if reproduction/readiness fails.
6. Integrate and test learned-manager action masks and event-level credit assignment; then run matched, gated manager comparisons and ablations.
7. Write an implementation report and a results report separating (a) reproduction fidelity, (b) reader memory dependence, and (c) manager effects. Explicitly enumerate unfinished tasks and deviations from upstream.

**Do not silently weaken gates, redefine the task, use privileged state as a model input, or present offline demonstration outcomes as closed-loop policy outcomes.**
