# Step 3 measured results — 2026-10-13

**Outcome D: inconclusive.** The implementation and CPU integration checks are
in place. Autoencode's prepared reader did not pass the required held-out
memory-dependence gate. The native RememberColor renderer could not initialize
on this host. Consequently, the research manager-training and fixed-system
comparisons were not executed. These results do not establish successful
coefficient elimination or adaptive blending.

Step 2 was committed first as `5830178`, preserving its measured results and
historical artifacts. Step 4 was not started.

## 1. Exact task configurations

| Setting | AutoencodeEasy | RememberColor3-v0 |
| --- | --- | --- |
| Native source | POPGym 1.0.7 | MIKASA commit `509b875f3d207c287497c0a897661062de928bb0` |
| Target | 52 shuffled cards, reverse suit recall | 3 color choices, `delta_time=5`, 60-step limit |
| Reader observation | Six one-hot mode/suit features | Two RGB cameras, `[6,128,128]`; no oracle/proprioception |
| Task action | Four discrete suits | Eight bounded joint-delta controls |
| Manager reward | Actual native episodic reward | Actual native normalized dense reward |
| Empty memory | Upstream zero vectors, positions −1 | Upstream random initialization std 0.1, positions −1 |
| Small reader | 1 ELMUR layer, 2 slots, width 32, single-token windows | Same configured architecture |
| Reader seed | 17 | 17, initialization only; no native trajectories completed |
| Data seed | 1000, with separate split blocks | Same configured split scheme |

The smoke uses width 16 and four-episode training batches. The larger Autoencode
preparation uses width 32 and batches of 16. Both use the original ELMUR task
encoder, reader and slot-conditioned tok2mem. These settings are a small Step 2
extension, **not a reproduction of the paper's architecture or budgets**.

Verified native script coefficients differ slightly from rounded paper values:

| Task | Native script new-candidate weight | Equivalent retention | Rounded paper retention |
| --- | ---: | ---: | ---: |
| RememberColor3 | 0.40683708068133906 | 0.5931629193186609 | 0.60 |
| AutoencodeEasy | 0.79 | 0.21 | 0.20 |

See [implementation and source links](step_3_implementation.md) for the pinned
RATE scripts and MIKASA source. The sweep includes rounded paper equivalents
0.6 and 0.2; native LRU uses the verified script weights. None is called an
optimal coefficient.

## 2. Curriculum and reader preparation

Autoencode preparation ran the configured **500 updates**:

| Sequence length | Updates | Validation return at end of phase |
| ---: | ---: | ---: |
| 2 | 50 | +0.62500 |
| 4 | 50 | +0.18750 |
| 8 | 75 | −0.12109 |
| 16 | 75 | −0.40820 |
| 32 | 100 | −0.46973 |
| 52 | 150 | −0.46514 |

Shorter phases sample shorter prefixes of the original shuffled deck while
retaining all suits, the reverse-recall objective, observation/action semantics,
native reward scale and native step implementation. Length two still requires
remembering an earlier card. The target 52-card distribution was both trained
and evaluated. The native last-card visibility at the first PLAY action is
preserved.

The initialization manager was frozen throughout, with identical slots,
conditional coefficient probabilities `(0.1,0.1,0.6,0.1,0.1)`, and no-op 0.02.
Autoencode reader preparation used expert-suit cross entropy. A local per-slot
candidate reconstruction loss, weight 0.1, trained tok2mem without gradients
through memory transitions. Its target was the current detached observation
encoding; it supplied no manager inputs or artificial manager rewards.

RememberColor's configured delays are 1, 3 and 5, with the original color-recall
objective. Its native runtime failed before any phase produced an episode.
This curriculum is implemented but has no measured robot-training result.

## 3. Reader readiness

All following full-length Autoencode readiness measurements use **64 held-out
validation episodes**, sampled initialization management, and native returns.
Confidence intervals bootstrap episodes with 2,000 fixed-seed resamples.

| Condition | Mean episodic return | Reconstruction accuracy |
| --- | ---: | ---: |
| Genuine memory | −0.46514 | 26.7428% |
| Memory contents ablated | −0.47957 | 26.0216% |
| Earlier encoder-write snapshots, oracle retrieval diagnostic | −0.51202 | 24.3990% |
| Uniform slots, replacement alpha=0 | −0.46334 | 26.8329% |
| Uniform slots, retention alpha=0.95 | −0.47296 | 26.3522% |

The paired genuine-minus-ablated return difference was **+0.014423**, with
95% interval **[−0.003005, +0.032452]**. Its lower endpoint does not exceed zero;
the memory-dependence gate **failed**. The improvement in accuracy is only
0.72 percentage points and is not supported by this gate. High full-task
success was not required.

The decoder diagnostic retrieves actual earlier tok2mem write snapshots using
native recall indices. It injects no synthetic vectors. Its poor performance
provides no positive evidence of useful decoding at the target length, although
snapshot-induced distribution shift is a confound. The management-variation
results are diagnostics, not proof of robustness to every valid policy.

The six-update smoke also failed the gate: contribution **−0.002404**, interval
**[−0.031250, +0.026502]**. Its target-length prediction loss was initially large
(215.25 at update 5, 55.50 at update 6); the larger curriculum reduced final
target-length cross entropy to **1.41083**, but did not establish supported
memory use. Smoke results are execution evidence only.

## 4. Frozen-reader coefficient comparisons

These are complete closed-loop **64-episode test** trajectories of the prepared
Autoencode reader. They use sampled management and greedy task actions. Each
fixed coefficient draws uniform slots; coefficient values are applied directly
and independently of categorical action IDs. These are diagnostic common-reader
comparisons, not independently trained fixed systems.

| Management | Mean return | Reconstruction accuracy |
| --- | ---: | ---: |
| Frozen initialization | −0.49159 | 25.4207% |
| Native LRU, new weight 0.79 | −0.47236 | 26.3822% |
| Global no-op | −0.48077 | 25.9615% |
| Fixed alpha 0.0 | −0.48498 | 25.7512% |
| Fixed alpha 0.1 | −0.48197 | 25.9014% |
| Fixed alpha 0.2 | −0.45913 | 27.0433% |
| Fixed alpha 0.5 | −0.45553 | 27.2236% |
| Fixed alpha 0.6 | −0.46815 | 26.5925% |
| Fixed alpha 0.8 | −0.47957 | 26.0216% |
| Fixed alpha 0.95 | −0.46635 | 26.6827% |
| Fixed alpha 1.0 | −0.48077 | 25.9615% |

These point estimates do not establish a statistically preferred coefficient.
The reader was prepared under the initialization mixture, not separately under
each coefficient. Alpha 1 refreshes a slot's timestamp; no-op preserves it.
They happened to have identical returns here, with zero initial memory.

## 5. Manager optimization and coefficient behavior

The configured study has three manager seeds (1, 2, 3), each receiving 100 fresh
batches of 32 trajectories for full, slot-only and coefficient-only management.
**None of these research runs started**, because the reader gate failed.
Accordingly there are no measured research manager learning curves, trained
coefficient distributions, per-seed improvements, H2 ablation effects, or
independent fixed-system performance estimates to report.

Mechanical CPU tests execute fresh native-reward updates for all three
variants, verify reader/manager gradient isolation, and exercise all nine
seed/variant orchestration conditions using a supplied readiness fixture.
These unit-test runs are not reader-readiness evidence or algorithm results.

The full-length **frozen initialization** validation rollout recorded conditional
write coefficient frequencies of **10.19%, 10.14%, 60.14%, 9.58%, 9.94%**.
Slot frequencies were **48.74% and 49.42%** of all decisions, with **1.84% no-op**.
There were 6,280 overwrites; mean selected-slot age was 2.02 steps and mean bank
change norm 18.77. These measurements verify the intended stochastic baseline
and repeated competition, not learned behavior.

## 6. Native RememberColor availability

A separate Python 3.11.15 runtime was actually provisioned with MIKASA at the
verified commit, Torch 2.2.1+cpu, NumPy 1.23.5, Gymnasium 0.29.1,
ManiSkill 3.0.0b15 and SAPIEN 3.0.0b1. The upstream setuptools compatibility
range was installed after an initial `pkg_resources` import failure.

With the default GPU renderer, native environment creation failed to find a
CUDA device. With CPU rendering explicitly requested, it reached native scene
construction and failed with:

```text
RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent
```

The final adapter makes simulation and rendering backends explicit and records
this failure as `native_task_unavailable`. No native robot episode completed;
no robot reader or manager checkpoint, success rate, memory-dependence score,
or capacity-sensitivity result is claimed. The RGB reader and eight-dimensional
bounded-action interfaces are CPU-tested independently, but full task integration
still needs a compatible native renderer. This is not a substitute benchmark.

## 7. Hypotheses and failure analysis

| Hypothesis | Evidence and conclusion |
| --- | --- |
| H1: learning improves management | Unresolved; no research manager training passed the readiness prerequisite |
| H2: coefficients contribute independently | Unresolved; controlled variants are implemented and mechanically tested, not evaluated as a study |
| H3: manual tuning can be replaced | Unresolved; the diagnostic sweep cannot replace matched independently trained fixed systems |
| H4: adaptive state-dependent blending | Unresolved; no trained decisions or causal retention/replacement probes exist |
| H5: results generalize across tasks | Unresolved; native RememberColor could not run |

Reader limitations precede manager optimization here. The short task is learned,
but reconstruction accuracy declines as memory competition and sequence length
increase. Failure A (insufficient supported reader memory use) is demonstrated
by the failed target gate. Failure B (unable to decode useful memory) is a
plausible explanation, with distribution shift in the snapshot protocol a
confound. Failure C (management-distribution incompatibility) and Failure E
(weak coefficient sensitivity) remain possible. Neither manager optimization
failure nor slot-selection dominance can be diagnosed without training.

Increasing manager iterations would bypass the actual limitation. A follow-up
should first improve target-length reader preparation/representation and validate
encoder-produced informative trajectories, keeping the preparation manager
frozen. The current two-slot, one-layer, one-token architecture may be insufficient;
this study did not isolate capacity from curriculum or representation learning.
Color object count does not establish required slot count, and true robot
capacity pressure remains unmeasured. No task objective was changed to create
a coefficient effect, and the failed gate was not relaxed.

## 8. Compute and validation

Hardware: AMD Ryzen 9 4900HS CPU, one Torch thread. Main runtime: Python 3.12.13,
PyTorch 2.14.1+cpu. No CUDA training was available.

| Stage | Training interactions | Measured training seconds |
| --- | ---: | ---: |
| Autoencode smoke reader, 6 updates × 4 episodes | 968 | 1.33 |
| Autoencode curriculum reader, 500 updates × 16 episodes | 411,200 | 134.82 |
| Research manager rollout/update | 0 | Not started |
| Independent fixed-system study | 0 | Not started |
| Native robot training | 0 | Environment initialization failed |

Reader throughput over the larger run was approximately **3,050 environment
steps/s**. Training reader-forward counts equal 411,200 episode-observation
evaluations, batched across 16 episodes. The final genuine-memory validation
rollout performed 6,592 observation evaluations; compact artifacts retain its
separate elapsed time and throughput. CUDA peak allocation is zero. Reader
checkpoints are approximately 0.74 MB for the larger preparation and 0.30 MB
for smoke. Timings exclude validation, checkpoint I/O, dependency installation
and main-process startup; see the implementation document for timer boundaries. These are
single-host measurements, not a throughput comparison between algorithms.

The validation suite covers exact initial probabilities and empirical sampling,
uniform slots, coefficient conventions, per-slot candidate preservation,
continuous fixed coefficients and no-op ages, native Autoencode equivalence,
rewards/termination, discounted post-write returns, freezing, fresh-policy
checks, independent splits and RNG preservation. Interrupted reader and manager
runs restore exactly to uninterrupted parameter values on CPU. Completed smoke
and the larger preparation were resumed without retraining or changing recorded
episode measurements. Optional CUDA and full native robot tests are skipped on
this host; their runtime limits are recorded rather than reported as passes.

Final verification: **118 tests passed, 2 skipped** (CUDA and the optional full
native robot episode), Ruff lint/formatting passed, and the package built.
Step 1 A–G and all-parameter finite differences passed unchanged: worst
regression-gradient identity error **2.36×10⁻¹⁵**, worst finite-difference error
**3.39×10⁻¹¹**, below the `1e-8` absolute tolerance.

## 9. Artifacts and exact executed commands

Compact measurements, per-episode returns, preparation curves, fixed sweeps,
failure records and the actual native dependency manifest are in:

- [experiments/step3/measurements.json](experiments/step3/measurements.json)
- [experiments/step3/native_runtime.json](experiments/step3/native_runtime.json)

Full histories and phase/final reader checkpoints remain under ignored
`runs/step3_smoke_final/` and `runs/step3_reader_preparation/`. Native retry
tracebacks also remain in `runs/step3_robot_cpu_renderer/`. Initial failed
attempts were retained in separate directories. Step 2 measurements and
historical artifacts were preserved; the reusable reader and CLI extensions
retain their Step 2 defaults.

The primary entry points are `prepare_reader` in `step3/preparation.py`,
`collect_native` in `step3/rollout.py`, `train_manager` in `step3/training.py`,
and the unchanged REINFORCE branch of `elmur_klpo/trainer.py`. Full commands
for every supported stage, native setup and CUDA are in
[step_3_implementation.md](step_3_implementation.md).

Executed experiment commands (exit 2 correctly denotes failed gates):

```bash
export UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache
export ELMUR_NATIVE_PYTHON=/tmp/step3-native/bin/python
uv run python -m elmur_klpo.experiment --config configs/step3_smoke.json --output runs/step3_smoke_final
uv run python -m elmur_klpo.experiment --config configs/step3_smoke.json --output runs/step3_smoke_final --resume
uv run python -m elmur_klpo.step3.experiment --config configs/step3_reader_preparation.json --output runs/step3_reader_preparation --stage prepare --task autoencode
uv run python -m elmur_klpo.step3.experiment --config configs/step3_comparison.json --output runs/step3_reader_preparation --task autoencode --stage compare --resume
uv run python -m elmur_klpo.step3.experiment --config configs/step3_smoke.json --output runs/step3_robot_cpu_renderer --task remember_color
uv run python -m elmur_klpo.step3.aggregation --inputs runs/step3_smoke_final runs/step3_robot_cpu_renderer runs/step3_reader_preparation --output experiments/step3/measurements.json
```

The larger preparation and comparison task configurations match exactly;
the second command resumes the completed reader and adds the diagnostic sweep.
Final test scores were not used to adjust hyperparameters or readiness criteria.

## 10. Follow-up: RememberColor offline reader on CUDA — reported 2026-10-14

**Outcome D remains inconclusive.** The run in
[runs/step3_remembercolor_reader_gpu/results.json](runs/step3_remembercolor_reader_gpu/results.json)
completed all **500 reader-preparation updates**, using downloaded original
RememberColor3-v0 expert trajectories. Its status is `reader_gate_failed`:
offline imitation did not demonstrate beneficial memory use, and the required
native closed-loop reward memory-dependence gate was not evaluated. No manager
training or system-level coefficient comparison is reported by this run.

This follow-up adds a CUDA-trained robot reader checkpoint to the earlier CPU
findings. The earlier statements about unavailable CUDA training and absent
robot checkpoints describe those historical runs. Offline preparation here
required no simulator or renderer; it does not establish that the native
renderer now works.

### Configuration and data provenance

The run used [configs/step3_remembercolor_reader.json](configs/step3_remembercolor_reader.json):
two memory slots of width 32, initialization standard deviation 0.1, reader seed
17, data seed 1000, Adam learning rate 0.0002, batch size 8, reconstruction
weight 0.1, and validation every 25 updates. Preparation used the recorded
target task (configured difficulty 5) throughout, with no shorter-delay
curriculum. The initialization manager remained frozen, with conditional
coefficient probabilities `(0.1, 0.1, 0.6, 0.1, 0.1)` and no-op probability 0.02.
Training used float32, expert-action MSE, and a candidate reconstruction
auxiliary loss on the first five timesteps; memory writes were detached.

The [dataset manifest](experiments/step3/remember_color_dataset.json) identifies
the published `avanturist/mikasa-robo` release at revision
`c362f52143c61e7f778bdb5d410ce4c22a7185d4`, archive `RememberColor3-v0.zip`,
SHA256 `388b16d0984cb19d6a6cd20fe0029a67fef100aed378a195435f1f0acf5b7d82`.
The result records the same archive checksum. Its 1,000 downloaded episodes
contain 60 transitions each; filename-disjoint splits allocate 800 training,
100 validation, and 100 test episodes. Source collection seeds are absent, so
seed-disjointness and correspondence to the original ELMUR splits are unverified.
The reported scores use the validation split; no test-split score is reported.

Inputs were recorded two-camera RGB observations, scaled by 255; targets were
recorded eight-dimensional actions explicitly clipped to `[-1, 1]`. These are
downloaded demonstrations, not newly generated episodes or oracle-derived
labels. The training history records 44,426 clipped action-element presentations
across repeated sampled batches, not that many distinct dataset elements.
Recorded expert rewards and success flags are not returns or success rates of
the prepared reader.

These measurements describe the historical clipped-target run. The current
`configs/step3_remembercolor_reader.json` instead preserves raw expert actions
and uses an unbounded linear reader output. No results for that configuration
are reported here; it requires a fresh preparation run rather than resuming the
historical checkpoint.

### Measured preparation and memory ablation

| Measurement | Value |
| --- | ---: |
| Training action MSE, update 1 | 0.917071 |
| Training action MSE, update 500 | 0.345370 |
| Candidate reconstruction loss, update 1 → 500 | 0.926003 → 0.080490 |
| Validation action MSE, update 25 | 0.817967 |
| Validation action MSE, update 250 | 0.544349 |
| Validation action MSE, update 500, genuine memory | 0.379940 |
| Validation action MSE, memory contents zeroed | 0.102966 |
| Paired ablated-minus-genuine MSE | −0.276974 |
| 95% interval for paired difference | [−0.292680, −0.260622] |

The final ablation compares the same 100 validation episodes with memory
contents zeroed at each read, retaining memory positions. The interval uses
episode-level percentile bootstrap with 2,000 resamples. Positive
ablated-minus-genuine MSE would indicate an imitation benefit from memory;
the measured negative difference instead shows substantially lower error
under ablation. Training and validation losses improved over preparation,
but these measurements provide no evidence of useful memory decoding in this
protocol. They do not isolate the cause or measure closed-loop task performance.

The readiness record explicitly states: “Native closed-loop reward
memory-dependence gate has not been evaluated; offline imitation scores cannot
satisfy it.” Thus `reader_gate_failed` must not be interpreted as a measured
failure of a native reward rollout. There are no measured reader-policy native
returns, native success rates, manager learning curves, robustness results at
difficulties 10/20, or independent fixed-system comparisons in this artifact.
The configured three manager seeds and variants are prospective budgets only.

### Compute, artifacts, and executed command

The result records device `cuda`, CUDA available, and PyTorch `2.7.1+cu126`.
The job started on 2026-10-13 at 18:30:37 Asia/Taipei. GPU model and end-to-end
wall time are not recorded in these artifacts.

| Resource measurement | Value |
| --- | ---: |
| Reader updates × episodes per batch | 500 × 8 |
| Recorded transition presentations / reader-forward evaluations | 240,000 |
| New environment interactions | 0 |
| Summed training-update time | 713.23 s (11.89 min) |
| Training throughput from that timer | 336.50 transition presentations/s |
| Maximum recorded CUDA peak allocation | 132,821,504 bytes (126.67 MiB) |
| Reported checkpoint size | 5,164,241 bytes (4.93 MiB) |

The update timer synchronizes CUDA and covers reader forward/backward passes,
memory transitions, and optimizer updates. It excludes dataset loading and
transfer before each update, validation, checkpoint I/O, and startup. CUDA peak
allocation is a training-update measurement, not total process or device memory.

The exact launch recorded in
[job.json](runs/step3_remembercolor_reader_gpu/job.json) was:

```bash
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache UV_PROJECT_ENVIRONMENT=runs/step3_gpu \
uv run --no-sync python -m elmur_klpo.experiment \
  --config configs/step3_remembercolor_reader.json \
  --output runs/step3_remembercolor_reader_gpu --device cuda
```

The entry point routes to `prepare_demonstration_reader` in
[elmur_klpo/step3/offline_preparation.py](elmur_klpo/step3/offline_preparation.py);
`demonstration_update` performs training and `demonstration_evaluation` computes
held-out imitation scores. Full history, split manifest, reader report, and
`reader.pt` remain under `runs/step3_remembercolor_reader_gpu/remember_color/`;
the parent directory retains `training.log`, `job.json`, and `results.json`.
This report was checked against those results, the launch record, configuration,
dataset manifest, and measurement code. No training, resume check, or test suite
was rerun for this documentation append.
