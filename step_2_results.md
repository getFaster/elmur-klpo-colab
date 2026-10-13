# Step 2 measured results

Executed 2026-10-12 on an AMD Ryzen 9 4900HS CPU, one Torch thread,
Python 3.12 and PyTorch 2.14.1+cpu. Both experiments use the current
slot-conditioned adapter: upstream candidates retain shape `[N, slots, width]`,
manager inputs contain every candidate, and writing slot `k` uses its own
`candidate[n,k]`. No slot averaging is used.

The smoke run and three-seed comparison completed, including closed-loop
evaluation and checkpoint restoration. Resuming the completed comparison
reproduced all measurements and training histories exactly, excluding elapsed
evaluation time. All 600 comparison updates reported zero numerical failures.

## Validation and reader preparation

Code validation for this adapter: **92 tests passed**, Ruff lint and formatting
passed, and numerical gradient verification passed A–G and all-parameter
finite differences. Worst regression-identity error was **2.36e-15** and
finite-difference error **3.39e-11**. No implementation changes were needed
for these experiment runs.

The reader was prepared with 500 native-LRU behavior-cloning updates, then
frozen and shared across all methods and manager seeds. The mandatory
64-episode validation gate passed: junction NLL **0.02129** with memory versus
**0.97490** ablated, and junction accuracy **98.44%** versus **57.81%**.

Reader preparation and manager training use the same 256 training episodes,
sampled separately into batches. They do not use disjoint training datasets.
The validation and test splits are separate from training.

## Controlled comparison

The supplied configuration uses 256 training episodes, 64 validation episodes,
64 test episodes, two slots, width 32, and a maximum of ten expert-action
timesteps. REINFORCE and KLPO each receive **100 fresh batches of 32
trajectories per seed**, or **3,200 trajectories per method per seed**,
for manager seeds 1, 2, and 3. Adam learning rate 0.001 and gradient clipping
at 5 are identical. KLPO beta is 0.1; REINFORCE has no KL penalty.
Management actions are sampled during evaluation; reader task actions are
greedy in closed loop.

The following are means over three manager seeds on the shared test split.
Closed-loop evaluation uses independent observation noise.

| Method | Expert-action NLL | Expert-action accuracy | Imitation return | Junction NLL | Junction accuracy | Closed-loop success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native ELMUR-LRU | 0.00026 | 100.00% | -0.00210 | 0.00200 | 100.00% | 100.00% |
| Uniform random | 0.48577 | 94.27% | -3.88619 | 3.78774 | 56.77% | 51.04% |
| Forced no-op | 0.13717 | 92.97% | -1.09736 | 1.09736 | 43.75% | 51.04% |
| RTG-REINFORCE | 0.13147 | 96.16% | -1.05176 | 1.05015 | 69.27% | 68.75% |
| RTG-KLPO | 0.15352 | 95.83% | -1.22815 | 1.22811 | 66.67% | 66.15% |

| Seed | Initial learned junction accuracy | REINFORCE final | KLPO final | REINFORCE closed loop | KLPO closed loop |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 62.50% | 75.00% | 67.19% | 67.19% | 68.75% |
| 2 | 51.56% | 51.56% | 56.25% | 56.25% | 59.38% |
| 3 | 54.69% | 81.25% | 76.56% | 82.81% | 70.31% |

Mean validation junction accuracy during training:

| Fresh batches | REINFORCE | KLPO |
| --- | ---: | ---: |
| 20 | 54.17% | 54.17% |
| 40 | 60.94% | 58.33% |
| 60 | 64.58% | 66.15% |
| 80 | 65.62% | 70.31% |
| 100 | 67.71% | 66.15% |

## Memory and action diagnostics

With memory ablated, both learned methods score **43.75%** junction accuracy
in every seed. Forced no-op also scores **43.75%** offline and **51.04%**
closed-loop success on average. Both learned methods exceed these baselines
on average; REINFORCE seed 2 shows no offline junction-accuracy improvement
over its initial manager.

Final KLPO action utilization, as percentages of all valid decisions:

| Seed | alpha=0 | alpha=0.25 | alpha=0.5 | alpha=0.8 | alpha=0.95 | Slot 0 | Slot 1 | No-op |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.45% | 0.45% | 98.21% | 0.45% | 0.45% | 61.16% | 38.84% | 0.00% |
| 2 | 0.00% | 6.47% | 5.13% | 62.50% | 1.79% | 12.28% | 63.62% | 24.11% |
| 3 | 1.12% | 3.12% | 95.09% | 0.45% | 0.00% | 58.71% | 41.07% | 0.22% |

Alpha and slot frequencies exclude no-op, so each group sums to one minus
the no-op frequency. KLPO seeds 1 and 3 concentrate on alpha=0.5; seed 2
prefers alpha=0.8 and slot 1, with substantially more no-ops. Every seed uses
both slots. The artifacts include action distributions, memory-change norms,
unchanged-bank fractions, before/after metrics, and ablations for both
learned methods.

## Compute and smoke run

Reader preparation took **20.27 seconds**. The following totals cover
300 updates per algorithm across the three seeds; throughput is the mean
per-update decision throughput. Evaluation, dataset construction, checkpoint
I/O, and process startup are excluded.

| Algorithm | Rollout seconds | Update seconds | Decisions/s | Mean gradient norm | Mean KL(q‖p) | Mean KL(p‖q) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| REINFORCE | 7.63 | 0.49 | 8,933 | 80.37 | 0.004586 | 0.004363 |
| KLPO | 7.15 | 0.46 | 9,424 | 79.06 | 0.004669 | 0.004446 |

Gradient norms are measured before clipping, and both KL directions after
each update on recorded states. Peak GPU allocation is zero; these runs
used CPU.

The current-adapter smoke run completed five updates per learned method,
the reader memory gate, all baselines, closed-loop evaluation, and checkpoint
restoration. Both learned methods scored **59.375%** offline junction
accuracy and **62.5%** closed-loop success. The smoke budget verifies
execution and does not establish learning quality.

## Interpretation and limits

With slot-conditioned candidates, REINFORCE achieves **69.27%** mean test
junction accuracy and **68.75%** closed-loop success, versus **66.67%** and
**66.15%** for KLPO. REINFORCE also has lower mean junction NLL. KLPO is
better on some seed-specific measurements, but the aggregate results do
not support a KLPO advantage at this budget. Native LRU achieves **100%**
on both accuracy measures. Three manager seeds and one shared reader
preparation seed are too small a study to establish algorithm superiority.

All methods now use the upstream slot-conditioned candidate definition.
The reader is still trained under native LRU, and LRU uses its original
slot selection and blending conventions, including empty-slot behavior.
Its performance gap therefore cannot be attributed solely to learned
management quality.

With one fresh rollout and one update, p=q at gradient evaluation, so
KLPO beta's ratio contribution is zero. This comparison tests exact score
centering rather than an active first-step KL penalty. The scalar losses
are gradient surrogates, not exact expected-return objectives.

The earlier slot-averaged results are preserved separately. Changing the
candidate interface also changes manager input dimensions and sampled
trajectories, so historical differences are not a paired experiment holding
the manager architecture fixed.

The supplied small Step 2 experiment is complete. Paper-scale training,
additional reader seeds, CUDA, mixed precision, longer horizons, and other
tasks remain untested. Steps 3 and 4 were not run.

## Artifacts and reproduction

The experiment is implemented by `run` in
[elmur_klpo/experiment.py](elmur_klpo/experiment.py): generate independent
T-Maze splits, prepare and freeze the native reader, check its memory dependence,
evaluate fixed baselines, and train paired managers with fresh rollouts.
[rollout.py](elmur_klpo/rollout.py) records trajectories and post-write RTGs;
[trainer.py](elmur_klpo/trainer.py) applies the KLPO or REINFORCE update;
[evaluation.py](elmur_klpo/evaluation.py) measures closed-loop task success.
The exact settings are in [step2_comparison.json](configs/step2_comparison.json)
and [step2_smoke.json](configs/step2_smoke.json).

- Current compact results: [comparison.json](experiments/step2/comparison.json)
  and [smoke.json](experiments/step2/smoke.json). They include the candidate
  interface, execution date, compute summaries, and validation curves.
- Historical slot-averaged results:
  [comparison_slot_averaged.json](experiments/step2/comparison_slot_averaged.json)
  and [smoke_slot_averaged.json](experiments/step2/smoke_slot_averaged.json).
- Full per-update histories, memory-gate measurements, reader weights, and
  manager/optimizer/RNG checkpoints remain in the ignored directories
  `runs/step2_slot_conditioned_comparison/` and
  `runs/step2_slot_conditioned_smoke/`.

```bash
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache uv run python -m elmur_klpo.experiment --config configs/step2_comparison.json --output runs/step2_slot_conditioned_comparison
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache uv run python -m elmur_klpo.experiment --config configs/step2_comparison.json --output runs/step2_slot_conditioned_comparison --resume
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache uv run python -m elmur_klpo.experiment --config configs/step2_smoke.json --output runs/step2_slot_conditioned_smoke
```

Old multi-slot manager checkpoints are incompatible with the expanded
candidate features; use the new output directories above. Further
implementation details are in [step_2_implementation.md](step_2_implementation.md).

## Training-time rerun (2026-10-12)

Reran the supplied three-seed configuration from scratch with the current
slot-conditioned adapter. Each method trained for 100 updates of 32
trajectories per seed (3,200 per seed; 9,600 across three seeds). The reader
was first prepared using LRU behavior cloning, then frozen for manager
training. Reader and managers reused the same 256 training episodes.
Hardware: AMD Ryzen 9 4900HS CPU, one Torch thread.

**Training compute time** below is the measured sum of trajectory rollout
and manager update durations. Updates include post-update KL and finite
diagnostics. These numbers exclude reader preparation, data generation and
batch selection, held-out evaluation, checkpoint I/O, and process startup;
they are not end-to-end wall-clock durations.

| Method | Rollout seconds, all seeds | Update seconds, all seeds | Training compute seconds, all seeds | Mean seconds per seed |
| --- | ---: | ---: | ---: | ---: |
| RTG-REINFORCE | 7.074 | 0.459 | 7.532 | 2.511 |
| RTG-KLPO | 6.965 | 0.448 | 7.413 | 2.471 |

| Seed | REINFORCE training compute seconds | KLPO training compute seconds |
| --- | ---: | ---: |
| 1 | 2.487 | 2.348 |
| 2 | 2.506 | 2.587 |
| 3 | 2.540 | 2.478 |

Shared reader preparation took **17.241 seconds** separately.
All 600 manager updates completed without numerical failures. All final
performance, before-training, and memory-ablation measurements exactly
matched the previous comparison: REINFORCE averaged **69.27%** offline
junction accuracy and **68.75%** closed-loop success; KLPO averaged
**66.67%** and **66.15%**, respectively. Timings are from a single serial
rerun, with REINFORCE preceding KLPO within each seed; small timing
differences do not establish an algorithmic speed advantage.

Exact command:

```bash
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache uv run python -m elmur_klpo.experiment --config configs/step2_comparison.json --output runs/step2_slot_conditioned_timing_rerun
```

The training loop and rollout timer are in [experiment.py](elmur_klpo/experiment.py);
the update timer is in [trainer.py](elmur_klpo/trainer.py). Per-seed timings
and timing definitions are saved in
[timing_rerun.json](experiments/step2/timing_rerun.json). Full per-update
records and checkpoints remain under the ignored
`runs/step2_slot_conditioned_timing_rerun/` directory.
