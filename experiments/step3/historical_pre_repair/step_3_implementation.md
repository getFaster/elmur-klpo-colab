# Step 3 implementation and reproduction

This implements the supplied [Step 3 plan](Step_3_Implementation_Plan.md).
It preserves the Step 1 estimator, exact objectives and finite differences, and
the Step 2 experiment and historical results. Step 2 was committed first as
`5830178`. Step 4 is outside this change.

## Responsibilities

| Module under `elmur_klpo/step3/` | Responsibility |
| --- | --- |
| `tasks.py`, `robot_worker.py` | Native environments, observable features, terminal handling, isolated simulator |
| `curriculum.py` | Difficulty schedules and independent seed blocks |
| `policies.py` | Exact initialization, constrained ablations, continuous fixed-alpha sweeps |
| `reader.py`, `preparation.py` | Original ELMUR task encoders, reader/tok2mem preparation, phase checkpoints |
| `rollout.py` | Native-reward interaction, detached recorded states, post-write discounted RTG |
| `training.py` | Frozen-reader manager training using **Step 2 `trainer.update`** |
| `evaluation.py`, `diagnostics.py` | Closed-loop measurements, bootstrap readiness, coefficient/slot diagnostics |
| `baselines.py` | Independently trained fixed systems with matched reader budgets |
| `artifacts.py`, `aggregation.py` | Atomic stage artifacts and compact measurements |
| `experiment.py` | Stage orchestration and failure reports |

The top-level `elmur_klpo.experiment` dispatches configurations with `step: 3`.
The Step 3 CLI additionally exposes stage, task and reader-source selection.

## Task semantics and verified references

Autoencode uses pinned **POPGym 1.0.7**, the native `AutoencodeEasy` objective,
and the original `step`, observations, termination and reward definition. Its
target is one shuffled 52-card deck: 51 WATCH actions followed by 52 reverse-suit
predictions, for 103 actions. The four task actions are suit IDs. Observations
are a mode and suit, flattened into six one-hot features. The last card remains
visible at the first PLAY decision, as in upstream; it is not hidden by the
adapter. Correct/incorrect PLAY actions earn `+1/52`/`-1/52`. Success is not
invented for an environment that does not define it; reconstruction accuracy is
reported instead. The adapter is tested step-for-step against upstream.

The short-sequence curriculum extension substitutes a prefix of a shuffled
**full deck**, retaining all four suits. It updates deck size and state-space
length; the original environment handles rewards and episode ordering. Lengths
start at two, so older information must still be recalled. Longer robustness
sequences use additional decks through the original `num_decks` interface,
retaining the same reverse-suit objective and observation/action semantics.
The target evaluation distribution remains length 52.

RememberColor uses native **`RememberColor3-v0`**, not the newer VLA task. The
verified MIKASA source commit is
[`509b875f3d207c287497c0a897661062de928bb0`](https://github.com/CognitiveAISystems/MIKASA-Robo/tree/509b875f3d207c287497c0a897661062de928bb0).
Its supported `delta_time` parameter gives curriculum delays 1, 3 and 5;
the original target is 5. It exposes a cue for five steps, hides it during the
delay, then requires continuous robot control. The adapter consumes only the
two native 128×128 RGB cameras, concatenated as `[6,128,128]`, and emits eight
bounded `pd_joint_delta_pos` controls. It excludes oracle and proprioceptive
fields, matching RATE's RGB-only reader path. It uses native normalized dense
reward and reports native success-once. RATE's `InitialZeroActionWrapper`
receives zero initial forced steps from `env_info`; controls pass unchanged.
Native simulation and rendering backends are independently configurable.

Offline demonstration preparation in `configs/step3_remembercolor_reader.json`
uses raw eight-dimensional expert actions without clipping or rescaling and
an unbounded linear output head (`continuous_output: unbounded`). Its stochastic
task policy is an unsquashed Normal with the configured standard deviation.
Other configurations retain the upstream bounded decoder by default. The
historical clipped-target measurements in `step_3_results.md` are unchanged;
new preparation requires a fresh output directory and new measurements.

The pinned RATE source is `8abce2dce9ceacd411d46d4a99c287f7975892f5`.
Its [RememberColor3 script](https://github.com/CognitiveAISystems/RATE/blob/8abce2dce9ceacd411d46d4a99c287f7975892f5/run_experiments/MIKASA_Robo/ELMUR_1/rc3.sh)
uses new-candidate weight **0.40683708068133906**, 256 slots and initial memory
standard deviation 0.1. Its [ELMUR README Autoencode command](https://github.com/CognitiveAISystems/RATE/blob/8abce2dce9ceacd411d46d4a99c287f7975892f5/offline_rl_baselines/ELMUR/README.md)
uses **0.79**, eight slots and zero initial memory standard deviation.
The [paper's rounded coefficients](https://openreview.net/forum?id=bm3rbtEMFj)
are 0.40 and 0.80. Native script references and rounded paper references are
stored separately. **Retention alpha equals `1 - native_lambda`.**

The supplied small architecture uses two slots, one layer, and one-token
windows for both tasks. It deliberately reuses Step 2 rather than reproducing
the paper's larger multi-layer, multi-token architecture. Repeated writes create
competition, but three colored objects do **not** imply three independently
required memory segments. Actual capacity pressure and coefficient sensitivity
on RememberColor remain unverified until its native runtime can be evaluated.

## Training and gradient boundaries

Initial weights are zero except the intercept. Joint write probabilities are
`0.98 / slots * (0.1,0.1,0.6,0.1,0.1)` and no-op probability is 0.02. There is
no task or slot preference. Exact probabilities are saved with reader artifacts.
The same policy prepares the reader and provides the initialization baseline.

Stage B freezes the manager and trains the original ELMUR reader and encoders.
Autoencode preparation uses privileged native suit labels as supervised targets
only; manager inputs never receive them. RememberColor preparation uses
task-action REINFORCE with actual environment reward. With every memory
transition detached, future prediction loss cannot train tok2mem. A local
auxiliary head therefore reconstructs the current detached observation encoding
from **each slot's candidate**, on cue observations. The default auxiliary
weight is 0.1. This head trains original tok2mem without memory BPTT and is not
used for acting. It is an explicit representation-training modification, not a
paper reproduction or a claim that reconstruction guarantees usable memory.

Stage C freezes the complete reader, tok2mem and auxiliary head in evaluation
mode. It uses native environment rewards, Adam at 0.001, clipping at 5, float32,
one update per fresh batch, no critic and no memory BPTT. The shared Step 2 update
rejects stale REINFORCE records. Memory candidates retain `[N,slots,width]`;
manager features flatten every slot in order. Recorded features and returns
are detached. Manager decision `t` receives rewards starting at **`t+1`**;
terminal and padding entries have no management action. There is no KLPO or
alternating optimization in this experiment.

The shared linear manager's intercept supplies its biases. Slot-only has
learnable slot logits and fixed alpha 0.5, with no no-op. Coefficient-only has
one shared coefficient/no-op distribution and **uniform slots**, including
greedy evaluation. Full management learns joint actions. Fixed sweeps draw
uniform slots and apply continuous alpha directly, independently of categorical
IDs. Alpha 1 refreshes the chosen slot's timestamp while preserving contents;
global no-op preserves both contents and timestamps. Native LRU retains its
upstream empty-slot blending implementation, which is reported separately.

## Validation, diagnostics and uncertainty

The readiness gate uses paired held-out **native return** differences, requires
at least 16 episodes, and requires the lower endpoint of a 95% episode-bootstrap
interval to exceed the configured margin. Full task success is not required.
Target exposure is checked from completed curriculum updates. Failed gates are
saved and stop manager training. CLIs exit **2** for failed gates or unavailable
native tasks; this is not a successful training run.

Decodability is diagnostic: it retrieves actual earlier encoder-generated write
snapshots using oracle recall indices, or the last observed robot cue. It never
invents memory vectors. Such valid snapshots can still shift the frozen reader's
memory distribution; a poor score does not uniquely identify the cause.
Retention/replacement robustness uses uniform slots with alpha 0 and 0.95.

Final manager evaluation regenerates complete trajectories, with sampled
management primary and greedy management secondary. Reader task actions are
greedy in evaluation. Coefficient-only greedy evaluation retains uniform slots.
Task seeds use disjoint 100-million-seed blocks for train, validation, test and
robustness. Evaluation preserves Python, NumPy, Torch and CUDA training RNGs.
Manager variants receive matched trajectory budgets and paired test references
for each manager seed.

Artifacts contain native episodic returns, accuracy/success where defined,
coefficient and joint slot frequencies, conditional coefficient distributions,
entropy, no-op and overwrite counts, slot ages, memory-change norms and strata
by timestep, progress, difficulty, age and event. Episode-bootstrap differences
from initialization are exploratory, with no multiplicity correction. Changes
in frequencies are not called successful learning or adaptive behavior.

Primary effects use seed-level t intervals, conditional on the available reader.
The predeclared noninferiority margin is 0.05 native return for both tasks.
System references are selected using validation returns, never final test
returns. A frozen-reader sweep cannot establish system-level H3. Independent
fixed-system readers use matched architectures, curriculum and reader budgets;
their preparation compute is counted separately. Additional reader seeds and
causal memory-content probes remain necessary before broad H3/H4 conclusions.

## Checkpoints and compute

Every reader/manager update writes an atomic checkpoint with optimizer state,
RNG state, exact configuration, history, and frozen components. Reader phase
boundaries additionally save `reader_phase_N.pt`. Final reader checkpoints
include readiness and ablation measurements and can be shared across manager
seeds. Resume checks reader-producing settings, training configuration and the
frozen reader's SHA-256. Interrupted reader and manager runs are tested against
uninterrupted runs, with exact parameter equality on the same device.

Training artifacts live under ignored `runs/`. The aggregation CLI exports
compact measurements to `experiments/step3/`. Timings distinguish reader
preparation, manager rollout and manager update. Rollout includes environment
creation/reset, model inference and recording; native workers currently restart
per rollout batch. Timings exclude subsequent worker shutdown and checkpoint
I/O. Environment interactions, model forward calls, throughput, checkpoint
bytes and CUDA peak allocation are recorded. Dependency download/setup time is
not reader-training compute. This implementation is not optimized for native
robotics process startup or paper-scale throughput.

## Commands

Run all repository commands in the pinned main Python 3.12 environment:

```bash
export UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache
uv sync
uv run pytest
uv run python scripts/verify_gradients.py
uv run ruff check .
uv run ruff format --check .
uv build
```

Provision the separate native simulator only when running RememberColor:

```bash
bash scripts/prepare_step3_robot_runtime.sh runs/step3_native
export ELMUR_NATIVE_PYTHON="$PWD/runs/step3_native/bin/python"
```

This installs a minimal task runtime at the verified MIKASA commit, with
Python 3.11, Torch 2.2.1+cpu, NumPy 1.23.5, Gymnasium 0.29.1,
ManiSkill 3.0.0b15 and setuptools below 81. Other resolved native package
versions from the measured attempt are in `experiments/step3/native_runtime.json`.
The simulator still needs a renderer supporting SAPIEN's required Vulkan
extensions. The observed host cannot initialize that renderer. For a GPU
simulator, configure `task.sim_backend` and `task.render_backend`, and provision
the native environment with a compatible CUDA Torch build. Its dependencies
are deliberately absent from the main project's lockfile.

Run smoke before the larger preparation/comparison:

```bash
uv run python -m elmur_klpo.experiment --config configs/step3_smoke.json --output runs/step3_smoke
uv run python -m elmur_klpo.experiment --config configs/step3_smoke.json --output runs/step3_smoke --resume
uv run python -m elmur_klpo.experiment --config configs/step3_reader_preparation.json --output runs/step3_readers
uv run python -m elmur_klpo.experiment --config configs/step3_reader_preparation.json --output runs/step3_readers --resume
```

For independent task selection, use the Step 3 CLI:

```bash
uv run python -m elmur_klpo.step3.experiment --config configs/step3_reader_preparation.json --output runs/step3_readers --task autoencode
```

After a reader passes its gate, train/resume/evaluate its three-seed managers:

```bash
uv run python -m elmur_klpo.step3.experiment --config configs/step3_manager_training.json --output runs/step3_managers --reader-root runs/step3_readers
uv run python -m elmur_klpo.step3.experiment --config configs/step3_manager_training.json --output runs/step3_managers --reader-root runs/step3_readers --resume
uv run python -m elmur_klpo.step3.experiment --config configs/step3_manager_training.json --output runs/step3_managers --reader-root runs/step3_readers --stage evaluate
uv run python -m elmur_klpo.step3.experiment --config configs/step3_manager_training.json --output runs/step3_managers --reader-root runs/step3_readers --stage robustness
```

Run matched fixed systems independently, or the complete gated comparison:

```bash
uv run python -m elmur_klpo.experiment --config configs/step3_fixed_baselines.json --output runs/step3_fixed
uv run python -m elmur_klpo.experiment --config configs/step3_comparison.json --output runs/step3_comparison
uv run python -m elmur_klpo.experiment --config configs/step3_comparison.json --output runs/step3_comparison --resume
uv run python -m elmur_klpo.step3.experiment --config configs/step3_comparison.json --output runs/step3_comparison --reader-root runs/step3_readers
uv run python -m elmur_klpo.step3.aggregation --inputs runs/step3_comparison --output experiments/step3/comparison.json
```

Use fresh output directories for changed configurations or incompatible
interfaces. The checked-in smoke is integration-only; the full configuration
requests three manager seeds with identical budgets. Current failed gates
intentionally prevent executing those research comparisons.

CUDA reader/manager execution is explicit through `--device cuda`. The project
retains its original CPU wheel default. To use CUDA, provision the main
environment with a compatible CUDA PyTorch build and invoke `uv run --no-sync`
so the CPU lock source does not replace it. CUDA integration tests run when
available and are skipped on the measured CPU-only host.

## Dataset and evaluation details from ELMUR paper

This part is just documentation. Step 3 experiments don't need to match these exactly.


ELMUR Appendix A.8 provides information directly relevant to our Step 3 dataset review:

| Task           | Upstream training data                      | Collection method                       |
| -------------- | ------------------------------------------- | --------------------------------------- |
| RememberColor3 | 1,000 expert demonstrations                 | PPO expert with privileged state access |
| AutoencodeEasy | 3,000 demonstrations per POPGym environment | PPO-GRU expert                          |
| T-Maze         | 6,000 successful trajectories               | Oracle-level policy                     |

