# Step 2 implementation

This stage trains one categorical manager against one frozen ELMUR reader.
The original Step 1 estimator and exact-enumeration tests remain in place.
No population training, critic, counterfactual trajectories, or memory BPTT is
implemented.

| Files changed or added | Responsibility |
| --- | --- |
| `elmur_klpo/memory.py` | Inverse categorical action mapping |
| `elmur_klpo/training_memory.py` | Batched vector memory writes |
| `elmur_klpo/reader.py`, `vendor/` | Adapter and pinned upstream implementation |
| `elmur_klpo/demonstrations.py`, `rollout.py` | Offline data and recorded frozen-reader trajectories |
| `elmur_klpo/trainer.py` | Both policy-gradient losses and checkpointing |
| `elmur_klpo/evaluation.py`, `experiment.py` | Environment evaluation and training CLI |
| `configs/step2_*.json` | Smoke and three-seed comparison budgets |
| `tests/test_step2.py` | Step 2 integration/regression tests, including slot-conditioned candidates |
| `experiments/step2/*.json`, `step_2_results.md` | Measured results and learning curves |
| `pyproject.toml`, `uv.lock`, `.gitignore`, `README.md` | Dependencies, formatting policy, artifacts, usage |

## Repository and upstream decisions

The local repository initially contained only Step 1, with no reader, dataset,
or behavior-cloning infrastructure. RATE provides these components. The pinned
authors' model, observation/action encoders, T-Maze environment, and demonstration
generator are vendored under `elmur_klpo/vendor`; their MIT license and exact
commit are recorded in [PROVENANCE.md](elmur_klpo/vendor/PROVENANCE.md).
Import paths and Ruff formatting are the only changes to those files. Vendored
code is exempt from lint rules, while formatting covers all Python files.

The existing RATE trainer imports broader environment, logging, and online
evaluation infrastructure that this small verification package does not have.
The new CLI uses the authors' model and generator directly with standard
behavior cloning, rather than installing that unrelated infrastructure.

The supported initial task is RATE's T-Maze, using its optimal demonstration
generator. Observations are the original `[y, hint, junction_flag, noise]`;
post-terminal observations are discarded. Both goal directions are balanced.
Corridor lengths vary from 5 to 9 in the supplied configurations. Train,
validation, and test have independent noise seeds and corridor draws; episodes
are never split across datasets. No artificial benchmark is introduced.

The small reader uses one upstream transformer layer, width 32, two slots,
state-only tokens, and context length one. This is a reduced configuration,
not reproduction of the paper's two-layer, context-ten T-Maze experiment.
It uses upstream zero-standard-deviation memory initialization (all null vectors
zero, positions -1), and no dropout. Native LRU uses the authors' T-Maze
`lru_blend_alpha=0.79` and untouched slot/timestamp selection and initialization
behavior. In upstream code this coefficient weights the **new** candidate.
The learned manager's alpha weights **old** memory, as specified in Step 2.

Upstream tok2mem produces `[N, slots, width]` slot-conditioned candidates.
The adapter preserves that tensor, detached, for all management methods.
Selecting slot `k` and old-memory coefficient `alpha` writes
`memory[n,k] = alpha * memory[n,k] + (1-alpha) * candidate[n,k]`;
other slots stay unchanged, and no-op preserves the entire bank. KLPO,
REINFORCE, random, and native LRU use the same upstream candidate definition;
native LRU retains its own selection and blending conventions. No new memory
encoder is introduced. Frozen candidate outputs depend on the current memory,
so caching them by observation would be incorrect; they are recomputed during
each batched forward pass.

## Reader preparation and temporal semantics

With no supplied compatible checkpoint, prepare the reader with 500 LRU
behavior-cloning updates, learning rate 0.002, batch size 32. Each window's
memory is detached before the next window, following upstream semantics.
The existing representation is prepared with the reader, then the entire model
is frozen and placed in evaluation mode. Writer-only parameters do not receive
future reader gradients through memory transitions. Manager losses cannot
update any reader or encoder parameter.

Before manager training, validation must show at least 0.02 lower junction NLL
with genuine memory and junction accuracy above 60%. The ablation feeds zero
vectors to the reader at every window; it also removes memory conditioning of
its write candidates. Positional metadata is retained. The CLI writes the
measurements and stops if the gate fails.

Reading precedes writing. The final valid expert action yields a reward but
no memory decision. For each recorded decision, `G[t] = sum(r[t+1:])` within
its episode row. Padding and terminal decisions are excluded. Manager input
contains an intercept, observation encoding, flattened memory, all candidates
flattened in slot order, slot ages divided by 100, and time divided by 100.
For `S` slots and width `D`, its dimension is `(2*S+1)*D + S + 2`.
Rollout and replay use the same recorded feature tensor; replay never rebuilds
memory with a new policy.

## Manager optimization and checkpoints

The configurable Step 1 linear categorical manager is reused. Its joint action
space has five old-memory coefficients per slot plus a global no-op. Vector
writes use gather/scatter and are detached, out of place, and loop-free over
slots. The action-space mapping now has an explicit inverse.

Each fresh rollout stores features, categorical IDs, full behavior log
probabilities and probabilities, selected log probabilities, post-write RTGs,
valid decision masks, and episode IDs. Only learned-policy records can enter
the optimizer. LRU action IDs are slot-count instrumentation, not a claim that
its continuous coefficient belongs to the categorical alphabet; baseline
probability fields are placeholders and never used for learning.

KLPO calls the gradient-verified `backward_surrogate`; exact behavior-weighted
centering stays differentiable and the entire feedback is detached. Invalid
timesteps have zero feedback and ratio. REINFORCE is a separate loss and rejects
stale off-policy records. Both average episode sums, use Adam at 0.001, clip
gradient norm to 5, and perform one update per fresh batch. There is no critic,
reward clipping, return normalization, importance weighting, or added baseline.
Finite checks reject invalid policy outputs, returns, gradients, and parameters.

With a fresh snapshot and exactly one update, p=q when the gradient is evaluated,
so the KLPO beta ratio contribution is zero. The supplied beta=0.1 does not
actively constrain that first step; the measured difference from REINFORCE is
score centering and resulting sample variance, not an established regularization
benefit. Away from p=q the estimator targets frozen-sampler regression, not the
current manager's expected-return gradient. Both KL directions are measured
after each update on the recorded states.

Atomic iteration checkpoints contain manager and optimizer states, iteration,
configuration, reader state, metrics, and Python/NumPy/Torch/CUDA RNG states.
`--resume` restores completed iterations and reevaluates completed runs. Reader
preparation saves its completed checkpoint; an interrupted preparation starts
again. Matching configuration is required. See the exact restoration regression
test.

Manager checkpoints made with the previous slot-averaged candidate interface
have a different input dimension when there is more than one slot and must be
retrained in a fresh output directory. Frozen reader checkpoints remain
compatible.

## Running and inspecting results

```bash
uv sync
uv run pytest
uv run python scripts/verify_gradients.py
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run python -m elmur_klpo.experiment --config configs/step2_smoke.json --output runs/step2_slot_conditioned_smoke
uv run python -m elmur_klpo.experiment --config configs/step2_comparison.json --output runs/step2_slot_conditioned_comparison
uv run python -m elmur_klpo.experiment --config configs/step2_comparison.json --output runs/step2_slot_conditioned_comparison --resume
```

Prefix with `UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache` if the default cache is
read-only. `results.json` contains every update, validation learning curves,
before/after and memory-ablation metrics, alpha/slot/no-op frequencies, memory
change norms and unchanged-bank fractions, both KL directions, gradient norms,
rollout/update times, throughput, and peak GPU allocation. `memory_dependence.json`
records the mandatory reader gate. Closed-loop evaluation uses the original
environment, greedy task actions, and sampled manager actions, on the held-out
goals/corridor lengths with independent noise.

The locked Linux/Windows environment uses CPU PyTorch. `--device cuda` is
implemented, but requires an environment with a compatible CUDA PyTorch build;
use `uv run --no-sync` to preserve such an installation. GPU behavior and
mixed precision were not validated here. Float32 is the canonical training
precision; numerical gradient tests remain float64. No mixed precision is enabled.

Measured results, compute limits, and acceptance status are in
[step_2_results.md](step_2_results.md). Checkpoints and full per-update logs remain
under ignored `runs/`; compact measured artifacts are under `experiments/step2/`.
