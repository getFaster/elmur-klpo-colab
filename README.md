# ELMUR-KLPO: Steps 1–3

An exactly enumerable, float64 PyTorch experiment verifies the categorical
RTG-KLPO memory-manager gradient from [step_1_plan.md](step_1_plan.md).
Step 2 adds single-reader manager training with the authors' ELMUR model and
T-Maze demonstrations. See [implementation and commands](step_2_implementation.md)
and [measured three-seed results](step_2_results.md).

Step 3's repair implements execution items 1–3 of [step_3_fix.md](step_3_fix.md):
pinned upstream task-specific readers, fixed-LRU imitation preparation, and
independent deterministic parity/resume smoke tests. Full reader preparation
has **not** run. Learned-manager integration and comparisons remain gated.
See the [new implementation report](step_3_implementation.md) for commands,
configuration differences, write/gradient audits, and unfinished work. The
[previous implementation](step_3_implementation_legacy.md) and
[historical results](step_3_results.md) remain available unchanged.

The [ELMUR authors' implementation](https://github.com/CognitiveAISystems/RATE/tree/main/offline_rl_baselines/ELMUR)
is the architectural reference. Pinned MIT-licensed components are vendored for
Step 2; the Step 1 mathematical experiment remains independent of the reader.

## Run

Install [uv](https://docs.astral.sh/uv/) and run from this directory:

```bash
uv run pytest
```

This single command installs the locked environment and runs all prescribed
tests, including A–G and finite differences for every parameter. Python 3.12 is
pinned in `.python-version`; uv can provision it if needed. Linux and Windows
use CUDA 12.8 PyTorch wheels from the official PyTorch index. GPU execution
requires a compatible NVIDIA driver and GPU access; CPU execution remains
available. Verify the installed build and GPU availability with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Step 2 also uses NumPy, Gymnasium, and tqdm for the authors'
T-Maze generator; pytest and Ruff are development dependencies.
Step 3 also installs pinned POPGym/Gymnasium. MIKASA runs in a separately
provisioned interpreter because its upstream dependencies conflict with the
main runtime. The adapter always uses the native robot task.

Print the numerical verification report:

```bash
uv run python scripts/verify_gradients.py
```

The script exits with status 1 if any check fails, and reports the worst absolute
and relative errors over the seed/beta grid. The recorded run is in
[verification_report.txt](verification_report.txt), with interpretation in
[verification_results.md](verification_results.md).

Format and check:

```bash
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
```

In a restricted environment where the default uv cache is read-only, prefix
commands with a writable cache directory. The exact test command used here was:

```bash
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache uv run pytest
```

## Toy experiment

The two-slot scalar memory starts at `[0, 0]`. Three manager decisions see
candidates `+1` (useful cue), `-1`, and `-0.5` (distractors). Encoding is the
identity. At each step the frozen reader acts **before** the memory write.
All expert labels are 1; before the final query the reader assigns probability
`1/2` to label 1 regardless of memory. At the final query it assigns
`sigmoid(4 * (sum(memory) - 0.5))`. Rewards are log likelihoods.

Only the final reward depends on memory. Preserving the early cue therefore
receives credit through future reader performance. Later writes can overwrite
it while competing for the same two slots. There is no terminal manager action.
The complete set is `11**3 = 1,331` trajectories.

Action IDs are stable and slot-major:

| IDs | Slot | Alpha values, in order |
| --- | --- | --- |
| 0–4 | 0 | 0, 0.25, 0.5, 0.8, 0.95 |
| 5–9 | 1 | 0, 0.25, 0.5, 0.8, 0.95 |
| 10 | none | 1 (global no-op) |

Each write sets `m[k] = alpha*m[k] + (1-alpha)*candidate`. Memory updates are
detached and out of place. The manager is a linear softmax over joint actions,
with features `[1, time/3, candidate, m[0], m[1]]` and 55 trainable parameters.
Time and memory are a sufficient state for this deterministic environment.
Enumeration also preserves each action prefix separately, including histories
that happen to have identical memory.

## Gradient targets and assumptions

`exact_expected_return` enumerates complete trajectories with **differentiable
learner trajectory probabilities**. It computes the ordinary scalar expected
return `J` and its ascent gradient. Rewards and memory paths for each fixed action
sequence are constants; all dependence on the manager comes through probabilities.

`regression_objective` computes the ordinary scalar `L_reg` from exact conditional
`Q^q`, `V^q`, and `A^q`. A backward sum over the full prefix tree independently
enumerates every possible continuation under frozen behavior `q`. State
visitation, behavior probabilities, and conditional values stay fixed while
differentiating `L_reg`.

`backward_surrogate` implements the plan's stop-gradient loss. It detaches feedback
`h = G - beta*(log_p - log_q)` and behavior probabilities, while retaining the
learner gradient inside exact score centering. `post_write_rtg` assigns
`G_t = r[t+1] + ... + r[H]`; current and earlier rewards are excluded. The
surrogate's scalar value is not the regression objective, and finite differences
are applied only to the explicit regression objective.

For a fixed state, write `s(u) = grad log p(u)` and
`s_c(u) = s(u) - E_q[s]`. Then `E_q[s_c] = 0`, and
`grad(ell + KL(q || p)) = s_c`. Differentiating the regression objective yields
`-E_q[(A - beta*(ell + KL)) * s_c]`. The difference from the surrogate coefficient
`Q - beta*ell` is a state-only baseline `V + beta*KL`, which vanishes under
centering. Conditional expectation replaces sampled `G` with exact `Q`.

This identity requires fixed `q`, fixed rollout states, fixed reader/encoder,
finite positive action probabilities, exact action summation, and `beta > 0`.
It does not imply equality with the current-policy return gradient away from
`p=q`. At `p=q`, the negative surrogate gradient equals the expected-return
gradient. Full episodic return and post-write RTG produce the same **population**
gradient because rewards through time `t` are action-independent baselines at
the pre-action state. Individual sample gradients can differ.

For later reuse, the core estimator accepts arbitrary recorded categorical
log-probability arrays shaped `[..., H, U]`, with actions and returns `[..., H]`.
It returns one summed loss per trajectory, leaving batch averaging to the caller.
Callers must pass current-policy log probabilities evaluated at detached states
and freeze behavior policies for each experiment. No importance ratios or
Monte Carlo auxiliary actions are used.

## Files created

```text
.gitignore
.python-version
pyproject.toml                  # uv package, dev tools, CPU PyTorch index
uv.lock                         # reproducible dependency resolution
README.md
verification_results.md        # numerical results and interpretation
verification_report.txt        # saved diagnostic output
elmur_klpo/
    __init__.py
    manager.py                  # linear joint categorical policy and snapshot
    memory.py                   # action mapping and detached memory writes
    toy_env.py                  # frozen reader, candidates, sufficient states
    returns.py                  # read-before-write return timing
    klpo.py                     # reusable stop-gradient backward surrogate
    exact_enumeration.py        # trajectories, exact Q, J, and L_reg
    verification.py             # numerical diagnostics and incorrect controls
tests/
    conftest.py
    test_action_space.py
    test_memory_updates.py
    test_rtg.py
    test_klpo_gradient.py
    test_exact_policy_gradient.py
    test_delayed_credit.py
scripts/
    verify_gradients.py
```

The supplied `step_1_plan.md` is unchanged. Tests E and F deliberately construct
incorrect losses inside verification utilities to prove sensitivity to detaching
the centering term or differentiating feedback. These controls are separate from
the reusable estimator. Step 2 modules, tests, and configurations are documented
in [step_2_implementation.md](step_2_implementation.md).
