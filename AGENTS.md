# Repository Guidelines

## Scope and structure

Implemented: Step 1 gradient verification and Step 2 single-reader T-Maze
training. Do not begin Steps 3–4 without an explicit user request.

Within `elmur_klpo/`, keep reusable estimators in `klpo.py`, independent exact
objectives in `exact_enumeration.py`, and diagnostics in `verification.py`.
Step 2 uses `reader.py`, `training_memory.py`, `demonstrations.py`, `rollout.py`,
`trainer.py`, `evaluation.py`, and the `experiment.py` CLI. Preserve upstream
behavior and attribution in `vendor/`; provenance is in `vendor/PROVENANCE.md`.

`tests/` covers mathematical and integration behavior. `configs/` defines
experiment budgets. Consult `README.md`, `step_2_implementation.md`, and
`step_2_results.md` for usage, design decisions, and measured results;
`step_*_plan.md` records development specifications.

## Commands and style

Use uv and pinned Python 3.12. Prefix commands with
`UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache` when the default cache is read-only.

```bash
uv sync
uv run pytest
uv run python scripts/verify_gradients.py
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv build
```

Use four-space indentation, public-function type annotations, and concise
docstrings specifying tensor shapes and gradient boundaries. Follow Ruff;
vendored files are exempt from lint but included in formatting. Use
`snake_case`, `PascalCase`, and `UPPER_SNAKE_CASE` appropriately.

## Mathematical and memory invariants

- Preserve tests A–G, independent exact objectives, and finite differences for
  every parameter. Use fixed seeds and float64; target absolute gradient error
  below `1e-8`. Step 2 training uses float32.
- Keep behavior policies and recorded states fixed. Detach feedback, returns,
  and memory transitions; retain gradients through exact score centering.
  Replay recorded features without rebuilding trajectories.
- Read before writing. Post-write RTG excludes the current reward. Mask padding
  and terminal decisions; the final valid action has no memory decision.
- Preserve candidates as `[N, slots, width]`; never average across slots.
  Flatten all candidates into manager inputs. Selecting slot `k` writes
  `alpha * memory[n,k] + (1-alpha) * candidate[n,k]` only there. No-op preserves
  the bank. Keep the `5*slots+1` action space and coefficients
  `(0, 0.25, 0.5, 0.8, 0.95)`. Native LRU retains its upstream
  blending convention, which weights the new candidate.
- Freeze the prepared reader and encoders during manager training. REINFORCE
  requires fresh on-policy records. The surrogate scalar is not the regression
  objective; off-policy KLPO targets frozen-sampler regression rather than the
  current-policy expected-return gradient.

## Experiments and delivery

Run `python -m elmur_klpo.experiment` through `uv run`, supplying `--config`
and `--output`. Use `configs/step2_smoke.json` before
`configs/step2_comparison.json`; match budgets across at least three manager
seeds. Require the reader memory-dependence gate. Verify `--resume`.

Use fresh output directories for incompatible checkpoints. Keep full histories
and checkpoints under ignored `runs/`; save compact measurements in
`experiments/step2/`. Preserve historical results when interfaces change.
Reports must identify configurations, exact commands, code entry points,
measured performance, ablations, compute, and untested limits.

Keep commits focused and imperative. PRs explain behavior, assumptions,
validation, and numerical results. Preserve unrelated edits; exclude virtual
environments, caches, checkpoints, and build artifacts.
