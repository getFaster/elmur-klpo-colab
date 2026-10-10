# Step 1 verification results

Verified on 2026-10-10 with Python 3.12.13, CPU PyTorch 2.14.1, pytest 9.1.1,
and Ruff 0.16.10. All gradient experiments use float64. The experiment has two
memory slots, 11 actions, three effective manager decisions, and 1,331 complete
trajectories. Each gradient comparison covers every trainable parameter.

Commands executed:

```bash
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache uv run pytest
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache uv run python scripts/verify_gradients.py
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/elmur-klpo-uv-cache uv run ruff format --check .
```

All **68 tests pass**. Ruff lint and formatting checks pass. There are no failed
or inconclusive mathematical tests. A PyTorch import warning reports that
optional NumPy support is unavailable; all calculations use PyTorch tensors and
require no NumPy functionality.

| Check | Maximum absolute error / diagnostic | Relative error | Result |
| --- | ---: | ---: | --- |
| A: surrogate vs. explicit regression gradient | 2.359223927328e-15 | 4.518702636577e-15 | PASS |
| B: on-policy KLPO ascent vs. true return gradient | 3.330669073875e-15 | 6.954925479775e-15 | PASS |
| C: off-policy discrepancy (infinity norm) | 1.583745546561e-01 | — | EXPECTED DIFFERENCE |
| D: RTG vs. full-return population gradients | 3.663735981263e-15 | 5.436362550592e-15 | PASS |
| E: exact centering, including independent analytic scores | 1.908195823574e-17 | — | PASS |
| E: incorrect detached-center gradient discrepancy | 6.377944999090e-01 | — | Detected |
| F: incorrect differentiable-feedback gradient discrepancy | 9.433696468821e-02 | — | Detected |
| G: isolated early-action gradient L2 norm | 2.127162774663e-01 | — | PASS |
| All-parameter regression finite differences | 3.386440433628e-11 | 1.496325065150e-10 | PASS |

Absolute error means the infinity norm of the gradient difference. Relative
error is the L2 difference divided by the larger gradient L2 norm, with a
`1e-15` floor. For repeated comparisons each reported column is the maximum
over cases, potentially from different cases. The absolute tolerance is `1e-8`.

Tests A and D use seeds `(7, 19, 41)` crossed with beta `(0.1, 0.7, 2.0)`.
Learners use scale `0.35`; behavior policies use seed `seed+100` and scale `0.2`.
Test B uses the same learner seeds, frozen identical behavior snapshots, and
beta `0.7`.

Tests C/E/F and finite differences use an explicit uniform behavior policy and a
learner whose action-indexed coefficient columns are:

- Intercept: 11 points from `-0.45` to `0.55`.
- First memory slot: 11 points from `0.6` to `-0.4`.
- Second memory slot: 11 points from `-0.3` to `0.5`.
- Time and candidate columns: zero.

Beta is `0.7`. The true expected-return gradient has L2 norm
`8.585769077996e-01`, confirming this counterexample is nondegenerate. The
central finite-difference step is `1e-5`, applied separately to all 55 parameters.
Test G uses a uniform on-policy manager and independently checks its early
logit gradient against `q(u) * A^q(x_0,u)`. Saving the cue has positive credit;
discarding it has negative credit. All preterminal rewards are identical.

Additional checks independently reconstruct conditional Q-values from complete
trajectory weights, compare return gradients against analytic softmax score
vectors, verify sampling/log-probability consistency, and exercise action IDs,
memory updates, detach boundaries, reward timing, and terminal-action exclusion.

No mathematical discrepancies were found relative to the supplied definitions.
The off-policy difference in C is the stated theoretical property, and the
surrogate is never treated as the original scalar regression loss.

The estimator is verified for this finite experiment and is modular enough to
reuse in Step 2 under the documented fixed-behavior and detached-state
assumptions. This does not establish training convergence or results on ELMUR
benchmarks. Work stops at Step 1 pending review of these results.
