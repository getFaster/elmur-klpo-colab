# Codex Implementation Task: ELMUR-KLPO — Step 1: Gradient Correctness Verification

## Project context

We are developing **ELMUR-KLPO**, a memory-management method inspired by ELMUR (External Long-term Memory with Update and Retrieval).

The objective is to learn how an agent manages a fixed-capacity external memory bank using **Return-to-Go KL-Regularized Policy Optimization (RTG-KLPO)**, without a learned critic or backpropagation through memory-management trajectories.

The planned development sequence is:

1. **Gradient correctness verification** — Implement an exactly enumerable toy environment and verify RTG-KLPO gradients against mathematical ground truth.
2. **Single reader–manager training** — Integrate RTG-KLPO into a memory-aware imitation-learning setup.
3. **Multiple reader–manager pairs** — Investigate population-based learning and cross-training to reduce reader–manager co-adaptation.
4. **Long-horizon evaluation** — Benchmark memory preservation, capacity competition, delayed credit assignment, and performance against ELMUR's LRU baseline.

**Implement Step 1 only.** Do not implement the later steps. However, organize the code so that the tested gradient estimator can be reused by the later training system.

Do not assume any existing ELMUR or KLPO implementation is available.

---

# 1. Method definitions

## 1.1 Memory bank

The memory manager maintains a fixed-capacity bank

\[
M_t=[m_{t,1},\ldots,m_{t,B}].
\]

At each timestep \(t\), it receives an observation/context \(c_t\) and a proposed memory representation

\[
z_t=f(c_t).
\]

For the toy environment, use scalar or low-dimensional representations and a fixed, deterministic \(f\).

The initial memory is

\[
M_0=[m_\nu]^B.
\]

For this test, use \(m_\nu=0\).

## 1.2 Manager actions

The stochastic manager is

\[
p_\theta(u_t\mid x_t),\qquad x_t=(c_t,M_t),
\]

where \(u_t\) specifies a slot and a merge coefficient.

Allowed coefficients:

\[
\mathcal A_\alpha=\{0,0.25,0.5,0.8,0.95,1\}.
\]

For a selected slot \(k\),

\[
m_{t+1,k}
=
\alpha m_{t,k}+(1-\alpha)z_t.
\]

Other slots remain unchanged.

Interpretation:

- \(\alpha=0\): overwrite.
- \(\alpha=0.25\): mostly new content.
- \(\alpha=0.5\): equal merge.
- \(\alpha=0.8\): conservative merge.
- \(\alpha=0.95\): very conservative merge.
- \(\alpha=1\): no-op.

Since \(\alpha=1\) makes the selected slot irrelevant, represent it as one **global no-op** action.

Thus, the action space has

\[
|\mathcal U|=5B+1
\]

distinct actions.

For the initial toy experiment, use \(B=2\), giving 11 actions.

Use a single categorical policy over these 11 actions, parameterized by a small differentiable neural network or a tabular/linear softmax model.

The policy must support:

- Computing every action probability.
- Computing sampled-action log-probabilities.
- Evaluating action probabilities at arbitrary recorded states.
- Exact summation over the entire action space.

Use PyTorch and float64 for gradient-verification experiments.

## 1.3 Reader and imitation reward

The reader is a frozen policy

\[
\pi_\phi(a_t^*\mid c_t,M_t)
\]

that predicts an expert action \(a_t^*\).

Its per-step imitation reward is

\[
r_t=\log\pi_\phi(a_t^*\mid c_t,M_t).
\]

The reader should be a small deterministic-parameter model whose likelihood explicitly depends on memory.

For Step 1, it does not need training.

The manager writes after the current reader decision. Therefore:

1. The reader evaluates the current memory \(M_t\).
2. The resulting reward \(r_t\) is recorded.
3. The manager chooses \(u_t\).
4. The memory transitions to \(M_{t+1}\).

Under this convention, action \(u_t\) first affects reward \(r_{t+1}\).

Therefore, define

\[
G_t=\sum_{j=t+1}^{T}r_j.
\]

Use undiscounted RTG (\(\gamma=1\)).

The terminal action, if any, must be excluded when it cannot affect future rewards.

## 1.4 Behavior manager and learner

Define:

\[
q(u\mid x)=p_{\bar\theta}(u\mid x)
\]

as the frozen behavior policy used to generate trajectories.

Define

\[
p_\theta(u\mid x)
\]

as the current trainable manager.

During each gradient-verification experiment:

- \(q\) is fixed.
- The reader is fixed.
- The encoder is fixed.
- The environment is fixed.
- Recorded states are detached from \(\theta\).
- Gradients flow only through current-policy action log-probabilities.

The behavior and learner policies must both assign positive probability to every action.

---

# 2. RTG-KLPO gradient

For a recorded state-action pair \((x_t,u_t)\), define

\[
\ell_t
=
\log p_\theta(u_t\mid x_t)
-
\log q(u_t\mid x_t).
\]

Define the feedback coefficient

\[
h_t=G_t-\beta\ell_t,
\]

where \(\beta>0\) is the KL regularization coefficient.

The exact score-centering term is

\[
C_\theta(x_t)
=
\sum_{v\in\mathcal U}
q(v\mid x_t)\log p_\theta(v\mid x_t).
\]

The sample backward surrogate is

\[
\boxed{
\widehat L_{\mathrm{bp}}(\theta)
=
-\sum_t
\operatorname{sg}(h_t)
\left[
\log p_\theta(u_t\mid x_t)-C_\theta(x_t)
\right].
}
\]

Here, \(\operatorname{sg}\) means stop-gradient.

**Important:** This is a gradient surrogate, not the original scalar KLPO regression objective. Its scalar value should not be compared directly with that objective.

The intended population gradient is

\[
\boxed{
g_{\mathrm{KLPO}}(\theta;q)
=
-\mathbb E_{\tau\sim q}
\sum_t
(G_t-\beta\ell_t)
\left[
\nabla_\theta\log p_\theta(u_t\mid x_t)
-
\sum_vq(v\mid x_t)\nabla_\theta\log p_\theta(v\mid x_t)
\right].
}
\]

This is the gradient of the KLPO regression loss (a descent direction), not generally the gradient of current-policy expected return.

Implementation requirements:

- Detach \(h_t\) when constructing the backward surrogate.
- Do not detach the current-policy log-probabilities inside \(C_\theta\).
- Treat \(q\)'s probabilities as constants.
- Do not use sampled auxiliary actions: calculate \(C_\theta\) exactly.
- Do not backpropagate through memory transitions.

---

# 3. Ground-truth objectives

Implement two distinct mathematical targets.

## 3.1 Expected imitation return

For a manager policy \(p_\theta\), define

\[
J(\theta)
=
\mathbb E_{\tau\sim p_\theta}
\left[
\sum_t r_t
\right].
\]

In our toy environment, all action sequences must be enumerable.

Compute \(J(\theta)\) exactly by enumerating complete action sequences and weighting their returns by their trajectory probabilities.

Use autograd to calculate

\[
\nabla_\theta J(\theta).
\]

Do not detach policy probabilities in this calculation.

This is the exact on-policy expected-return gradient.

## 3.2 KLPO regression objective

For each recorded state \(x_t\), define

\[
Q^q(x_t,u)
=
\mathbb E_q[G_t\mid x_t,u_t=u]
\]

and

\[
V^q(x_t)
=
\sum_uq(u\mid x_t)Q^q(x_t,u).
\]

Then

\[
A^q(x_t,u)=Q^q(x_t,u)-V^q(x_t).
\]

For this finite deterministic toy system, calculate \(Q^q\) by enumerating future manager actions under \(q\).

Define

\[
k_\theta(x)
=
D_{\mathrm{KL}}\big(q(\cdot\mid x)\Vert p_\theta(\cdot\mid x)\big).
\]

The explicit KLPO regression objective is

\[
\boxed{
L_{\mathrm{reg}}(\theta;q)
=
\frac{1}{2\beta}
\mathbb E_{\tau\sim q}
\sum_t
\left[
A^q(x_t,u_t)
-\beta\left(
\ell_\theta(x_t,u_t)+k_\theta(x_t)
\right)
\right]^2.
}
\]

Calculate its exact gradient using autograd.

When differentiating this objective, treat \(Q^q,V^q,A^q\), and \(q\)'s state-visitation distribution as fixed.

If representing the underlying environment as partially observable, use the complete pre-action toy history as the conditioning variable for exact \(Q^q\), or design the toy so that \(x_t\) is sufficient. Avoid incorrectly equating distinct histories that happen to produce the same \(x_t\).

---

# 4. Required mathematical tests

## Test A: KLPO gradient equivalence

Set \(p_\theta\neq q\).

Calculate the exact expected gradient of the RTG-KLPO backward surrogate by enumerating every trajectory under \(q\).

Compare with

\[
\nabla_\theta L_{\mathrm{reg}}(\theta;q).
\]

Verify:

\[
\boxed{
\mathbb E_{\tau\sim q}
[\nabla_\theta\widehat L_{\mathrm{bp}}]
=
\nabla_\theta L_{\mathrm{reg}}.
}
\]

This is the most important test.

Repeat for multiple policy parameter values and multiple \(\beta\) values.

## Test B: Equality with the true policy gradient at \(p=q\)

Set

\[
p_\theta=q.
\]

Then \(\ell_t=0\), and exact score centering should yield

\[
\boxed{
-\mathbb E_{\tau\sim q}
[\nabla_\theta\widehat L_{\mathrm{bp}}]
=
\nabla_\theta J(\theta)
\quad\text{at }p_\theta=q.
}
\]

Be careful with the sign: the KLPO backward surrogate is minimized, while expected return is maximized.

Test equality over every trainable parameter.

## Test C: Difference away from the sampler

Choose \(p_\theta\neq q\) with nonzero expected-return gradient.

Compare

\[
-\nabla_\theta L_{\mathrm{reg}}(\theta;q)
\]

with

\[
\nabla_\theta J(\theta).
\]

They should generally differ.

This is an expected theoretical property, not a bug.

The test should explicitly demonstrate a case where the mismatch is nonzero.

Do not assert that a mismatch must occur for every parameter configuration.

## Test D: RTG versus full episodic return

For undiscounted episodic trajectories, test replacing each \(G_t\) with full trajectory return

\[
R=\sum_jr_j.
\]

Verify that their **population KLPO gradients** are equal, assuming rewards before action \(u_t\) cannot depend on that action.

Do not require per-trajectory sample gradients to be identical.

This test checks the action-independent-baseline identity.

## Test E: Score-centering correctness

Verify numerically that

\[
\sum_uq(u\mid x)
\left[
\nabla_\theta\log p_\theta(u\mid x)
-\sum_vq(v\mid x)\nabla_\theta\log p_\theta(v\mid x)
\right]
=0.
\]

Compare exact score centering with an independently computed sum of score vectors.

Also test that detaching the centering term produces an incorrect off-policy gradient in a nondegenerate example.

## Test F: Stop-gradient correctness

Verify that removing stop-gradient from

\[
h_t=G_t-\beta\ell_t
\]

changes the backward gradient.

The test should fail if the implementation accidentally allows differentiation through \(h_t\).

## Test G: Delayed credit assignment

Construct a toy sequence where:

- A useful cue arrives at an early timestep.
- Later writes compete for the same limited memory capacity.
- The final reader reward depends on preservation of the early cue.

Verify that the exact on-policy gradient assigns nonzero credit to early manager decisions.

The memory manager must not receive a direct immediate reward for preserving the cue.

The signal should arise from future reader performance.

Do not require that a randomly initialized manager immediately learns the optimal behavior. This step verifies gradient correctness, not training convergence.

---

# 5. Toy environment design

Implement at least one minimal deterministic environment with the following properties:

- Memory capacity \(B=2\).
- Scalar or low-dimensional memory vectors.
- Horizon short enough for exact trajectory enumeration.
- At least one early cue that is needed later.
- At least one distracting memory candidate.
- A frozen reader with a known memory-dependent prediction rule.
- Manager actions alter future imitation rewards.

The complete action space contains 11 actions.

Avoid unnecessarily long horizons because naive trajectory enumeration scales as

\[
|\mathcal U|^{T_{\mathrm{decisions}}}.
\]

Prefer 2–4 effective manager decisions for the exact-enumeration tests.

Keep the environment deterministic initially so that all stochasticity comes from the manager.

Optionally add a small stochastic-reward test if deterministic tests pass. This is not a prerequisite for completion.

Use a stable, explicit mapping from categorical action IDs to \((k,\alpha)\).

---

# 6. Implementation structure

Create a self-contained test package organized approximately as follows:

```text
elmur_klpo/
    __init__.py
    manager.py
    memory.py
    toy_env.py
    returns.py
    klpo.py
    exact_enumeration.py

tests/
    test_action_space.py
    test_memory_updates.py
    test_rtg.py
    test_klpo_gradient.py
    test_exact_policy_gradient.py
    test_delayed_credit.py

scripts/
    verify_gradients.py
```

Adapt to existing repository conventions if needed.

Responsibilities:

**manager.py**
- Categorical memory-manager policy.
- Action enumeration and log-probability evaluation.
- Exact joint action probability calculation.

**memory.py**
- Immutable or out-of-place memory transition.
- No-op behavior.
- Slot update logic.

**toy_env.py**
- Deterministic expert trajectory.
- Fixed encoding.
- Fixed memory-aware reader.
- Reward computation.

**returns.py**
- Correct post-write RTG.
- Support masks if needed.

**klpo.py**
- RTG-KLPO backward surrogate.
- Exact score centering.
- Appropriate detach operations.

**exact_enumeration.py**
- Enumerate trajectories.
- Calculate exact expected returns.
- Calculate exact \(Q^q\) and \(A^q\).
- Calculate the explicit KLPO regression objective.
- Calculate population-expected estimator gradients.

**verify_gradients.py**
- Run all gradient comparisons.
- Report numerical errors.

Do not introduce a full RL training framework or third-party RL dependency.

PyTorch and pytest should be sufficient.

---

# 7. Numerical verification

Use double precision.

For gradient comparisons, report

\[
\epsilon_{\mathrm{abs}}
=
\|g_1-g_2\|_\infty
\]

and

\[
\epsilon_{\mathrm{rel}}
=
\frac{\|g_1-g_2\|_2}
{\max(\|g_1\|_2,\|g_2\|_2,\varepsilon)}.
\]

For exact-enumeration comparisons, aim for an absolute error below \(10^{-8}\).

Use appropriate absolute tolerances near zero instead of relying exclusively on relative error.

Test several fixed random seeds.

Also perform at least one finite-difference gradient check against the explicitly constructed KLPO regression objective.

---

# 8. Required diagnostics

The verification script should print a compact report containing:

```text
ELMUR-KLPO Step 1: Gradient Verification

Action space: ...
Memory slots: ...
Effective manager decisions: ...
Enumerated trajectories: ...

Test A: KLPO regression gradient identity
  Absolute error: ...
  Relative error: ...
  PASS/FAIL

Test B: On-policy equality at p=q
  Absolute error: ...
  Relative error: ...
  PASS/FAIL

Test C: Off-policy difference
  Gradient discrepancy: ...
  EXPECTED DIFFERENCE / FAIL

Test D: RTG vs full return
  Absolute error: ...
  PASS/FAIL

Test E: Exact score centering
  Error: ...
  PASS/FAIL

Test F: Stop-gradient
  Gradient difference: ...
  PASS/FAIL

Test G: Delayed credit assignment
  Early-action gradient norm: ...
  PASS/FAIL

Overall: PASS/FAIL
```

Include concrete parameter values and reproducible seeds in the report or accompanying test logs.

A test that asserts a nonzero difference must use an explicitly verified nondegenerate example rather than an arbitrary random initialization that could accidentally produce zero.

---

# 9. Constraints

Do not:

- Train a critic or value function.
- Implement PPO, SAC, or DQN.
- Implement BCST counterfactual training.
- Add learned memory compression.
- Introduce continuous merge coefficients.
- Implement multiple readers or multiple managers yet.
- Backpropagate through memory transitions when computing the KLPO backward surrogate.
- Approximate score centering with Monte Carlo samples.
- Claim that off-policy KLPO estimates the true current-policy return gradient.
- Optimize the toy environment for performance before verifying the mathematics.

Do:

- Keep the code modular.
- Preserve exact action definitions.
- Use explicit policy distributions.
- Separate the KLPO regression objective from its backward surrogate.
- Verify gradient signs.
- Test both \(p=q\) and \(p\neq q\).
- Keep rollout states fixed when evaluating KLPO gradients.
- Document any assumptions required by the derivation.
- Prefer simple, auditable implementations over optimization.

---

# 10. Completion criteria

Step 1 is complete only when:

1. All prescribed gradient tests pass.
2. The KLPO backward-surrogate gradient matches the exact KLPO regression gradient away from \(p=q\).
3. At \(p=q\), the KLPO ascent direction matches the exact expected imitation-return gradient.
4. A nondegenerate example demonstrates that off-policy KLPO and the true current-policy gradient are generally different.
5. RTG handling and reward-action timing are verified.
6. Exact score centering and stop-gradient behavior are tested.
7. The delayed-credit toy example produces a nonzero expected early-action learning signal.
8. All tests can be executed with a single documented command.
9. The implementation does not depend on ELMUR's full architecture or an RL training library.

## Final deliverable

Implement the code and tests, execute the verification suite, and provide:

- The files created or modified.
- The exact test command.
- Numerical gradient-comparison results.
- Any failed or inconclusive tests.
- Any mathematical discrepancies discovered during implementation.
- A short statement indicating whether the KLPO estimator is verified and reusable for Step 2.

**Stop after Step 1. Do not begin single reader–manager training until the verification results have been reviewed.**
