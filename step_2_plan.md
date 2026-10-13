# Implementation Task: ELMUR-KLPO — Single Reader–Manager Training

## Project context

We are developing **ELMUR-KLPO**, a memory-management method inspired by ELMUR (External Long-term Memory with Update and Retrieval).

The objective is to learn how an agent manages a fixed-capacity external memory bank using **Return-to-Go KL-Regularized Policy Optimization (RTG-KLPO)**, without a learned critic or backpropagation through memory-management trajectories.

The planned development sequence is:

1. **Gradient correctness verification** — Implement an exactly enumerable toy environment and verify RTG-KLPO gradients against mathematical ground truth. (Done)
2. **Single reader–manager training** — Integrate RTG-KLPO into a memory-aware imitation-learning setup.
3. **Multiple reader–manager pairs** — Investigate population-based learning and cross-training to reduce reader–manager co-adaptation.
4. **Long-horizon evaluation** — Benchmark memory preservation, capacity competition, delayed credit assignment, and performance against ELMUR's LRU baseline.

**Implement Step 2 only.** Do not implement the later steps. However, organize the code so that the trained manager can be reused by the later training system.


## Objective

The memory manager learns which memory slot to update and how much existing information to preserve. Its training signal is return-to-go (RTG) computed from a frozen reader's expert-action log-likelihood.

Use RTG-KLPO with exact categorical score centering.

This task covers **Development Step 2 only**: one reader, one memory manager, and controlled comparisons with LRU, random management, and RTG-REINFORCE.

Development Step 1 (gradient correctness) has already been completed. Locate and reuse its implementation and tests. Do not repeat that project unnecessarily.

Deliver working, tested code and runnable experiments, not just an implementation proposal.

## 1. Repository inspection

Before modifying code:

1. Inspect the repository structure, existing ELMUR architecture, datasets, memory representation, rollout pipeline, tests, and experiment configuration.
2. Locate the completed Step 1 gradient-validation implementation.
3. Identify the reader, tok2mem encoder, memory update mechanism, LRU manager, and behavior-cloning training code.
4. Reuse existing implementations wherever possible.
5. Record repository-specific decisions in the implementation documentation.

Preserve existing ELMUR functionality. Add the new manager as a configurable alternative.

Do not assume particular module paths, hidden dimensions, memory capacities, or frameworks before inspecting the codebase.

Do not replace established training infrastructure without a concrete reason.

## 2. Definitions

### 2.1 Reader

The reader is a policy network

\[
\pi_\phi(a_t\mid c_t,M_t)
\]

where:

- \(c_t\): current reader context or encoded observation/history.
- \(M_t\in\mathbb R^{B\times d}\): external memory bank.
- \(B\): number of memory slots.
- \(d\): memory-vector dimension.
- \(a_t\): task action.
- \(\phi\): reader parameters.

Given an expert demonstration action \(a_t^*\), define the imitation reward

\[
r_t=\log\pi_\phi(a_t^*\mid c_t,M_t).
\]

This reward is an imitation-learning signal, not necessarily an environment reward.

### 2.2 Memory encoder

The tok2mem encoder produces a candidate memory vector

\[
z_t=f_\psi(c_t).
\]

Initially freeze \(\psi\). Reuse ELMUR's existing representation mechanism; do not introduce a newly trained memory encoder.

### 2.3 Memory manager

The trainable manager is a categorical policy

\[
p_\theta(u_t\mid x_t)
\]

with

\[
x_t=(c_t,M_t,z_t).
\]

Include any additional context already required by the existing ELMUR implementation. Ensure the manager input is constructed consistently during rollout and replay.

The manager samples a memory-management action \(u_t\).

### 2.4 Action space

Let

\[
\mathcal A_\alpha=\{0,0.25,0.5,0.8,0.95\}.
\]

For each memory slot \(k\in\{0,\ldots,B-1\}\), provide five update actions:

\[
u=(k,\alpha),\quad \alpha\in\mathcal A_\alpha.
\]

Add one global no-op action corresponding to \(\alpha=1\), without selecting a slot.

Total number of actions:

\[
\boxed{|\mathcal U|=5B+1.}
\]

Use a single categorical distribution over these actions.

The memory update is

\[
M_{t+1}^{(k)}
=
\alpha M_t^{(k)}+(1-\alpha)z_t
\]

for the selected slot, with other slots unchanged.

For no-op:

\[
M_{t+1}=M_t.
\]

Initialize memory as

\[
M_0=[m_\nu]^B
\]

using the existing ELMUR null-memory initialization.

Use a deterministic, tested mapping between categorical action IDs and `(slot, alpha)`.

Maintain exact log-probabilities for sampled actions.

### 2.5 Behavior manager and learner

During each training iteration, define:

\[
q(u\mid x)=p_{\bar\theta}(u\mid x)
\]

as the frozen behavior manager used to generate trajectories.

The current trainable manager is \(p_\theta\).

Store the behavior distribution or enough information to reconstruct it exactly for every sampled decision.

The ratio term is

\[
\ell_t
=
\log p_\theta(u_t\mid x_t)
-
\log q(u_t\mid x_t).
\]

Never replace the historical behavior distribution with the updated policy when evaluating recorded trajectories.

## 3. Reader preparation

The reader must already be capable of using external memory.

Prefer loading a compatible, trained ELMUR reader checkpoint. If unavailable, implement a reproducible reader-pretraining procedure using the repository's existing imitation-learning pipeline and LRU memory manager.

Reader training must use meaningful memory-dependent supervision.

Before manager optimization, compare the reader with its genuine memory input against a memory-ablated or shuffled-memory condition on a memory-sensitive validation task.

Report whether memory actually improves imitation performance. Do not silently accept a reader that ignores memory.

Freeze reader parameters during memory-manager training.

Initially also freeze the memory encoder.

Do not backpropagate reader losses through memory-bank history into the manager. The manager learns through its categorical policy gradient.

## 4. Rollout and reward generation

Use offline expert demonstrations initially.

For each demonstration:

1. Initialize the external memory bank.
2. Run the existing context/encoding pipeline.
3. Evaluate the frozen reader at every decision point.
4. Obtain candidate memory vectors from the frozen tok2mem encoder.
5. Sample the manager action from frozen behavior policy \(q\).
6. Update the memory bank using that action.
7. Continue the trajectory and collect imitation rewards.

Preserve the repository's correct temporal ordering.

If the reader acts before memory writing at timestep \(t\), then manager action \(u_t\) must not receive credit for \(r_t\), which it cannot affect.

Under that ordering:

\[
G_t=\sum_{j=t+1}^{T-1}r_j.
\]

If the architecture writes before reading, use

\[
G_t=\sum_{j=t}^{T-1}r_j.
\]

Document the ordering and add a regression test.

For each manager decision, retain:

- Recorded manager input \(x_t\).
- Sampled categorical action \(u_t\).
- Behavior log-probability \(\log q(u_t\mid x_t)\).
- Full behavior action probabilities \(q(\cdot\mid x_t)\).
- RTG \(G_t\).
- Valid timestep mask.
- Demonstration/episode identifier.

Use detached rollout states during manager optimization.

Never reconstruct historical memory states with the updated manager.

Variable-length episodes and terminal boundaries must be handled correctly. Return calculations must not cross episode boundaries.

## 5. RTG-KLPO objective

Use the exact finite-action version of KLPO.

For each recorded decision, define

\[
\ell_t=\log p_\theta(u_t\mid x_t)-\log q(u_t\mid x_t)
\]

and

\[
h_t=G_t-\beta\ell_t
\]

where \(\beta>0\) is the KL regularization coefficient.

The exact conditional score-centering term is

\[
C_\theta(x_t)
=
\sum_{u\in\mathcal U}
q(u\mid x_t)\log p_\theta(u\mid x_t).
\]

Implement the gradient surrogate

\[
\boxed{
\mathcal L_{\mathrm{KLPO}}
=
-\mathbb E_{\tau\sim q}
\sum_t
\operatorname{sg}(h_t)
\left[
\log p_\theta(u_t\mid x_t)
-
C_\theta(x_t)
\right].
}
\]

Implementation requirements:

- Apply stop-gradient to the entire \(h_t\).
- Do NOT detach \(C_\theta(x_t)\).
- Weight \(C_\theta\) with behavior probabilities \(q\), not current probabilities \(p_\theta\).
- Compute the current policy's log-probabilities for all \(5B+1\) actions.
- Use stable `log_softmax`.
- Calculate exact score centering; no auxiliary Monte Carlo actions.
- Compute loss over valid decision timesteps only.
- Normalize consistently with the chosen timestep/episode weighting.
- Keep gradients confined to current manager parameters.
- Support configurable \(\beta\).
- Use fresh behavior rollouts for each manager update cycle initially.

Important theoretical limitation:

This surrogate estimates the gradient of the frozen-sampler KLPO regression objective. Away from \(p_\theta=q\), it is not generally an unbiased estimator of the current manager's true expected-return gradient.

Therefore, do not label its optimization loss as the exact expected-return objective.

## 6. RTG-REINFORCE baseline

Implement a separate on-policy manager trainer.

For freshly sampled manager trajectories, use

\[
\boxed{
\mathcal L_{\mathrm{RF}}
=
-\mathbb E_{\tau\sim p_\theta}
\sum_t
\operatorname{sg}(G_t)
\log p_\theta(u_t\mid x_t).
}
\]

Implement without a learned critic or learned value baseline.

Use the same action space, manager architecture, reward, rollout budget, and optimizer configuration wherever applicable.

For fair comparison, clearly distinguish algorithm-specific regularization settings.

RTG-REINFORCE is an essential baseline because, with on-policy rollouts, its expected gradient is the true policy gradient of the fixed-reader imitation-return objective.

## 7. LRU and random baselines

Implement or reuse:

**LRU baseline**

Use ELMUR's established LRU slot-selection rule and its original fixed blending coefficient for the selected task.

Preserve original preprocessing, capacity, and update timing.

No learned manager.

**Random baseline**

Choose valid memory-management actions randomly.

Use a documented distribution. Include a uniform-over-actions baseline initially.

All methods must use:

- Identical expert demonstration splits.
- Identical memory capacity.
- Identical frozen reader and encoder.
- Identical observation/context preprocessing.
- Identical evaluation metrics.

Do not allow differences in reader training to confound the initial manager comparison.

## 8. Optimization and implementation constraints

Target efficient execution on an NVIDIA RTX 2060 with 6 GB VRAM, or Google Colab.

Requirements:

1. Batch reader inference across compatible trajectories.
2. Vectorize categorical action evaluation, exact centering, and RTG calculations.
3. Avoid unnecessary CPU/GPU synchronization.
4. Avoid per-slot Python loops for memory updates when vectorized gather/scatter is available.
5. Do not enumerate alternative memory futures.
6. Do not evaluate counterfactual reader trajectories.
7. Do not implement BPTT through memory updates.
8. Cache or recompute frozen encoder outputs based on measured memory/compute tradeoffs.
9. Support mixed precision only where numerically safe; preserve stable policy log-probabilities.
10. Make training resumable and deterministic under controlled seeds as far as the backend permits.

Use one manager optimizer update per fresh rollout batch initially. Make rollout reuse configurable only if needed after this baseline is validated.

Add gradient clipping and finite-value checks.

Avoid arbitrary reward clipping, return normalization, or additional advantage baselines in the canonical implementation. If such techniques are later tested, keep them separately configurable and clearly labeled.

## 9. Evaluation tasks

Select an existing ELMUR-compatible task that genuinely requires memory. Prefer a repository-supported partially observable task with delayed cues and memory competition.

Use the same task for the first comparison.

Do not create an artificial benchmark if a suitable validated task already exists.

Use a small smoke-test configuration followed by a controlled multi-seed experiment.

Choose evaluation horizons and memory capacity that make the LRU baseline meaningful. If the task is solvable without memory, reject it as the primary evaluation.

Measure:

- Held-out expert-action negative log-likelihood.
- Held-out expert-action accuracy, if applicable.
- Total imitation return.
- Environment success/return if a compatible closed-loop evaluation exists.
- Manager action distribution.
- Frequency of each alpha value.
- Per-slot update frequency.
- No-op frequency.
- Memory retention statistics.
- KL divergence between current and behavior manager.
- Gradient norms and numerical failures.
- Rollout time, update time, throughput, and peak GPU memory.

Log both

\[
D_{\mathrm{KL}}(q\Vert p_\theta)
\]

and

\[
D_{\mathrm{KL}}(p_\theta\Vert q)
\]

when feasible.

Do not infer success from training loss alone.

## 10. Experiments

### Smoke test

Verify that:

- Reader and memory manager can execute a full trajectory.
- Memory updates match the specified categorical action.
- RTG is computed correctly.
- Only manager parameters are updated.
- Behavior probabilities remain frozen within an iteration.
- Exact score-centering gradients are nonzero when expected.
- No-op does not modify memory.
- Training produces no NaNs or invalid categorical probabilities.
- Evaluation and checkpoint restoration work.

### Primary comparison

Train and evaluate:

| Method | Reader | Manager |
|---|---|---|
| ELMUR-LRU | Frozen | Fixed LRU |
| ELMUR-Random | Frozen | Uniform random |
| ELMUR-REINFORCE | Frozen | Learned categorical |
| ELMUR-KLPO | Frozen | Learned categorical |

Use at least three independent seeds for the main experiment, subject to practical compute limits.

Match manager trajectory budgets between REINFORCE and KLPO.

Compare methods using final held-out performance, learning curves, and actual compute—not just optimizer step counts.

### Essential diagnostic ablations

For the learned KLPO manager:

1. Compare performance before and after training.
2. Evaluate the learned action distribution versus forced no-op.
3. Evaluate memory-ablated reader performance.
4. Inspect whether the manager learns nontrivial slot selection.
5. Report alpha utilization rather than assuming every coefficient is useful.

Do not expand into population training, learned critics, adaptive merging networks, or BCST counterfactual supervision.

## 11. Testing

Reuse all applicable completed Step 1 gradient tests.

Add tests for:

- Action ID ↔ `(slot, alpha)` mappings.
- The \(5B+1\) action-space invariant.
- Correct no-op behavior.
- Exact convex update semantics.
- Memory initialization and slot indexing.
- RTG alignment with write/read ordering.
- Terminal masking and variable-length batches.
- Exact categorical score centering.
- Correct stop-gradient behavior.
- Frozen reader and frozen encoder.
- Frozen behavior policy within each batch.
- No accidental gradient through recorded memory states.
- Checkpoint save/resume consistency.
- Batched versus reference implementations.

For a tiny categorical example, compare the implemented KLPO update gradient against the already verified Step 1 reference.

Use numerical gradient comparisons with explicit tolerances.

## 12. Deliverables

Implement:

1. Configurable categorical memory-manager module.
2. Memory-update and no-op handling.
3. Frozen-reader rollout collector.
4. Correct imitation-reward/RTG construction.
5. RTG-KLPO trainer with exact categorical centering.
6. RTG-REINFORCE trainer.
7. LRU and random baseline integration.
8. Training/evaluation CLI entrypoints using the repository's conventions.
9. Experiment configurations.
10. Automated tests.
11. Checkpointing and reproducible seeding.
12. Concise implementation documentation.
13. A short experiment report with measured results.

Use existing project conventions for file organization, logging, configuration, and testing.

Do not fabricate performance results or mark unexecuted tests as passed.

## 13. Acceptance criteria

The task is complete when:

- All new correctness tests pass.
- Existing tests remain passing.
- A memory-aware frozen reader is available and its memory dependence is measured.
- RTG-KLPO successfully updates a categorical manager.
- The sampled memory trajectories produce valid imitation rewards.
- KLPO and REINFORCE can be evaluated under matched rollout budgets.
- LRU and random baselines are runnable.
- At least one full controlled experiment runs successfully.
- Logs expose learned merge/slot behavior and computational cost.
- All results and limitations are documented.
- Reproduction commands are provided.

If the main experiment is too expensive to finish within the available runtime, complete the implementation and smoke tests, save reproducible experiment configurations, and report exactly which experiments remain unexecuted.

## 14. Scope restrictions

Do not implement:

- Multiple readers or multiple managers.
- Reader-manager population cross-training.
- A learned value function or critic.
- BCST counterfactual targets.
- Continuous merge coefficients.
- Trainable matrix-valued memory merges.
- Backpropagation through memory history.
- A replacement RL algorithm such as PPO or SAC.
- Unrequested architectural rewrites.

Maintain a clear separation between Step 2 and future Step 3 population training.

## 15. Execution instructions

Work autonomously through repository inspection, implementation, testing, and available experiments.

Resolve ordinary implementation decisions using existing project conventions.

Prioritize mathematical correctness, reproducibility, and compatibility over speculative optimization.

Before finishing, review all changed code for:

- Incorrect RTG timing.
- Leaked gradients.
- Old/current policy confusion.
- Incorrect categorical probability weighting.
- Reader or encoder accidentally being trained.
- Policy-state reconstruction errors.
- Misleading comparisons or inconsistent sample budgets.

Run the relevant test suite and available smoke experiments.

Report at completion:

1. Files changed.
2. Implemented functionality.
3. Tests executed and their results.
4. Experiment commands and actual measurements.
5. Remaining limitations or blockers.
6. Whether the Step 2 acceptance criteria have been met.
