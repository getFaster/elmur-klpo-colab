# Step 3 Implementation Plan: Eliminating Task-Dependent Memory Blending Coefficients

## 0. Objective

Implement and evaluate a learned ELMUR memory manager that replaces the manually tuned, task-dependent LRU blending coefficient \(\lambda\).

**Primary research question:**

Can a memory manager trained from downstream task rewards discover effective memory-writing coefficients without manually selecting \(\lambda\) for each task?

**Secondary research question:**

Does the manager learn state-dependent blending, or does it simply converge to an effective fixed coefficient?

Use at least two tasks with different memory requirements. Start with:

1. **MIKASA-Robo RememberColor3-v0**, under memory-capacity pressure.
2. **POPGym AutoencodeEasy**.

The ELMUR paper reports LRU blend coefficients of approximately 0.40 for RememberColor and 0.80 for AutoencodeEasy. Verify the exact task configurations against the upstream implementation before using them as reference baselines.

Do not assume these reported values are mathematically optimal.

Reuse the existing Step 1 policy-gradient implementation and Step 2 ELMUR reader adapter where appropriate.

Implement the experiment in independently testable stages. Preserve existing Step 1 and Step 2 results and tests.

---

## 1. Mathematical Definitions

### 1.1 Reader and memory

Let:

- \(o_t\): observation at timestep \(t\).
- \(h_t\): reader context or recurrent state, if applicable.
- \(M_t\in\mathbb R^{S\times D}\): external memory bank.
- \(S\): number of memory slots.
- \(D\): memory-vector dimension.
- \(\pi_\phi\): ELMUR reader with parameters \(\phi\).
- \(g_\phi\): ELMUR tok2mem candidate generator.
- \(\mu_\theta\): trainable categorical memory manager with parameters \(\theta\).

The reader produces task-action predictions:

\[
\pi_\phi(a_t^{\mathrm{task}}\mid o_t,h_t,M_t).
\]

The upstream tok2mem module produces slot-conditioned write candidates:

\[
Z_t=g_\phi(o_t,h_t,M_t)
\in\mathbb R^{S\times D}.
\]

Preserve the upstream candidate interface.

Do not average candidates across slots.

### 1.2 Memory-management actions

The manager chooses a memory slot and an old-memory retention coefficient:

\[
u_t=(k_t,\alpha_t)
\]

where

\[
k_t\in\{0,\ldots,S-1\}
\]

and

\[
\alpha_t\in\mathcal A
=\{0,0.25,0.5,0.8,0.95\}.
\]

Include an additional global no-op action.

The memory update is:

\[
M_{t+1}[k_t]
=
\alpha_tM_t[k_t]
+
(1-\alpha_t)Z_t[k_t].
\]

All other slots remain unchanged.

For no-op:

\[
M_{t+1}=M_t.
\]

**Coefficient convention:**

- \(\alpha=0\): completely replace old memory.
- \(\alpha=0.5\): equal blending.
- \(\alpha=0.95\): strongly preserve old memory.
- \(\alpha=1\): preserve old memory entirely, equivalent to no-op for the selected slot.

The upstream ELMUR LRU coefficient \(\lambda\) weights the new candidate:

\[
M_{t+1}[k]
=
(1-\lambda)M_t[k]+\lambda Z_t[k].
\]

Therefore:

\[
\boxed{\alpha=1-\lambda}
\]

Never compare \(\alpha\) against upstream \(\lambda\) without converting conventions.

### 1.3 Manager state

Define the manager's input as:

\[
s_t^{\mathrm{mgr}}
=
\left(
e(o_t),\,
M_t,\,
Z_t,\,
\operatorname{age}_t,\,
t
\right).
\]

Use the existing Step 2 manager feature construction initially.

Flatten slot-conditioned memory and candidate tensors while preserving slot order.

Do not include future observations, future rewards, ground-truth task labels unavailable at decision time, or any other privileged information.

The manager outputs:

\[
\mu_\theta(u_t\mid s_t^{\mathrm{mgr}}).
\]

---

## 2. Overall Training Procedure

Implement three stages:

### Stage A — Initialize the manager

Construct a stochastic manager with uniform slot-selection probabilities and a blending distribution biased toward \(\alpha=0.5\).

### Stage B — Freeze the manager and prepare the reader

Train the reader and tok2mem using a memory-dependent curriculum.

The manager must remain frozen throughout reader preparation.

### Stage C — Freeze the reader and train the manager

Train the manager using fresh trajectories and downstream task rewards.

The reader and tok2mem must remain frozen throughout manager training.

The overall procedure is:

\[
\mu_{\theta_0}
\xrightarrow{\mathrm{freeze}}
\operatorname{Train}(\phi)
\xrightarrow{\mathrm{freeze}\ \phi}
\operatorname{Train}(\theta).
\]

Do not implement alternating reader-manager optimization in the primary experiment.

---

## 3. Stage A: Manager Initialization

### 3.1 Initial categorical distribution

Conditional on performing a write, initialize:

\[
P(k)=\frac1S.
\]

For each slot, initialize the same coefficient distribution:

| Retention coefficient | Initial probability |
|---|---:|
| \(\alpha=0\) | 0.10 |
| \(\alpha=0.25\) | 0.10 |
| \(\alpha=0.5\) | 0.60 |
| \(\alpha=0.8\) | 0.10 |
| \(\alpha=0.95\) | 0.10 |

Initialize:

\[
P(\mathrm{no-op})=0.02.
\]

Scale all write-action probabilities by 0.98.

Thus:

\[
P(k,\alpha)=
\frac{0.98}{S}P(\alpha\mid k).
\]

Set the linear categorical manager's feature weights to zero and its biases to the logarithms of these probabilities, up to an additive constant.

For a two-slot memory bank:

\[
P(k=0,\alpha=0.5)=0.294
\]

and similarly for slot 1.

Verify initialization with unit tests and empirical sampling.

### 3.2 Important constraints

- All slots must have identical initial selection probabilities.
- No slot should receive a learned or hand-coded preference.
- The initial distribution must not depend on task identity.
- Use identical initialization rules across tasks.
- Do not initialize manager parameters from native LRU.
- Store the exact initial probability distribution in the experiment metadata.

Use the same initial manager distribution for reader preparation and the frozen-initial-manager baseline.

---

## 4. Stage B: Reader Preparation with Curriculum

### 4.1 Trainable parameters

Train:

- ELMUR reader parameters.
- Original tok2mem parameters.
- Other reader-associated components required by the upstream implementation.

Freeze:

- Entire categorical memory manager.

The manager continues sampling memory-management actions, but its parameters are not updated.

No gradients should be propagated through sampled management decisions.

Preserve upstream temporal ordering and memory semantics. Unless upstream task integration requires otherwise:

1. Read the current memory.
2. Predict the task action.
3. Produce slot-conditioned memory candidates.
4. Sample a memory-management action.
5. Apply the memory update.
6. Advance to the next timestep.

Initialize memory using the upstream empty-memory convention, not with memories produced by LRU.

### 4.2 Curriculum definition

A curriculum is an ordered or adaptive sequence of training-task distributions with increasing difficulty.

Its purpose is to train a reader that understands memory while initially reducing the difficulty of memory management.

**Easy must not mean memory-independent.**

For example, in a cue-recall task:

- Easy: remember a cue across a short delay with few distractors.
- Medium: remember the same cue across a longer delay.
- Hard: remember the cue across distractors and competing memory-writing opportunities.

At every level, correctly solving the task must still require information that is no longer directly observable.

For each selected environment, inspect the upstream task implementation and identify supported difficulty parameters.

Prefer varying existing task parameters, such as:

- Recall delay.
- Distractor frequency.
- Number of objects or cues.
- Memory competition.
- Sequence length.

Do not silently invent a replacement benchmark.

If the environment does not expose suitable curriculum parameters, implement a minimal environment wrapper or configuration extension that preserves its original objective and observation/action semantics.

Document the modification.

### 4.3 Curriculum for RememberColor3-v0

Inspect the environment and upstream ELMUR training configuration.

Use the original color-recall objective.

Create an easier distribution by reducing available difficulty parameters while preserving the requirement to remember the color information.

Gradually approach the original benchmark distribution.

The final curriculum phase must include the target evaluation difficulty.

Maintain memory-capacity pressure in the final task configuration.

Use the paper's capacity-sensitivity results to guide configuration selection, but verify the actual slot capacity and required information segments rather than assuming that object count directly equals memory requirement.

### 4.4 Curriculum for AutoencodeEasy

Inspect the actual POPGym AutoencodeEasy environment.

Preserve the original sequence reconstruction or recall objective and its reward definition.

Identify supported sequence-length or difficulty controls.

Start with shorter sequences that still require memory, then increase toward the original task distribution.

Do not allow the curriculum to collapse into a purely reactive prediction problem.

### 4.5 Reader readiness checks

Reader preparation must not require high full-task success.

Instead, establish the following:

**Check A: Memory dependence**

Evaluate the reader with genuine memory and with memory content ablated.

Define:

\[
\Delta_{\mathrm{mem}}
=
J_{\mathrm{real}}
-
J_{\mathrm{ablated}}.
\]

Use the environment's actual reward or success metric.

Require a positive memory contribution supported by held-out evaluation.

**Check B: Memory decodability**

Determine whether the reader can use informative memory states to make correct task predictions.

When feasible, construct valid memory trajectories from actual observations and the original tok2mem encoder.

Do not inject arbitrary synthetic memory vectors that were never produced by the encoder.

**Check C: Target-distribution exposure**

Verify that the reader has been trained and evaluated on the target task distribution, even if full-task performance remains poor.

**Check D: Robustness to management variation**

Evaluate the frozen reader under:

- Its original initialization manager.
- A fixed coefficient favoring retention.
- A fixed coefficient favoring replacement.

Use the same slot-selection policy when comparing coefficients.

This is a diagnostic, not a requirement that all alternative policies achieve high success.

### 4.6 Reader checkpoint

Save:

- Reader and tok2mem parameters.
- Frozen initialization manager.
- Curriculum schedule.
- Training and validation curves.
- Memory-ablation measurements.
- Environment configurations.
- RNG states.
- Dataset or environment seeds.

The checkpoint must be reusable for multiple independent manager-training seeds.

---

## 5. Stage C: Manager Training

### 5.1 Frozen components

Freeze the reader and tok2mem completely.

Set them to evaluation mode.

Verify:

\[
\nabla_\theta J\neq0
\]

where learning signal exists, but

\[
\nabla_\phi J=0.
\]

Memory transitions must not propagate reader gradients through earlier memory states.

Only the memory manager is trainable.

### 5.2 Primary optimization algorithm

Use the existing Step 2 RTG-REINFORCE implementation.

Do not begin with KLPO or counterfactual training.

For manager actions:

\[
u_t\sim\mu_\theta(\cdot\mid s_t^{\mathrm{mgr}}).
\]

Define the return-to-go:

\[
G_t=\sum_{j=t+1}^{T-1}\gamma^{j-t-1}r_j.
\]

Here \(r_j\) denotes the downstream reward associated with task performance, not an artificial reward for selecting a particular coefficient.

For imitation-learning rewards, use the existing Step 2 expert-action log-probability reward if the task is being trained from demonstrations.

For native reinforcement-learning tasks, use their actual environment reward.

Document which reward is used for each task.

Do not silently substitute behavior-cloning likelihood for environment reward when the benchmark evaluates closed-loop task performance.

The REINFORCE estimator is:

\[
\widehat{\nabla_\theta J}
=
\frac1B
\sum_{b=1}^{B}
\sum_t
G_{b,t}
\nabla_\theta
\log\mu_\theta(u_{b,t}\mid s_{b,t}^{\mathrm{mgr}}).
\]

Respect the existing Step 2 post-write reward and RTG indexing convention.

Use fresh trajectories for each update.

### 5.3 Training implementation

Reuse the existing:

- Categorical manager.
- Joint slot/coefficient action mapping.
- Batched memory-writing implementation.
- Rollout recording.
- REINFORCE optimizer.
- Gradient checks.
- Checkpointing and resume support.

Extend them only as necessary to support the selected environments and longer episodes.

Start with the optimizer settings from Step 2:

- Adam learning rate: 0.001.
- Gradient norm clipping: 5.
- One gradient update per fresh rollout batch.
- Float32 training.
- No critic.
- No memory BPTT.

Treat these as initial settings, not automatically optimal settings.

If instability requires modifications, record them and apply matched changes to relevant comparisons.

Support CUDA PyTorch explicitly and test it where available.

### 5.4 Evaluation

Evaluate using full trajectories generated by the current manager.

Do not evaluate manager quality solely by replaying states recorded under a previous manager.

At each evaluation timestep, the manager must interact with the memory bank resulting from its own previous decisions.

Use held-out environment seeds.

Report both:

- Sampled management actions.
- Greedy management actions.

Use sampled actions as the primary evaluation convention, consistent with Step 2.

Do not tune hyperparameters using the final test set.

---

## 6. Experimental Baselines

### 6.1 Primary manager-learning comparison

Use the exact same reader checkpoint for:

| Baseline | Slot selection | Coefficient selection |
|---|---|---|
| Initial manager | Uniform | Initial distribution biased toward 0.5 |
| Learned manager | Learned | Learned |
| Fixed-0.5 manager | Uniform | Fixed 0.5 |
| Fixed-coefficient sweep | Uniform | Fixed coefficients |
| Learned-slot-only | Learned | Fixed 0.5 |
| Learned-coefficient-only | Uniform | Learned |

The learned-coefficient-only manager must share one coefficient distribution across slots unless explicitly testing slot-conditioned coefficients.

For the learned-slot-only manager, keep the coefficient fixed at 0.5 throughout training and evaluation.

When comparing learned coefficients against fixed coefficients, hold the slot-selection mechanism constant.

When comparing learned slots against uniform selection, hold the coefficient mechanism constant.

This prevents attributing improvements from slot selection to coefficient learning.

### 6.2 Fixed-coefficient sweep

Evaluate:

\[
\alpha\in\{0,0.1,0.2,0.5,0.6,0.8,0.95,1\}.
\]

This grid includes the old-memory retention equivalents of the relevant paper coefficients:

- RememberColor: \(\lambda=0.40\Rightarrow\alpha=0.60\).
- AutoencodeEasy: \(\lambda=0.80\Rightarrow\alpha=0.20\).

Use an independent fixed-coefficient evaluation mechanism; do not require these values to belong to the learned categorical action space.

No-op and \(\alpha=1\) may produce the same memory contents but can differ in metadata handling. Preserve and document timestamp and slot-age semantics.

A frozen-reader coefficient sweep is diagnostic because the reader was trained under the initialization manager.

It is not a substitute for a fully trained fixed-coefficient reference system.

### 6.3 System-level baselines

Train independent readers using:

1. The original native LRU configuration.
2. The paper's reported fixed coefficient.
3. A fixed coefficient of 0.5.

Keep reader architecture, tok2mem architecture, environment distribution, reader-training budget, and evaluation procedure matched wherever possible.

The learned-manager system uses the Stage B reader.

Do not use an LRU-trained reader as the sole reference for judging a learned manager.

Distinguish two evaluation categories:

**Frozen-reader comparison:** isolates manager optimization under a common reader.

**System-level comparison:** evaluates the complete reader–memory-management training procedure.

Do not conflate these conclusions.

---

## 7. Experiment Matrix and Compute Budget

Implement the following experiment progression.

### Phase 1 — Integration smoke test

For each task:

- Run the environment adapter.
- Verify observation and action shapes.
- Verify slot-conditioned candidate shapes.
- Verify correct initialization.
- Execute a short reader-training run.
- Execute a short manager-training run.
- Test checkpoint restoration.
- Evaluate one complete held-out episode.

Use minimal episode counts and training updates.

Smoke tests must not be interpreted as evidence of algorithm quality.

### Phase 2 — Reader preparation

For each task:

- Prepare one reader using the frozen initialization manager.
- Save intermediate and final checkpoints.
- Evaluate memory dependence and decodability.
- Verify target-task exposure.

Use an initial reader-preparation seed for integration.

Add at least two further reader seeds for the final study if resources permit.

### Phase 3 — Manager learning

For each task:

- Train the full learned manager.
- Train learned-slot-only.
- Train learned-coefficient-only.
- Evaluate the frozen initialization manager.
- Evaluate fixed-coefficient baselines.

Use at least three independent manager-training seeds for initial results.

Prefer five for final reporting.

All variants must receive matched trajectory budgets.

### Phase 4 — Fixed-system references

Train the necessary fixed-coefficient reader systems using matched reader-training budgets.

Report training compute and environment interactions separately from manager training.

Do not claim that a method is cheaper merely because it excludes reader-preparation cost.

### Phase 5 — Robustness

Evaluate:

- Longer retention horizons than those seen during training.
- Different levels of memory competition.
- Different initial manager random seeds.
- Multiple reader checkpoints when available.

Keep task identity and benchmark semantics unchanged.

---

## 8. Required Diagnostics

### 8.1 Task performance

Record:

- Mean episodic return.
- Success rate, if defined by the environment.
- Task-specific accuracy or reconstruction error.
- Learning curves.
- Mean and standard deviation across seeds.
- Confidence intervals for primary comparisons.

### 8.2 Coefficient selection

Record the marginal distribution:

\[
P(\alpha=a).
\]

Record the slot-conditioned distribution:

\[
P(\alpha=a\mid k).
\]

Record coefficient frequencies against:

- Timestep.
- Episode progress.
- Memory age.
- Task difficulty.
- Observed event type, where event labels can be computed for analysis.

Measure coefficient entropy and no-op frequency.

Determine whether the learned coefficient distribution differs significantly from initialization.

Do not equate a change in coefficient distribution with successful learning unless performance also improves.

### 8.3 Slot behavior

Record:

- Slot-selection frequencies.
- Slot ages.
- Number of overwrites per episode.
- Number of no-op decisions.
- Memory-change norms.
- Unchanged-memory fractions.

### 8.4 Retention and replacement behavior

Where the environment provides sufficient ground truth, identify:

- Important information that remains relevant.
- Obsolete information that can safely be replaced.
- Distractors that do not need to be retained.

These labels are for diagnostics, not privileged manager inputs.

For example, if an initial cue must be recalled at the final timestep, label it as task-relevant until that decision. If a later cue explicitly supersedes it, label the earlier cue obsolete from that point onward.

Measure whether the manager preserves task-relevant memory and updates obsolete content.

Inspect joint slot/coefficient decisions rather than only marginal coefficient distributions.

### 8.5 Compute

Record:

- Reader-training time.
- Manager rollout time.
- Manager update time.
- Environment steps per second.
- GPU memory allocation.
- Reader forward-pass counts.
- Checkpoint size.
- Total training environment interactions.

Keep all timing measurements attributable to the relevant stage.

---

## 9. Hypothesis Tests and Acceptance Criteria

### H1 — Learning improves management

The learned manager achieves higher held-out task performance than its frozen initialization policy.

Evaluate across both selected tasks.

### H2 — Coefficient learning contributes independently

Under a fixed slot-selection mechanism, the learned coefficient policy outperforms fixed-0.5 management.

If learned-slot-only explains nearly all the improvement, do not attribute success to adaptive blending.

### H3 — Manual coefficient tuning can be replaced

The fully learned manager matches or exceeds a strong task-tuned fixed-coefficient reference within statistical uncertainty.

Report effect sizes and confidence intervals.

Do not interpret a non-significant difference as proof of equivalence. Define a practical non-inferiority margin using the task metric before final evaluation.

### H4 — The manager learns adaptive behavior

The policy's coefficient and slot decisions respond appropriately to the current memory-management situation.

For example, it might preserve a relevant memory during a long delay but overwrite obsolete information after a cue replacement.

A manager converging to a single useful coefficient can satisfy H1–H3 but does not establish H4.

### H5 — Results generalize across tasks

Demonstrate H1–H3 on at least two task environments.

Each task may have its own trained manager for this experiment.

Training one shared manager across tasks without task-specific retraining is a separate transfer-learning experiment and is not required for Step 3.

---

## 10. Failure Analysis

If manager training fails, diagnose the following possibilities separately.

### Failure A — Reader ignores memory

Evidence: little difference between genuine-memory and memory-ablated performance.

Action: revise reader curriculum to require memory-dependent predictions.

### Failure B — Reader cannot decode useful memory

Evidence: performance remains poor even when informative memory states are supplied through valid encoder-generated trajectories.

Action: improve reader preparation or representation learning.

Do not merely increase manager-training iterations.

### Failure C — Reader is incompatible with changed memory distributions

Evidence: the frozen reader performs well under the initialization manager but fails under alternative valid coefficient distributions.

Action: expand reader preparation across memory-management behaviors, or extend the curriculum.

Keep the manager frozen while performing this diagnostic preparation.

### Failure D — Insufficient learning signal

Evidence: high reward variance, unstable policy updates, collapsed action distributions, or insignificant learning despite confirmed memory usefulness.

Action: investigate reward scaling, episode length, batch size, and gradient statistics.

Only after diagnosing this issue consider KLPO or counterfactual methods.

### Failure E — Coefficient has little effect

Evidence: fixed-coefficient sweeps produce nearly identical task performance.

Action: check memory capacity, overwrite frequency, and whether the task can be solved without competing writes.

Do not manufacture a performance difference by changing task objectives.

### Failure F — Slot selection dominates

Evidence: learned-slot-only matches the full learned manager, while learned-coefficient-only does not improve.

Conclusion: the experiment supports learning memory placement but does not yet support the coefficient-elimination hypothesis.

---

## 11. Implementation Requirements

Organize the implementation into separate responsibilities:

1. Task environment adapters.
2. Configurable curriculum scheduling.
3. Reader preparation.
4. Manager initialization.
5. Frozen-reader manager training.
6. Fixed-coefficient and LRU baselines.
7. Closed-loop evaluation.
8. Diagnostics and result aggregation.
9. Checkpointing and reproducibility tests.

Reuse existing Step 2 modules rather than duplicating their policy-gradient logic.

Implement tests for:

- Initial categorical probabilities.
- Uniform slot selection.
- Alpha convention conversion.
- Memory-update correctness.
- No-op correctness.
- Slot-conditioned candidate preservation.
- Reader freezing during manager training.
- Manager freezing during reader training.
- Reward and return-to-go indexing.
- Episode termination and reset.
- Gradient isolation.
- Environment action compatibility.
- Checkpoint restoration.
- Deterministic evaluation under fixed seeds.
- Independent train/validation/test environment seeds.

Use structured configuration files for all hyperparameters.

Write intermediate artifacts after each completed stage.

Avoid one monolithic experiment script.

---

## 12. Deliverables

Produce:

**A. Implementation**

Runnable code for both selected tasks, reader preparation, manager training, baselines, and evaluation.

**B. Configuration files**

Include separate smoke, reader-preparation, manager-training, fixed-baseline and full-comparison configurations.

**C. Experiment results**

Include:

- Reader memory-dependence measurements.
- Reader preparation curves.
- Manager learning curves.
- Fixed-coefficient comparisons.
- Slot-only and coefficient-only ablations.
- Final coefficient distributions.
- Closed-loop task performance.
- Compute and throughput statistics.
- Per-seed measurements.

**D. Written report**

Create `step_3_results.md` containing:

1. Exact task configurations.
2. Curriculum implementation.
3. Reader readiness results.
4. Manager optimization results.
5. Fixed-coefficient reference comparisons.
6. Evidence for or against H1–H5.
7. Identified failure modes.
8. Unresolved confounds.
9. Reproduction commands.

Clearly distinguish measured results from hypotheses.

**E. Reproducibility**

Provide complete commands to:

- Run tests.
- Prepare the readers.
- Resume reader preparation.
- Train managers.
- Resume manager training.
- Evaluate checkpoints.
- Reproduce comparisons.

Preserve previous Step 1 and Step 2 experiments.

---

## 13. Implementation Order

Execute development in this order:

1. Inspect the existing repository and upstream task implementations.
2. Verify task availability, exact benchmark settings and coefficient conventions.
3. Implement task adapters.
4. Implement and test manager initialization.
5. Implement curriculum reader preparation.
6. Validate memory dependence and decodability.
7. Implement frozen-reader REINFORCE training.
8. Run integration smoke tests on both tasks.
9. Implement controlled baselines and ablations.
10. Execute full experiments when validation gates pass.
11. Aggregate results and write `step_3_results.md`.

Do not silently bypass a failed validation gate.

Do not fabricate unexecuted results.

Do not introduce KLPO, a critic, counterfactual rollouts, memory BPTT or alternating reader-manager optimization into the primary experiment.

## 14. Final Research Decision

After the experiments, classify the outcome into one of four categories:

**Outcome A — Successful coefficient elimination**

The learned manager reaches competitive performance on both tasks without task-specific coefficient tuning.

**Outcome B — Successful adaptive blending**

Outcome A holds, and the manager additionally demonstrates meaningful state-dependent coefficient selection.

**Outcome C — Partial success**

Learning improves performance but does not match strong fixed-coefficient systems, or benefits arise primarily from slot selection.

**Outcome D — Inconclusive or unsuccessful**

Reader limitations, insufficient learning signal, weak coefficient sensitivity, or unstable optimization prevent a reliable conclusion.

Do not claim that adaptive memory management is superior merely because it uses a learned coefficient.

The goal is to determine whether task-dependent manual memory-blending coefficients can be eliminated through learning, and whether that learning produces genuine adaptive memory behavior.
