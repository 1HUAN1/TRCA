# TRCA Appendix: Verified Values and Required Corrections

This note records only values that can be traced to files in
`/root/data/zh/gigpo+fcsr_verl`. It is intended as the source of truth for
revising the manuscript appendix. It must not be cited as evidence for claims
that are absent from the repository.

## 1. Diagnostic study supported by the repository

Source directory:
`analysis/all_failure_groups_20260720_205720/`

The available study uses Qwen2.5-1.5B-Instruct on ALFWorld, one training seed,
104 task-conditioned groups per checkpoint, and 8 stochastic rollouts per
group. The sampled checkpoints are global steps 20, 80, and 140. Each
checkpoint therefore contains 832 rollouts; the pooled study contains 312
groups and 2,496 rollout rows.

At the early checkpoint (global step 20):

- 83 of 104 groups contain no successful rollout.
- The all-failure group rate is 79.81%.
- The group-bootstrap 95% interval reported by the analysis is
  [72.12%, 87.50%].
- The marginal rollout failure rate is 93.27%.

At global steps 80 and 140, the all-failure group rates are respectively
34.62% and 14.42%.

These values come from `analysis_report.md`, `phase_summary.csv`,
`checkpoint_summary.csv`, and `qa_report.json`. They do not support the
manuscript's current 96.5% marginal failure and 85.6% all-failure-group claims.

Paper-ready wording:

> We evaluate success scarcity using Qwen2.5-1.5B-Instruct checkpoints on
> ALFWorld. At each checkpoint, we sample eight stochastic rollouts for each of
> 104 task-conditioned initial states. At global step 20, 83 of 104 groups
> contain no successful rollout, corresponding to an all-failure-group rate of
> 79.8% (group-bootstrap 95% CI: 72.1--87.5%). The marginal rollout failure rate
> is 93.3%. The all-failure-group rate decreases to 34.6% at step 80 and 14.4%
> at step 140. These intervals quantify task-level sampling uncertainty for one
> training seed and should not be interpreted as cross-seed uncertainty.

The repository does not contain a completed human/LLM transition annotation
study supporting the manuscript's 72.2% useful-transition claim. In
particular, no verified judge model, annotator count, agreement statistic, or
annotation file was found. The corresponding claim and placeholders must not
be filled without the missing annotation artifact.

## 2. Verified ALFWorld training configuration

Source:
`examples/fcsr_trainer/run_alfworld_fcsr_gigpo_qwen25_1p5b.sh`

- Policy: Qwen2.5-1.5B-Instruct
- Rollout group size: 8
- Maximum interaction steps: 25
- Discount factor: 0.95
- Actor learning rate: 1e-6
- PPO mini-batch size: 256
- PPO micro-batch size per GPU: 32
- KL loss coefficient: 0.01
- PPO clipping lower/upper ranges: 0.20/0.28
- Entropy coefficient: 0.001
- Training task batch size: 16
- Validation batch size: 128
- Rollout temperature: 1.0
- Validation temperature: 0.4
- Environment seed in the launcher: 2
- Dense/DCSR reward weight: 0.10
- HBRR reward weight: 0.05
- Terminal reward weight: 1.0
- Invalid/no-op/repeat penalties: 0.20/0.10/0.10
- Environment/transition advantage weights: 0.8/1.0
- Fallback coefficient: 0.02

These are launcher defaults. A paper table should report values from each
run's resolved Hydra configuration, because task- and run-specific overrides
may differ.

## 3. Paper-to-code mismatch that must be corrected

The manuscript describes exactly five `Evidence`, five `Invalidity`, and five
`Execution` Boolean rubric items per benchmark, instantiated by an LLM and
evaluated by deterministic functions. The inspected implementation does not
contain these three five-item libraries.

The actual implementation uses:

- `reward_methods/diagnostics.py` for rule-based signals including action
  validity, state change, no-op, repeated action, tool errors, progress,
  evidence gain, risk, premature completion, recoverability, goal relevance,
  and completion-critical actions;
- `reward_methods/dcsr.py` for the signed foundational transition reward;
- `reward_methods/hbrr.py` for failure-type and hindsight/breakthrough-style
  reward rewriting;
- `reward_methods/mixer.py` for component weighting and clipping; and
- `gigpo/core_fcsr_gigpo.py` for combining environment and dense transition
  advantages.

Therefore, Appendix H.2 must not claim that 45 five-item rubric entries are
"reported below" unless a separate implementation and artifact are supplied.
The appendix should instead document the actual diagnostic fields, rules,
weights, and code locations, or the code must be changed to implement the
paper's stated rubric library.

## 4. Items that remain unsupported

The following cannot be completed from the inspected repository:

- number of human annotators;
- LLM judge identity and version;
- inter-annotator agreement metric and value;
- a transition-level annotation count supporting 72.2%;
- five-item benchmark rubric libraries;
- natural-language-rubric-to-rule conversion procedure;
- verified per-benchmark hyperparameter table for every reported main result.

Filling these fields with inferred or invented values would make the appendix
factually incorrect.
