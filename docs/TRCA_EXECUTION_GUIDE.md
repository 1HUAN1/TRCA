# TRCA Code and Paper Execution Guide

This document links the paper **TRCA: Transition-Wise Rubric Credit Assignment for Long-Horizon LLM Agents** to the runnable code in `/root/data/zh/gigpo+fcsr_verl`.

The paper name is **TRCA**. The implementation still uses internal names such as `FCSR`, `FGPO`, and `fcsr_gigpo`. Treat these as implementation aliases for the paper's transition-wise rubric credit assignment pipeline.

## 1. Repository Entry Points

```bash
cd /root/data/zh/gigpo+fcsr_verl
export PYTHON_BIN=/root/data/conda/envs/verl-agent/bin/python
export RAY_BIN=/root/data/conda/envs/verl-agent/bin/ray
export PYTHONPATH="$PWD:$PWD/agent_system/environments/env_package/alfworld:${PYTHONPATH:-}"
```

Record the environment:

```bash
git rev-parse --short HEAD
git status --short
$PYTHON_BIN --version
$PYTHON_BIN - <<'PY'
import torch, transformers, ray
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "gpu_count", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("ray", ray.__version__)
PY
```

## 2. Paper-to-Code Map

The paper argues that sparse terminal success is insufficient for long-horizon RL and that failed rollouts still contain useful transition evidence. The implementation realizes this through the following pipeline.

### 2.1 Rollout adaptation

Code:

```text
reward_methods/schema.py
agent_system/multi_turn_rollout/rollout_loop.py
agent_system/multi_turn_rollout/utils.py
```

Purpose:

- Convert raw rollout rows into `TrajectoryRecord` and `StepRecord`.
- Extract task ID, action, observation, next observation, terminal reward, and success.
- Preserve environment-specific fields for later diagnostics.

### 2.2 Transition diagnostics

Code:

```text
reward_methods/diagnostics.py
agent_system/environments/prompts/alfworld.py
agent_system/environments/prompts/search.py
agent_system/environments/env_package/alfworld/projection.py
```

Purpose:

- Identify valid/invalid actions.
- Detect no-op, repeated action, premature finish/buy, state change, progress, evidence gain, and risk.
- Dispatch environment-specific parsers for ALFWorld, WebShop, SearchQA, Sokoban, and generic tasks.

### 2.3 Foundational transition reward

Code:

```text
reward_methods/dcsr.py
```

Purpose:

- Compute a signed transition reward from progress, validity, recovery, risk, invalid/no-op/repeat penalties, and deadlock risk.
- This is the implementation-side dense transition signal corresponding to the paper's rubric-grounded step reward.

### 2.4 Hindsight / breakthrough reward rewriting

Code:

```text
reward_methods/hbrr.py
```

Purpose:

- Classify failure type as progress, exploratory, recoverable, irreversible, redundant, premature, recovery, or unknown.
- Add hindsight reward components and bottleneck penalties.
- Provide the implementation-side complement to the paper's breakthrough / milestone-style credit.

### 2.5 Reward mixing and logging

Code:

```text
reward_methods/mixer.py
reward_methods/normalization.py
reward_methods/webshop_strict.py
```

Purpose:

- Select `reward_method`.
- Build `fcsr_lite` rewards by combining DCSR, HBRR, optional IPRR, terminal anchor, and format reward.
- Write `fcsr_score`, `fcsr_score_strict`, `fcsr_reward_components`, and `fcsr_diagnostic` back into the rollout batch.
- Emit `FCSR_METRIC_JSON` lines.

Key modes:

| Mode | Meaning |
| --- | --- |
| `reward_method=rewardflow` | Baseline reward path |
| `reward_method=dcsr` | Foundational dense reward only |
| `reward_method=fcsr_lite` | Full TRCA/FCSR reward stack |
| `reward_method=webshop_iprr_lite` | WebShop variant with information-potential reward |

### 2.6 Advantage estimation

Code:

```text
gigpo/core_fcsr_gigpo.py
verl/trainer/ppo/ray_trainer.py
```

Purpose:

- Add `AdvantageEstimator.FCSR_GiGPO = "fcsr_gigpo"`.
- Read `fcsr_score_strict` or `fcsr_score` from batch fields.
- Read environment rewards from `env_reward_original` or `rewards`.
- Compute group-level advantages from environment returns and transition-wise rubric returns.
- Combine them using `omega_env`, `omega_fcsr`, and `lambda_fallback`.

Important config keys:

```text
algorithm.adv_estimator=fcsr_gigpo
++algorithm.fcsr_gigpo.gamma_env=0.95
++algorithm.fcsr_gigpo.gamma_fcsr=0.95
++algorithm.fcsr_gigpo.omega_env=0.8
++algorithm.fcsr_gigpo.omega_fcsr=<task-specific>
++algorithm.fcsr_gigpo.lambda_fallback=0.02
++algorithm.fcsr_gigpo.min_anchor_group_size=2
++algorithm.fcsr_gigpo.min_distinct_actions=2
```

## 3. Clean Sanity Checks

### 3.1 Syntax check

```bash
cd /root/data/zh/gigpo+fcsr_verl
export PYTHON_BIN=/root/data/conda/envs/verl-agent/bin/python
export PYTHONPATH="$PWD:$PWD/agent_system/environments/env_package/alfworld:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

$PYTHON_BIN -m py_compile \
  reward_methods/schema.py reward_methods/normalization.py reward_methods/diagnostics.py \
  reward_methods/dcsr.py reward_methods/hbrr.py reward_methods/mixer.py \
  gigpo/core_fcsr_gigpo.py verl/trainer/ppo/ray_trainer.py \
  agent_system/evaluation/searchqa_eval.py agent_system/evaluation/hotpotqa_eval.py
```

Expected: exit code 0.

### 3.2 Unit tests

Install `pytest` if needed:

```bash
$PYTHON_BIN -m pip install pytest
```

Run:

```bash
$PYTHON_BIN -m pytest -q \
  tests/test_reward_schema.py \
  tests/test_dcsr_reward.py \
  tests/test_hbrr_reward.py \
  tests/test_fcsr_gigpo.py \
  tests/test_fcsr_mixer.py \
  tests/test_diagnostics.py \
  tests/test_validation_success_counts.py \
  tests/test_searchqa_eval_summary.py \
  tests/test_hotpotqa_fcsr_script.py
```

Expected: all tests pass. These tests are CPU-level tests for reward schema, diagnostics, FCSR reward mixing, advantage computation, validation accounting, and SearchQA/HotpotQA launch/eval utilities.

## 4. Running ALFWorld Experiments

### 4.1 Data preflight

```bash
export ALFWORLD_DATA=${ALFWORLD_DATA:-/root/data/.cache/alfworld}
find "$ALFWORLD_DATA/json_2.1.1/train" -name game.tw-pddl | wc -l
find "$ALFWORLD_DATA/json_2.1.1/valid_seen" -name game.tw-pddl | wc -l
```

Both counts must be nonzero.

### 4.2 ALFWorld 1.5B TRCA/FCSR-GiGPO

```bash
cd /root/data/zh/gigpo+fcsr_verl
WANDB_MODE=offline \
MODEL_PATH=/root/data/zh/huggingface/Qwen2.5-1.5B-Instruct \
N_GPUS_PER_NODE=4 \
bash examples/fcsr_trainer/run_alfworld_fcsr_gigpo_qwen25_1p5b.sh
```

The launcher uses:

```text
algorithm.adv_estimator=fcsr_gigpo
++reward_method=fcsr_lite
++use_diagnostics=true
++use_failure_rewriting=true
++dense_reward_weight=0.10
++hbrr_reward_weight=0.05
env.env_name=alfworld/AlfredTWEnv
env.rollout.n=8
```

Outputs:

```text
outputs/alfworld_fcsr_gigpo_runs/<run_name>/
checkpoints/verl_agent_alfworld/<run_name>/
```

### 4.3 ALFWorld 7B TRCA/FCSR-GiGPO

```bash
cd /root/data/zh/gigpo+fcsr_verl
WANDB_MODE=online \
MODEL_PATH=/root/data/zh/huggingface/Qwen2.5-7B-Instruct \
N_GPUS_PER_NODE=8 \
bash examples/fcsr_trainer/run_alfworld_fcsr_gigpo_qwen25_7b_8gpu.sh
```

Use `WANDB_MODE=offline` if reviewers cannot log to W&B.

### 4.4 ALFWorld smoke

```bash
RUN_NAME=ae_alfworld_smoke_$(date +%Y%m%d_%H%M%S) \
WANDB_MODE=offline \
TRAIN_DATA_SIZE=2 \
VAL_DATA_SIZE=4 \
VAL_BATCH_SIZE=4 \
GROUP_SIZE=2 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=-1 \
TEST_FREQ=1 \
N_GPUS_PER_NODE=2 \
TP_SIZE=1 \
PPO_MINI_BATCH_SIZE=4 \
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
bash examples/fcsr_trainer/run_alfworld_fcsr_gigpo_qwen25_1p5b.sh
```

Check:

```bash
grep -E "FCSR_METRIC_JSON|fcsr_gigpo/|episode/success_rate|alfworld_gate/" outputs/alfworld_fcsr_gigpo_runs/$RUN_NAME/* -R
```

## 5. Running WebShop Experiments

WebShop needs Java/JVM for the environment. The launcher searches for a JDK and `libjvm.so`.

### 5.1 WebShop 1.5B

```bash
cd /root/data/zh/gigpo+fcsr_verl
WANDB_MODE=offline \
bash examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_1p5b.sh vllm fcsr
```

Outputs:

```text
outputs/fcsr_gigpo_runs/<run_name>/train.log
checkpoints/verl_agent_webshop/<run_name>/
```

The WebShop path uses a stricter post-gate reward:

```text
++webshop_strict_gate.enabled=true
++use_information_potential=true
++reward_clip_min=-0.2
++reward_clip_max=0.2
++fcsr_strict_clip_min=-0.2
++fcsr_strict_clip_max=0.2
```

### 5.2 WebShop resume

```bash
cd /root/data/zh/gigpo+fcsr_verl
RESUME_STEP=160 \
bash examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_1p5b_resume.sh
```

If using a different checkpoint:

```bash
RESUME_FROM_PATH=/root/data/zh/gigpo+fcsr_verl/checkpoints/verl_agent_webshop/<run_name>/global_step_<step> \
bash examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_1p5b_resume.sh
```

## 6. Running SearchQA Experiments

SearchQA uses an external retriever service.

### 6.1 Start retriever

GPU FAISS path:

```bash
cd /root/data/zh/gigpo+fcsr_verl
PORT=8000 bash examples/search/retriever/retrieval_launch.sh
```

CPU fallback for smoke debugging:

```bash
RETRIEVER_DEVICE=cpu FAISS_GPU=0 PORT=8000 \
bash examples/search/retriever/retrieval_launch.sh
```

### 6.2 Verify retriever

```bash
curl -s http://127.0.0.1:8000/retrieve
```

The training launcher has a built-in retriever preflight unless `SKIP_RETRIEVER_CHECK=1`.

### 6.3 SearchQA dry-run

```bash
cd /root/data/zh/gigpo+fcsr_verl
DRY_RUN=1 \
SKIP_RETRIEVER_CHECK=1 \
REQUIRE_FREE_GPUS=0 \
bash examples/fcsr_trainer/run_search.sh vllm fgpo smoke MODEL_SIZE=3b DATASET=nq
```

Expected:

```text
RESOLVED_COMMAND:
algorithm.adv_estimator=fcsr_gigpo
++reward_method=fcsr_lite
++use_failure_rewriting=true
```

### 6.4 SearchQA smoke

```bash
WANDB_MODE=offline \
bash examples/fcsr_trainer/run_search.sh vllm fgpo smoke MODEL_SIZE=3b DATASET=nq
```

### 6.5 SearchQA full

```bash
WANDB_MODE=offline \
bash examples/fcsr_trainer/run_search.sh vllm fgpo full MODEL_SIZE=3b DATASET=all
```

For 7B:

```bash
WANDB_MODE=offline \
bash examples/fcsr_trainer/run_search.sh vllm fgpo full MODEL_SIZE=7b DATASET=all
```

### 6.6 SearchQA evaluation

```bash
VAL_ONLY=true \
RESUME_CKPT=/root/data/zh/gigpo+fcsr_verl/checkpoints/verl_agent_search/<run_name>/global_step_<step> \
bash examples/fcsr_trainer/run_search.sh vllm fgpo eval MODEL_SIZE=3b DATASET=all
```

After evaluation:

```bash
cat outputs/search_fcsr_runs/<run_name>/searchqa_eval_summary.json
```

## 7. Ablations

Use `run_search.sh` modes:

```bash
bash examples/fcsr_trainer/run_search.sh vllm gigpo full MODEL_SIZE=3b DATASET=all
bash examples/fcsr_trainer/run_search.sh vllm dcsr full MODEL_SIZE=3b DATASET=all
bash examples/fcsr_trainer/run_search.sh vllm hbrr full MODEL_SIZE=3b DATASET=all
bash examples/fcsr_trainer/run_search.sh vllm fgpo full MODEL_SIZE=3b DATASET=all
```

Meaning:

| Mode | Configuration |
| --- | --- |
| `gigpo` | `algorithm.adv_estimator=gigpo`, baseline reward path |
| `dcsr` | `algorithm.adv_estimator=fcsr_gigpo`, DCSR-only dense reward |
| `hbrr` | `algorithm.adv_estimator=fcsr_gigpo`, HBRR-only rewriting |
| `fgpo` | Full TRCA/FCSR dense reward stack |

For ALFWorld/WebShop ablations, use the corresponding launcher and override:

```bash
OMEGA_FCSR=0.0
DENSE_REWARD_WEIGHT=0.0
HBRR_REWARD_WEIGHT=0.0
```

or edit the launcher mode where supported.

## 8. Result Extraction

### 8.1 FCSR/TRCA reward metrics

```bash
grep "FCSR_METRIC_JSON" outputs/**/train.log
```

Useful fields:

```text
fcsr/reward_mean
fcsr/reward_positive_rate
fcsr/dcsr_mean
fcsr/hbrr_mean
fcsr/fgpo_reward_mean
fcsr/invalid_rate
fcsr/noop_rate
fcsr/repeated_rate
reward/fcsr_score_strict_mean
reward/fcsr_score_strict_positive_rate
```

### 8.2 Advantage metrics

```bash
grep -E "fcsr_gigpo/" outputs/**/train.log
```

Useful fields:

```text
fcsr_gigpo/anchor_coverage
fcsr_gigpo/env_step_adv_mean
fcsr_gigpo/fcsr_step_adv_mean
fcsr_gigpo/final_adv_mean
fcsr_gigpo/final_adv_clip_frac
fcsr_gigpo/non_anchor_fallback_rate
```

### 8.3 Task metrics

```bash
grep -E "episode/success_rate|eval/best_success_rate|eval/hotpotqa_f1|searchqa" outputs/**/train.log
```

For SearchQA eval summaries:

```bash
find outputs/search_fcsr_runs -name "searchqa_eval_summary.*" -maxdepth 3 -type f
```

## 9. Checkpoint Handling

Latest checkpoint:

```bash
cat checkpoints/<project>/<run_name>/latest_checkpointed_iteration.txt
```

Resume:

```bash
RESUME_FROM_PATH=/root/data/zh/gigpo+fcsr_verl/checkpoints/<project>/<run_name>/global_step_<step> \
bash <launcher>
```

If available, prefer:

```text
best_checkpoint_metadata.json
```

over the latest checkpoint when reporting final validation results.

## 10. Common Failure Modes

| Symptom | Fix |
| --- | --- |
| `No module named pytest` | Install pytest or run `py_compile` only |
| `torch.cuda.device_count() == 0` | Do not launch GPU training in this session |
| `retriever preflight` fails | Start `examples/search/retriever/retrieval_launch.sh` |
| `ALFWORLD_DATA` missing | Export the correct ALFWorld cache path |
| WebShop JVM error | Set `JAVA_HOME` or `JVM_PATH` |
| vLLM allocator crash | `unset PYTORCH_CUDA_ALLOC_CONF` |
| Missing model files | Point `MODEL_PATH` to a local Qwen2.5 model directory |

## 11. Minimal Reviewer Checklist

1. Read the paper abstract and method section.
2. Confirm the code mapping in Section 2.
3. Run the syntax check in Section 3.1.
4. Run unit tests if `pytest` can be installed.
5. Run SearchQA dry-run in Section 6.3.
6. If GPUs and retriever are available, run one smoke experiment.
7. Inspect `FCSR_METRIC_JSON` and `fcsr_gigpo/*` metrics in the produced log.

