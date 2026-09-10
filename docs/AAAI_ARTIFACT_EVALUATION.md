# AAAI Artifact Evaluation Guide

Artifact for: **TRCA: Transition-Wise Rubric Credit Assignment for Long-Horizon LLM Agents**

Repository path on the evaluation server:

```bash
/root/data/zh/gigpo+fcsr_verl
```

This guide is written for artifact evaluators who want to verify that the submitted code implements the paper's transition-wise rubric credit assignment pipeline and can reproduce the main experiment workflows.

## 1. Artifact Scope

The artifact contains a modified `verl`-based multi-turn RL training stack for long-horizon LLM agents. The paper names the method **TRCA**. Some source files retain earlier internal names:

| Paper concept | Code name / location |
| --- | --- |
| Transition-wise diagnostics | `reward_methods/diagnostics.py` |
| Rollout to step/trajectory schema | `reward_methods/schema.py` |
| Foundational transition reward | `reward_methods/dcsr.py`, mixed by `reward_methods/mixer.py` |
| Hindsight / breakthrough-style reward rewriting | `reward_methods/hbrr.py`, mixed by `reward_methods/mixer.py` |
| TRCA dense reward stack | `reward_method=fcsr_lite` |
| TRCA + GiGPO advantage estimator | `algorithm.adv_estimator=fcsr_gigpo`, implemented in `gigpo/core_fcsr_gigpo.py` |
| Trainer integration | `verl/trainer/ppo/ray_trainer.py` |
| ALFWorld/WebShop launchers | `examples/fcsr_trainer/run_alfworld_fcsr_gigpo_qwen25_*.sh`, `examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_*.sh` |
| SearchQA launchers and evaluation | `examples/fcsr_trainer/run_search.sh`, `agent_system/evaluation/searchqa_eval.py` |

The artifact is intended to support the following claims:

1. The code converts environment transitions into structured step records, diagnostics, and reward components.
2. The `fcsr_lite` reward path produces transition-wise dense supervision without requiring successful trajectory anchors.
3. The `fcsr_gigpo` advantage estimator combines environment returns and transition-wise rubric rewards before PPO updates.
4. The same implementation path supports ALFWorld, WebShop, and search-augmented QA experiments.
5. Reproduction can be checked at three levels: syntax/unit checks, smoke runs, and full training/evaluation runs.

## 2. Artifact Metadata

Record the exact repository state before evaluation:

```bash
cd /root/data/zh/gigpo+fcsr_verl
git rev-parse --short HEAD
git status --short
```

The artifact was inspected at base commit:

```text
20bd331
```

The working tree contains experiment-specific local modifications and generated files. Do not reset the tree unless the authors explicitly provide a clean archive or a target commit.

## 3. Hardware and Software Requirements

### Minimal checks

- CPU-only is enough for `py_compile` and most reward/advantage unit tests.
- Python environment used by the current server setup:

```bash
/root/data/conda/envs/verl-agent/bin/python
```

### GPU checks and training

Full training requires CUDA-visible GPUs. Before running GPU experiments, check:

```bash
nvidia-smi
/root/data/conda/envs/verl-agent/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
PY
```

Recommended hardware profiles:

| Task | Recommended GPUs | Notes |
| --- | ---: | --- |
| Syntax check | 0 | CPU only |
| Reward/advantage unit tests | 0 | Requires `pytest` |
| SearchQA dry-run | 0 | Uses `DRY_RUN=1` |
| SearchQA smoke | 2 | 3B model profile by default |
| ALFWorld 1.5B training | 4 | Default launcher profile |
| ALFWorld 7B training | 8 | Default launcher profile |
| WebShop 1.5B/7B training | 8 | Current launcher uses 8 GPUs |
| SearchQA full 3B/7B | 8 | Retriever service also required |

If `torch.cuda.device_count()` is 0, do not start GPU training. Run the CPU checks and dry-run checks only.

## 4. Data and Model Assets

The inspected server already contains these local assets:

```text
/root/data/zh/huggingface/Qwen2.5-1.5B-Instruct
/root/data/zh/huggingface/Qwen2.5-3B-Instruct
/root/data/zh/huggingface/Qwen2.5-7B-Instruct
/root/data/verl-agent/text/train.parquet
/root/data/verl-agent/text/test.parquet
/root/data/searchR1_processed_direct/train.parquet
/root/data/searchR1_processed_direct/test.parquet
data/searchqa_fgpo/smoke_train.parquet
data/searchqa_fgpo/smoke_val.parquet
data/searchqa_fgpo/eval_full/*.parquet
```

For ALFWorld, also confirm:

```bash
find /root/data/.cache/alfworld/json_2.1.1/train -name game.tw-pddl | head
find /root/data/.cache/alfworld/json_2.1.1/valid_seen -name game.tw-pddl | head
```

If ALFWorld data is stored elsewhere, set:

```bash
export ALFWORLD_DATA=/path/to/alfworld
```

## 5. Quick Functional Evaluation

Run from the repository root:

```bash
cd /root/data/zh/gigpo+fcsr_verl
export PYTHON_BIN=/root/data/conda/envs/verl-agent/bin/python
export PYTHONPATH="$PWD:$PWD/agent_system/environments/env_package/alfworld:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
```

### 5.1 Syntax check

This check does not require `pytest` and was verified to pass on the inspected server.

```bash
$PYTHON_BIN -m py_compile \
  reward_methods/schema.py reward_methods/normalization.py reward_methods/diagnostics.py \
  reward_methods/dcsr.py reward_methods/hbrr.py reward_methods/mixer.py \
  gigpo/core_fcsr_gigpo.py verl/trainer/ppo/ray_trainer.py \
  agent_system/evaluation/searchqa_eval.py agent_system/evaluation/hotpotqa_eval.py \
  examples/data_preprocess/prepare_searchqa_data.py examples/data_preprocess/prepare_hotpotqa_search.py
```

Expected result: no output and exit code 0.

### 5.2 Unit tests

If `pytest` is not installed in the evaluation environment, install it in the environment or in a temporary user-local location first:

```bash
$PYTHON_BIN -m pip install pytest
```

Then run:

```bash
$PYTHON_BIN -m pytest -q \
  tests/test_reward_schema.py \
  tests/test_dcsr_reward.py \
  tests/test_hbrr_reward.py \
  tests/test_fcsr_gigpo.py \
  tests/test_fcsr_mixer.py \
  tests/test_diagnostics.py \
  tests/test_hotpotqa_fcsr_script.py
```

Expected result: all selected tests pass. These tests cover schema adaptation, diagnostics, dense reward components, hindsight rewriting, `fcsr_gigpo` advantage shape/masking, and launcher safety checks.

### 5.3 SearchQA launcher dry-run

This validates argument construction without starting training:

```bash
DRY_RUN=1 \
SKIP_RETRIEVER_CHECK=1 \
REQUIRE_FREE_GPUS=0 \
bash examples/fcsr_trainer/run_search.sh vllm fgpo smoke MODEL_SIZE=3b DATASET=nq
```

Expected result:

- A log directory under `outputs/search_fcsr_runs/`.
- A `RESOLVED_COMMAND:` line showing `algorithm.adv_estimator=fcsr_gigpo`.
- Reward overrides including `++reward_method=fcsr_lite`, `++use_diagnostics=true`, `++use_failure_rewriting=true`.
- Exit code 0.

## 6. Smoke Evaluation

Use smoke runs to confirm end-to-end wiring before starting full experiments.

### 6.1 SearchQA smoke

Start or verify the retriever first. For CPU retriever debugging:

```bash
cd /root/data/zh/gigpo+fcsr_verl
RETRIEVER_DEVICE=cpu FAISS_GPU=0 PORT=8000 \
bash examples/search/retriever/retrieval_launch.sh
```

For GPU FAISS retrieval, use the default launcher after confirming GPUs are visible:

```bash
PORT=8000 bash examples/search/retriever/retrieval_launch.sh
```

In a second shell:

```bash
cd /root/data/zh/gigpo+fcsr_verl
WANDB_MODE=offline \
bash examples/fcsr_trainer/run_search.sh vllm fgpo smoke MODEL_SIZE=3b DATASET=nq
```

Expected evidence:

- Training log under `outputs/search_fcsr_runs/<run_name>/train.log`.
- `FCSR_METRIC_JSON` lines.
- Metrics such as `fcsr/reward_mean`, `fcsr/dcsr_mean`, `fcsr/hbrr_mean`, `fcsr_gigpo/anchor_coverage`, `search/retrieval_success_rate`.

### 6.2 ALFWorld wiring smoke

The default ALFWorld launcher is a training launcher. For a short smoke, override sizes and frequencies:

```bash
cd /root/data/zh/gigpo+fcsr_verl
RUN_NAME=ae_alfworld_fcsr_smoke_$(date +%Y%m%d_%H%M%S) \
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

Expected evidence:

- Log directory under `outputs/alfworld_fcsr_gigpo_runs/<run_name>/`.
- `FCSR_METRIC_JSON` lines.
- `alfworld/*format*`, `alfworld_gate/*`, `fcsr/*`, and `fcsr_gigpo/*` metrics.

## 7. Full Reproduction Commands

Full runs are expensive and should be launched only when the GPU allocation is clear.

### 7.1 ALFWorld 1.5B

```bash
cd /root/data/zh/gigpo+fcsr_verl
WANDB_MODE=offline \
N_GPUS_PER_NODE=4 \
bash examples/fcsr_trainer/run_alfworld_fcsr_gigpo_qwen25_1p5b.sh
```

Primary outputs:

```text
outputs/alfworld_fcsr_gigpo_runs/<run_name>/
checkpoints/verl_agent_alfworld/<run_name>/
```

### 7.2 ALFWorld 7B

```bash
cd /root/data/zh/gigpo+fcsr_verl
WANDB_MODE=online \
N_GPUS_PER_NODE=8 \
bash examples/fcsr_trainer/run_alfworld_fcsr_gigpo_qwen25_7b_8gpu.sh
```

### 7.3 WebShop 1.5B

```bash
cd /root/data/zh/gigpo+fcsr_verl
WANDB_MODE=offline \
bash examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_1p5b.sh vllm fcsr
```

The WebShop launcher includes strict gate options:

```text
++webshop_strict_gate.enabled=true
++webshop_strict_gate.allow_search_reward=false
++webshop_strict_gate.allow_navigation_positive_reward=false
++webshop_strict_gate.allow_generic_click_positive_reward=false
```

### 7.4 SearchQA 3B full

Start retriever:

```bash
cd /root/data/zh/gigpo+fcsr_verl
PORT=8000 bash examples/search/retriever/retrieval_launch.sh
```

Launch training:

```bash
WANDB_MODE=offline \
bash examples/fcsr_trainer/run_search.sh vllm fgpo full MODEL_SIZE=3b DATASET=all
```

### 7.5 SearchQA evaluation-only run

```bash
cd /root/data/zh/gigpo+fcsr_verl
VAL_ONLY=true \
RESUME_CKPT=/root/data/zh/gigpo+fcsr_verl/checkpoints/verl_agent_search/<run_name>/global_step_<step> \
bash examples/fcsr_trainer/run_search.sh vllm fgpo eval MODEL_SIZE=3b DATASET=all
```

After an eval run, the script writes:

```text
outputs/search_fcsr_runs/<run_name>/searchqa_eval_summary.json
outputs/search_fcsr_runs/<run_name>/searchqa_eval_summary.csv
outputs/search_fcsr_runs/<run_name>/searchqa_eval_summary.tex
```

## 8. Interpreting Results

For training logs, inspect:

```bash
grep -E "FCSR_METRIC_JSON|fcsr_gigpo/|episode/success_rate|eval/|searchqa" outputs/**/train.log
```

Important metrics:

| Metric | Meaning |
| --- | --- |
| `fcsr/reward_mean` | Mean mixed transition-wise rubric reward |
| `fcsr/dcsr_mean` | Foundational dense transition signal |
| `fcsr/hbrr_mean` | Hindsight / breakthrough-style component |
| `fcsr/invalid_rate`, `fcsr/noop_rate`, `fcsr/repeated_rate` | Failure diagnostics |
| `reward/fcsr_score_strict_mean` | Post-gate transition reward used by `fcsr_gigpo` |
| `fcsr_gigpo/anchor_coverage` | Fraction of steps covered by valid anchor groups |
| `fcsr_gigpo/final_adv_mean` | Final mixed advantage summary |
| `episode/success_rate` | Environment success during rollout |
| `eval/hotpotqa_f1`, `eval/hotpotqa_em` | SearchQA/HotpotQA evaluation metrics |

For ALFWorld checkpoint choice, check whether a run produced:

```text
best_checkpoint_metadata.json
latest_checkpointed_iteration.txt
```

Prefer the best validation checkpoint when `eval/best_success_rate` is available.

## 9. Troubleshooting

### No CUDA devices are visible

Do not start training. Run syntax checks and dry-runs only. Ask the provider to attach GPUs or start the job in the GPU-enabled container/session.

### `pytest` is missing

Install it before unit tests:

```bash
/root/data/conda/envs/verl-agent/bin/python -m pip install pytest
```

### Retriever preflight fails

Check:

```bash
curl http://127.0.0.1:8000/retrieve
```

Then restart:

```bash
PORT=8000 bash examples/search/retriever/retrieval_launch.sh
```

If GPU FAISS fails, try CPU retriever for smoke debugging:

```bash
RETRIEVER_DEVICE=cpu FAISS_GPU=0 PORT=8000 bash examples/search/retriever/retrieval_launch.sh
```

### ALFWorld data check fails

Set `ALFWORLD_DATA` to a directory containing:

```text
json_2.1.1/train
json_2.1.1/valid_seen
```

### vLLM allocator conflict

The launchers already unset incompatible `PYTORCH_CUDA_ALLOC_CONF` values. If a custom environment sets `expandable_segments:True`, unset it manually:

```bash
unset PYTORCH_CUDA_ALLOC_CONF
```

## 10. Evaluation Recommendation

For AAAI artifact review, use the following progression:

1. Run Section 5.1 syntax check.
2. Install `pytest` if allowed and run Section 5.2.
3. Run Section 5.3 dry-run to verify the command line and config path.
4. If GPUs are available, run Section 6 smoke experiments.
5. If time and allocation permit, run one full benchmark from Section 7 and compare logs/checkpoints against the expected metrics in Section 8.

