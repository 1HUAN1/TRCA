#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/data/zh/gigpo+fcsr_verl
SESSION=${SESSION:-webshop_sota_from0_300}

if [[ "${IN_WEBSHOP_TMUX_RUN:-0}" != "1" ]]; then
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[error] tmux session already exists: ${SESSION}" >&2
    echo "Attach with: tmux attach -t ${SESSION}" >&2
    exit 1
  fi
  tmux new-session -d -s "${SESSION}" "IN_WEBSHOP_TMUX_RUN=1 bash '$0'"
  echo "tmux session: ${SESSION}"
  echo "attach: tmux attach -t ${SESSION}"
  exit 0
fi

cd "${REPO_ROOT}"

export PATH=/root/data/conda/envs/verl-agent-webshop/bin:${PATH}
export PYTHON_BIN=/root/data/conda/envs/verl-agent-webshop/bin/python
unset JAVA_HOME
unset JVM_PATH

unset RESUME_STEP
unset RESUME_CKPT
unset RESUME_FROM_PATH
unset WANDB_RUN_ID
unset WANDB_RESUME

TS=$(date +%Y%m%d_%H%M%S)
export RUN_NAME="webshop_fcsr_gigpo_qwen25_1p5b_sota_from0_to300_${TS}"

export CKPT_DIR="${REPO_ROOT}/checkpoints/verl_agent_webshop/${RUN_NAME}"
export LOG_DIR="${REPO_ROOT}/outputs/fcsr_gigpo_runs/${RUN_NAME}"
export LOG_FILE="${LOG_DIR}/train.from0_to300.log"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

export MODEL_PATH=/root/data/zh/huggingface/Qwen2.5-1.5B-Instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS_PER_NODE=4
export TP_SIZE=2
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=128
export GROUP_SIZE=8
export NUM_CPUS_PER_ENV_WORKER=0.2
export SAVE_FREQ=20
export TEST_FREQ=10
export TOTAL_EPOCHS=300
export MONITOR_KL_ESTIMATORS=true
export KL_MONITOR_INTERVAL=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=online

export RESET_RAY=1

bash examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_1p5b.sh vllm fcsr \
  trainer.val_before_train=False \
  trainer.default_local_dir="${CKPT_DIR}" \
  data.val_batch_size=16 \
  2>&1 | tee -a "${LOG_FILE}"
