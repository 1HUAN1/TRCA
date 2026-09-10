#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROFILE="${1:-full}"
[[ $# -gt 0 ]] && shift

case "$PROFILE" in
  full|smoke) ;;
  *) echo "[error] profile must be full or smoke, got: $PROFILE" >&2; exit 2 ;;
esac

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_RUN_NAME="fgpo_searchqa_qwen25_3b_${PROFILE}_s${SEED:-0}_${RUN_STAMP}"
RUN_NAME="${1:-${RUN_NAME:-$DEFAULT_RUN_NAME}}"
[[ $# -gt 0 ]] && shift

export RUN_NAME
export MODEL_SIZE="${MODEL_SIZE:-3b}"
export DATASET="${DATASET:-all}"
export SHARED_RETRIEVER="${SHARED_RETRIEVER:-1}"
export SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
export SEARCH_TOPK="${SEARCH_TOPK:-3}"
export SEARCH_TIMEOUT="${SEARCH_TIMEOUT:-60}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED="${SEED:-0}"

if [[ "$PROFILE" == "full" ]]; then
  export TRAIN_FILE="${TRAIN_FILE:-/root/data/searchR1_processed_direct/train.parquet}"
  export VAL_FILE="${VAL_FILE:-/root/data/searchR1_processed_direct/test.parquet}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
  export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-512}"
  export GROUP_SIZE="${GROUP_SIZE:-8}"
  export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-512}"
  export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-16}"
  export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
  export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
  export MAX_STEPS="${MAX_STEPS:-4}"
  export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
  export SAVE_FREQ="${SAVE_FREQ:-50}"
  export TEST_FREQ="${TEST_FREQ:-50}"
  export ACTOR_LR="${ACTOR_LR:-1e-6}"
  export SIMILARITY_THRESH="${SIMILARITY_THRESH:-0.80}"
  export INVALID_ACTION_PENALTY_COEF="${INVALID_ACTION_PENALTY_COEF:-0.01}"
  export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
  export VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
  export VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-false}"
fi

export DENSE_REWARD_WEIGHT="${DENSE_REWARD_WEIGHT:-0.10}"
export HBRR_REWARD_WEIGHT="${HBRR_REWARD_WEIGHT:-0.03}"
export TERMINAL_REWARD_WEIGHT="${TERMINAL_REWARD_WEIGHT:-1.0}"
export OMEGA_ENV="${OMEGA_ENV:-0.8}"
export OMEGA_FCSR="${OMEGA_FCSR:-0.08}"
export LAMBDA_FALLBACK="${LAMBDA_FALLBACK:-0.02}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs/search_fcsr_runs/$RUN_NAME}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints/verl_agent_search/$RUN_NAME}"
export LOG_DIR CKPT_DIR
mkdir -p "$LOG_DIR" "$CKPT_DIR"
printf '%s\n' "$$" > "$LOG_DIR/launcher.pid"

record_exit() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$LOG_DIR/exit_code"
  exit "$status"
}
trap record_exit EXIT

EXTRA_ARGS=("$@")
if [[ "$PROFILE" == "smoke" ]]; then
  EXTRA_ARGS+=("trainer.total_training_steps=${SMOKE_TRAINING_STEPS:-1}")
fi

cd "$REPO_ROOT"
bash "$SCRIPT_DIR/run_search.sh" vllm fgpo "$PROFILE" MODEL_SIZE="$MODEL_SIZE" DATASET="$DATASET" "${EXTRA_ARGS[@]}"
