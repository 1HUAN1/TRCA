#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash examples/fcsr_trainer/run_search.sh [ENGINE] [MODE] [PROFILE] [KEY=VALUE ...] [hydra.override=...]

Modes: fgpo (default), gigpo, dcsr, hbrr
Profiles: full (default), smoke, eval
Datasets: nq, triviaqa, popqa, hotpotqa, 2wiki, musique, bamboogle, all, available

Examples:
  DRY_RUN=1 bash examples/fcsr_trainer/run_search.sh vllm fgpo smoke
  bash examples/fcsr_trainer/run_search.sh vllm fgpo full MODEL_SIZE=3b DATASET=all
  SHARED_RETRIEVER=1 bash examples/fcsr_trainer/run_search.sh vllm fgpo full MODEL_SIZE=3b DATASET=all
  bash examples/fcsr_trainer/run_search.sh vllm gigpo eval MODEL_SIZE=7b DATASET=nq RESUME_CKPT=/path/to/checkpoint
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$REPO_ROOT"

ENGINE="${ENGINE:-vllm}"
MODE="${MODE:-fgpo}"
PROFILE="${PROFILE:-full}"
HYDRA_EXTRA=()

for positional in ENGINE MODE PROFILE; do
  if [[ $# -gt 0 && "$1" != *=* && "$1" != --* ]]; then
    printf -v "$positional" '%s' "$1"
    shift
  fi
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
  esac
  if [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    export "$1"
  else
    HYDRA_EXTRA+=("$1")
  fi
  shift
done

[[ "$ENGINE" == "vllm" ]] || { echo "[error] only ENGINE=vllm is supported" >&2; exit 2; }
case "$PROFILE" in full|smoke|eval) ;; *) echo "[error] PROFILE=$PROFILE" >&2; exit 2 ;; esac

MODEL_SIZE="${MODEL_SIZE:-3b}"
SHARED_RETRIEVER="${SHARED_RETRIEVER:-1}"
case "$SHARED_RETRIEVER" in 0|1) ;; *) echo "[error] SHARED_RETRIEVER must be 0 or 1" >&2; exit 2 ;; esac
case "$MODEL_SIZE" in
  3b)
    DEFAULT_MODEL_PATH="/root/data/zh/huggingface/Qwen2.5-3B-Instruct"
    DEFAULT_ACTOR_MICRO_BATCH=16
    DEFAULT_GPU_MEMORY_UTILIZATION=0.60
    DEFAULT_GROUP_SIZE=8
    DEFAULT_SIMILARITY_THRESH=0.80
    ;;
  7b)
    DEFAULT_MODEL_PATH="/root/data/zh/huggingface/Qwen2.5-7B-Instruct"
    DEFAULT_ACTOR_MICRO_BATCH=16
    DEFAULT_GPU_MEMORY_UTILIZATION=0.60
    DEFAULT_GROUP_SIZE=5
    DEFAULT_SIMILARITY_THRESH=0.90
    ;;
  *) echo "[error] MODEL_SIZE must be 3b or 7b" >&2; exit 2 ;;
esac

DEFAULT_LOGPROB_MICRO_BATCH=32
DEFAULT_GPU_BUSY_LIMIT_MIB=4096
if [[ "$SHARED_RETRIEVER" == "1" ]]; then
  DEFAULT_GPU_BUSY_LIMIT_MIB=8192
fi
FREE_GPU_MAX_USED_MIB="${FREE_GPU_MAX_USED_MIB:-$DEFAULT_GPU_BUSY_LIMIT_MIB}"

PYTHON_BIN="${PYTHON_BIN:-/root/data/conda/envs/verl-agent/bin/python}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/searchqa_fgpo}"
EVAL_ROOT="${EVAL_ROOT:-$DATA_ROOT/eval_full}"
BASELINE_DATA_ROOT="${BASELINE_DATA_ROOT:-/root/data/searchR1_processed_direct}"
BASELINE_TRAIN_FILE="${BASELINE_TRAIN_FILE:-$BASELINE_DATA_ROOT/train.parquet}"
BASELINE_VAL_FILE="${BASELINE_VAL_FILE:-$BASELINE_DATA_ROOT/test.parquet}"
DATASET="${DATASET:-all}"

case "$DATASET" in
  nq|triviaqa|popqa|hotpotqa|2wiki|2wikimultihopqa|musique|bamboogle|all|available) ;;
  *) echo "[error] unsupported DATASET=$DATASET" >&2; exit 2 ;;
esac

if [[ "$PROFILE" == "smoke" ]]; then
  TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/smoke_train.parquet}"
  VAL_FILE="${VAL_FILE:-$DATA_ROOT/smoke_val.parquet}"
else
  TRAIN_FILE="${TRAIN_FILE:-$BASELINE_TRAIN_FILE}"
  if [[ -z "${VAL_FILE:-}" ]]; then
    case "$DATASET" in
      nq|triviaqa|popqa|hotpotqa|musique|bamboogle) VAL_FILE="$EVAL_ROOT/$DATASET.parquet" ;;
      2wiki|2wikimultihopqa) DATASET=2wiki; VAL_FILE="$EVAL_ROOT/2wiki.parquet" ;;
      available) VAL_FILE="$EVAL_ROOT/available.parquet" ;;
      all) VAL_FILE="$BASELINE_VAL_FILE" ;;
    esac
  fi
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "[warn] unsetting PYTORCH_CUDA_ALLOC_CONF for vLLM" >&2
  unset PYTORCH_CUDA_ALLOC_CONF
fi

if [[ "$PROFILE" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-3}"
  GROUP_SIZE="${GROUP_SIZE:-2}"
  PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
  PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
  MAX_STEPS="${MAX_STEPS:-2}"
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
  SAVE_FREQ="${SAVE_FREQ:--1}"
  TEST_FREQ="${TEST_FREQ:-1}"
  VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
elif [[ "$PROFILE" == "eval" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-512}"
  GROUP_SIZE="${GROUP_SIZE:-1}"
  PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
  PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
  MAX_STEPS="${MAX_STEPS:-4}"
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
  SAVE_FREQ="${SAVE_FREQ:--1}"
  TEST_FREQ="${TEST_FREQ:--1}"
  VAL_BEFORE_TRAIN=true
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-512}"
  GROUP_SIZE="${GROUP_SIZE:-$DEFAULT_GROUP_SIZE}"
  PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-512}"
  PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_ACTOR_MICRO_BATCH}"
  MAX_STEPS="${MAX_STEPS:-4}"
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
  SAVE_FREQ="${SAVE_FREQ:-50}"
  TEST_FREQ="${TEST_FREQ:-50}"
  VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
fi
export CUDA_VISIBLE_DEVICES

TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_LOGPROB_MICRO_BATCH}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_LOGPROB_MICRO_BATCH}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"
SEARCH_TOPK="${SEARCH_TOPK:-3}"
SEARCH_TIMEOUT="${SEARCH_TIMEOUT:-60}"
SEARCH_LOG_REQUESTS="${SEARCH_LOG_REQUESTS:-false}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
SEED="${SEED:-0}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"
VAL_TOP_K="${VAL_TOP_K:--1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-false}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"
INVALID_ACTION_PENALTY_COEF="${INVALID_ACTION_PENALTY_COEF:-0.01}"
SIMILARITY_THRESH="${SIMILARITY_THRESH:-$DEFAULT_SIMILARITY_THRESH}"

OMEGA_ENV="${OMEGA_ENV:-0.8}"
OMEGA_FCSR="${OMEGA_FCSR:-0.08}"
LAMBDA_FALLBACK="${LAMBDA_FALLBACK:-0.02}"
DENSE_REWARD_WEIGHT="${DENSE_REWARD_WEIGHT:-0.10}"
HBRR_REWARD_WEIGHT="${HBRR_REWARD_WEIGHT:-0.03}"
TERMINAL_REWARD_WEIGHT="${TERMINAL_REWARD_WEIGHT:-1.0}"
REWARD_CLIP_MIN="${REWARD_CLIP_MIN:--0.2}"
REWARD_CLIP_MAX="${REWARD_CLIP_MAX:-1.0}"
PENALTY_INVALID="${PENALTY_INVALID:-0.15}"
PENALTY_NOOP="${PENALTY_NOOP:-0.08}"
PENALTY_REPEAT="${PENALTY_REPEAT:-0.12}"
PENALTY_PREMATURE="${PENALTY_PREMATURE:-0.20}"
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.01}"
FORMAT_REWARD_MAX="${FORMAT_REWARD_MAX:-0.02}"

case "$MODE" in
  gigpo)
    MODE_ARGS=(algorithm.adv_estimator=gigpo ++reward_method=rewardflow algorithm.gigpo.step_advantage_w=1.0)
    ;;
  dcsr)
    MODE_ARGS=(algorithm.adv_estimator=fcsr_gigpo ++reward_method=dcsr ++use_failure_rewriting=false ++dense_reward_weight=1.0 ++hbrr_reward_weight=0.0)
    ;;
  hbrr)
    MODE_ARGS=(algorithm.adv_estimator=fcsr_gigpo ++reward_method=fcsr_lite ++use_failure_rewriting=true ++dense_reward_weight=0.0 ++hbrr_reward_weight="$HBRR_REWARD_WEIGHT")
    ;;
  fgpo)
    MODE_ARGS=(algorithm.adv_estimator=fcsr_gigpo ++reward_method=fcsr_lite ++use_failure_rewriting=true ++dense_reward_weight="$DENSE_REWARD_WEIGHT" ++hbrr_reward_weight="$HBRR_REWARD_WEIGHT")
    ;;
  *) echo "[error] MODE must be fgpo, gigpo, dcsr, or hbrr" >&2; exit 2 ;;
esac

test -x "$PYTHON_BIN" || { echo "[error] missing Python: $PYTHON_BIN" >&2; exit 1; }
test -d "$MODEL_PATH" || { echo "[error] missing local model: $MODEL_PATH" >&2; exit 1; }
test -s "$MODEL_PATH/config.json" || { echo "[error] model is missing config.json: $MODEL_PATH" >&2; exit 1; }
test -s "$MODEL_PATH/tokenizer_config.json" || { echo "[error] model is missing tokenizer_config.json: $MODEL_PATH" >&2; exit 1; }
shopt -s nullglob
MODEL_WEIGHT_FILES=("$MODEL_PATH"/*.safetensors "$MODEL_PATH"/pytorch_model*.bin)
shopt -u nullglob
[[ ${#MODEL_WEIGHT_FILES[@]} -gt 0 ]] || { echo "[error] model has no local weight files: $MODEL_PATH" >&2; exit 1; }
test -s "$TRAIN_FILE" || { echo "[error] missing train parquet: $TRAIN_FILE" >&2; exit 1; }
test -s "$VAL_FILE" || {
  echo "[error] missing validation parquet for DATASET=$DATASET: $VAL_FILE" >&2
  echo "[hint] use DATASET=all for the official full test parquet or materialize the requested split under EVAL_ROOT" >&2
  exit 1
}

"$PYTHON_BIN" - "$MODE" "$DENSE_REWARD_WEIGHT" "$HBRR_REWARD_WEIGHT" "$TERMINAL_REWARD_WEIGHT" \
  "$N_GPUS_PER_NODE" "$TRAIN_BATCH_SIZE" "$VAL_BATCH_SIZE" "$GROUP_SIZE" "$PPO_MINI_BATCH_SIZE" \
  "$SEARCH_TOPK" "$SEARCH_TIMEOUT" "$TOTAL_EPOCHS" "$MAX_STEPS" "$SIMILARITY_THRESH" <<'PY'
import math
import sys

(
    mode,
    dense,
    hbrr,
    terminal,
    n_gpus,
    train_batch,
    val_batch,
    group_size,
    ppo_mini_batch,
    search_topk,
    search_timeout,
    total_epochs,
    max_steps,
    similarity_threshold,
) = sys.argv[1:]
floats = {
    "DENSE_REWARD_WEIGHT": dense,
    "HBRR_REWARD_WEIGHT": hbrr,
    "TERMINAL_REWARD_WEIGHT": terminal,
    "SIMILARITY_THRESH": similarity_threshold,
}
parsed = {}
for name, value in floats.items():
    parsed[name] = float(value)
    if not math.isfinite(parsed[name]):
        raise SystemExit(f"{name} must be finite, got {value}")
for name, value in {
    "N_GPUS_PER_NODE": n_gpus,
    "TRAIN_BATCH_SIZE": train_batch,
    "VAL_BATCH_SIZE": val_batch,
    "GROUP_SIZE": group_size,
    "PPO_MINI_BATCH_SIZE": ppo_mini_batch,
    "SEARCH_TOPK": search_topk,
    "SEARCH_TIMEOUT": search_timeout,
    "TOTAL_EPOCHS": total_epochs,
    "MAX_STEPS": max_steps,
}.items():
    if not value.isdigit() or int(value) <= 0:
        raise SystemExit(f"{name} must be a positive integer, got {value}")
if int(train_batch) % int(n_gpus):
    raise SystemExit("TRAIN_BATCH_SIZE must be divisible by N_GPUS_PER_NODE")
if int(ppo_mini_batch) > int(train_batch) * int(group_size):
    raise SystemExit("PPO_MINI_BATCH_SIZE cannot exceed TRAIN_BATCH_SIZE * GROUP_SIZE")
if not 0.0 <= parsed["SIMILARITY_THRESH"] <= 1.0:
    raise SystemExit("SIMILARITY_THRESH must be in [0, 1]")
if mode == "fgpo" and (parsed["DENSE_REWARD_WEIGHT"] <= 0 or parsed["HBRR_REWARD_WEIGHT"] <= 0):
    raise SystemExit("MODE=fgpo requires positive DENSE_REWARD_WEIGHT and HBRR_REWARD_WEIGHT")
if mode == "fgpo" and parsed["TERMINAL_REWARD_WEIGHT"] <= 0:
    raise SystemExit("MODE=fgpo requires positive TERMINAL_REWARD_WEIGHT")
PY

TRAIN_VALIDATE_ARGS=(--input "$TRAIN_FILE" --validate-only)
VAL_VALIDATE_ARGS=(--input "$VAL_FILE" --validate-only --against "$TRAIN_FILE")
[[ "$TRAIN_FILE" == "$BASELINE_TRAIN_FILE" ]] && TRAIN_VALIDATE_ARGS+=(--expect-baseline-split train)
if [[ "$TRAIN_FILE" == "$BASELINE_TRAIN_FILE" && ( "$VAL_FILE" == "$BASELINE_VAL_FILE" || "$VAL_FILE" == "$EVAL_ROOT/"* ) ]]; then
  VAL_VALIDATE_ARGS+=(--allow-question-overlap)
fi
if [[ "$VAL_FILE" == "$BASELINE_VAL_FILE" ]]; then
  VAL_VALIDATE_ARGS+=(--expect-baseline-split test)
fi
if [[ "$PROFILE" != "smoke" ]]; then
  case "$DATASET" in
    nq|triviaqa|popqa|hotpotqa|2wiki|musique|bamboogle)
      VAL_VALIDATE_ARGS+=(--expect-eval-source "$DATASET")
      ;;
  esac
fi
"$PYTHON_BIN" -m examples.data_preprocess.prepare_searchqa_data "${TRAIN_VALIDATE_ARGS[@]}"
"$PYTHON_BIN" -m examples.data_preprocess.prepare_searchqa_data "${VAL_VALIDATE_ARGS[@]}"

IFS=',' read -r -a SEARCH_URLS <<< "$SEARCH_URL"
for i in "${!SEARCH_URLS[@]}"; do SEARCH_URLS[$i]="$(echo "${SEARCH_URLS[$i]}" | xargs)"; done
if [[ ${#SEARCH_URLS[@]} -eq 1 ]]; then
  SEARCH_URL_OVERRIDE="env.search.search_url=${SEARCH_URLS[0]}"
else
  SEARCH_URL_JSON="$("$PYTHON_BIN" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${SEARCH_URLS[@]}")"
  SEARCH_URL_OVERRIDE="env.search.search_url=$SEARCH_URL_JSON"
fi

preflight_retriever() {
  local url response start_ns elapsed_ms schema_summary
  command -v curl >/dev/null 2>&1 || { echo "[error] curl is required" >&2; return 1; }
  for url in "${SEARCH_URLS[@]}"; do
    start_ns="$(date +%s%N)"
    response="$(curl -fsS --max-time "$SEARCH_TIMEOUT" -H 'Content-Type: application/json' --data "{\"query\":\"Who wrote Hamlet?\",\"topk\":$SEARCH_TOPK,\"return_scores\":true}" "$url")" || {
      echo "[error] retriever is unreachable: $url" >&2
      return 1
    }
    elapsed_ms="$(( ($(date +%s%N) - start_ns) / 1000000 ))"
    schema_summary="$(RETRIEVER_RESPONSE="$response" EXPECTED_TOPK="$SEARCH_TOPK" "$PYTHON_BIN" -c '
import json, math, os
result = json.loads(os.environ["RETRIEVER_RESPONSE"]).get("result")
if not isinstance(result, list) or not result:
    raise SystemExit("retriever response has no result list")
docs = result[0] if isinstance(result[0], list) else result
expected_topk = int(os.environ["EXPECTED_TOPK"])
if len(docs) != expected_topk:
    raise SystemExit(f"retriever returned {len(docs)} documents, expected {expected_topk}")
for item in docs:
    if not isinstance(item, dict):
        raise SystemExit("retriever document entry is not an object")
    document = item.get("document", item)
    if not isinstance(document, dict) or not str(document.get("contents", "")).strip():
        raise SystemExit("retriever document is missing non-empty document.contents")
    score = item.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise SystemExit("retriever document is missing a finite score")
print(f"documents={len(docs)}")
')"
    echo "[ok] retriever preflight: url=$url topk=$SEARCH_TOPK latency_ms=$elapsed_ms $schema_summary"
  done
}

preflight_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "[error] nvidia-smi not found" >&2; return 1; }
  local gpu_csv
  gpu_csv="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)"
  GPU_CSV="$gpu_csv" SELECTED_GPUS="$CUDA_VISIBLE_DEVICES" MAX_USED_MIB="$FREE_GPU_MAX_USED_MIB" EXPECTED_GPUS="$N_GPUS_PER_NODE" "$PYTHON_BIN" -c '
import os
rows = {}
for line in os.environ["GPU_CSV"].splitlines():
    idx, used = [part.strip() for part in line.split(",", 1)]
    rows[idx] = float(used)
selected = [item.strip() for item in os.environ["SELECTED_GPUS"].split(",") if item.strip()]
expected = int(os.environ["EXPECTED_GPUS"])
if len(selected) != expected:
    raise SystemExit(f"CUDA_VISIBLE_DEVICES has {len(selected)} GPUs but N_GPUS_PER_NODE={expected}")
if len(set(selected)) != len(selected):
    raise SystemExit(f"CUDA_VISIBLE_DEVICES contains duplicate GPU IDs: {selected}")
limit = float(os.environ["MAX_USED_MIB"])
busy = {idx: rows.get(idx, float("inf")) for idx in selected if rows.get(idx, float("inf")) > limit}
if busy:
    raise SystemExit(f"selected GPUs are busy above {limit:.0f} MiB: {busy}")
'
}

DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != "1" ]]; then
  [[ "${SKIP_RETRIEVER_CHECK:-0}" == "1" ]] || preflight_retriever
  [[ "${REQUIRE_FREE_GPUS:-1}" == "1" ]] && preflight_gpus
fi

PROJECT_NAME="${PROJECT_NAME:-verl_agent_search}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-search_${MODE}_qwen25_${MODEL_SIZE}_${PROFILE}_${DATASET}_s${SEED}_${RUN_STAMP}}"
CKPT_DIR_OVERRIDE="${CKPT_DIR:-}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints/$PROJECT_NAME/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/outputs/search_fcsr_runs/$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train.log}"
mkdir -p "$CKPT_DIR" "$LOG_DIR"
export WANDB_DIR="${WANDB_DIR:-$LOG_DIR/wandb}"
mkdir -p "$WANDB_DIR"

RESUME_ARGS=(trainer.resume_mode=disable)
if [[ -n "${RESUME_STEP:-}" && -z "${RESUME_CKPT:-}" && -z "${RESUME_FROM_PATH:-}" ]]; then
  [[ -n "$CKPT_DIR_OVERRIDE" ]] || {
    echo "[error] RESUME_STEP requires an explicit CKPT_DIR from the original run" >&2
    exit 1
  }
  RESUME_CKPT="$CKPT_DIR/global_step_$RESUME_STEP"
fi
if [[ -n "${RESUME_CKPT:-}" || -n "${RESUME_FROM_PATH:-}" ]]; then
  RESUME_FROM_PATH="${RESUME_FROM_PATH:-$RESUME_CKPT}"
  test -d "$RESUME_FROM_PATH" || { echo "[error] missing checkpoint: $RESUME_FROM_PATH" >&2; exit 1; }
  [[ "$(basename "$RESUME_FROM_PATH")" =~ ^global_step_[0-9]+$ ]] || {
    echo "[error] checkpoint path must end in global_step_<N>: $RESUME_FROM_PATH" >&2
    exit 1
  }
  test -s "$RESUME_FROM_PATH/data.pt" || { echo "[error] checkpoint is missing non-empty data.pt: $RESUME_FROM_PATH" >&2; exit 1; }
  test -d "$RESUME_FROM_PATH/actor" || { echo "[error] checkpoint is missing actor/: $RESUME_FROM_PATH" >&2; exit 1; }
  find "$RESUME_FROM_PATH/actor" -type f -print -quit | grep -q . || {
    echo "[error] checkpoint actor/ contains no files: $RESUME_FROM_PATH" >&2
    exit 1
  }
  RESUME_ARGS=(trainer.resume_mode=resume_path trainer.resume_from_path="$RESUME_FROM_PATH")
fi

VAL_ONLY=false
[[ "$PROFILE" == "eval" ]] && VAL_ONLY=true
LOGGER="${LOGGER:-['console','wandb']}"

CMD=(
  "$PYTHON_BIN" -m verl.trainer.main_ppo
  "${MODE_ARGS[@]}"
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.val_batch_size="$VAL_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=true
  data.truncation=left
  data.return_raw_chat=true
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1
  actor_rollout_ref.actor.optim.weight_decay=0.01
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.grad_clip=1.0
  actor_rollout_ref.actor.clip_ratio="$CLIP_RATIO"
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF"
  actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW"
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH"
  actor_rollout_ref.actor.clip_ratio_c=3.0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.use_invalid_action_penalty=true
  actor_rollout_ref.actor.invalid_action_penalty_coef="$INVALID_ACTION_PENALTY_COEF"
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.rollout.name="$ENGINE"
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE"
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE"
  actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P"
  actor_rollout_ref.rollout.top_k="$ROLLOUT_TOP_K"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.rollout.enable_chunked_prefill=false
  actor_rollout_ref.rollout.enforce_eager=false
  actor_rollout_ref.rollout.free_cache_engine=false
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE"
  actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P"
  actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K"
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  algorithm.use_kl_in_reward=false
  algorithm.gamma=0.95
  algorithm.gigpo.mode=mean_std_norm
  algorithm.gigpo.enable_similarity=true
  algorithm.gigpo.similarity_thresh="$SIMILARITY_THRESH"
  ++algorithm.fcsr_gigpo.gamma_env=0.95
  ++algorithm.fcsr_gigpo.gamma_fcsr=0.95
  ++algorithm.fcsr_gigpo.omega_env="$OMEGA_ENV"
  ++algorithm.fcsr_gigpo.omega_fcsr="$OMEGA_FCSR"
  ++algorithm.fcsr_gigpo.lambda_fallback="$LAMBDA_FALLBACK"
  ++algorithm.fcsr_gigpo.norm_type=mean_std_norm
  ++algorithm.fcsr_gigpo.adv_clip=5.0
  ++algorithm.fcsr_gigpo.min_anchor_group_size=2
  ++algorithm.fcsr_gigpo.min_distinct_actions=2
  ++algorithm.fcsr_gigpo.enable_similarity=true
  ++algorithm.fcsr_gigpo.similarity_thresh="$SIMILARITY_THRESH"
  ++algorithm.fcsr_gigpo.use_env_step_advantage=true
  ++algorithm.fcsr_gigpo.use_fcsr_step_residual=true
  ++algorithm.fcsr_gigpo.use_fcsr_non_anchor_fallback=true
  ++algorithm.fcsr_gigpo.webshop_state_key_include_recovered_options=false
  ++algorithm.filter_groups.enable=false
  ++use_diagnostics=true
  ++use_information_potential=false
  ++use_action_format_reward=true
  ++format_reward_apply=terminal
  ++format_reward_weight="$FORMAT_REWARD_WEIGHT"
  ++format_reward_max="$FORMAT_REWARD_MAX"
  ++format_reward_require_not_invalid=true
  ++reward_normalization=none
  ++reward_clip_min="$REWARD_CLIP_MIN"
  ++reward_clip_max="$REWARD_CLIP_MAX"
  ++terminal_reward_weight="$TERMINAL_REWARD_WEIGHT"
  ++penalty_invalid="$PENALTY_INVALID"
  ++penalty_noop="$PENALTY_NOOP"
  ++penalty_repeat="$PENALTY_REPEAT"
  ++penalty_premature="$PENALTY_PREMATURE"
  ++dcsr_alpha_progress=0.8
  ++dcsr_beta_validity=0.05
  ++dcsr_eta_recovery=0.10
  ++dcsr_lambda_risk=0.5
  ++hbrr_alpha_delta_q=0.3
  ++hbrr_beta_progress=0.05
  ++hbrr_gamma_exploratory=0.08
  ++hbrr_eta_recovery=0.10
  ++hbrr_bottleneck_penalty=0.25
  ++hbrr_irreversible_penalty=0.40
  ++hbrr_redundant_penalty=0.12
  ++hbrr_premature_penalty=0.25
  ++monitor_kl_estimators=true
  ++kl_monitor_interval=1
  env.env_name=search
  env.seed="$SEED"
  env.max_steps="$MAX_STEPS"
  env.history_length="$HISTORY_LENGTH"
  env.rollout.n="$GROUP_SIZE"
  env.search.topk="$SEARCH_TOPK"
  env.search.timeout="$SEARCH_TIMEOUT"
  env.search.log_requests="$SEARCH_LOG_REQUESTS"
  "$SEARCH_URL_OVERRIDE"
  trainer.critic_warmup=0
  trainer.logger="$LOGGER"
  trainer.project_name="$PROJECT_NAME"
  trainer.experiment_name="$RUN_NAME"
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
  trainer.nnodes=1
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.total_epochs="$TOTAL_EPOCHS"
  trainer.val_before_train="$VAL_BEFORE_TRAIN"
  trainer.val_only="$VAL_ONLY"
  trainer.default_local_dir="$CKPT_DIR"
  "${RESUME_ARGS[@]}"
  ++ray_init.address=local
  "${HYDRA_EXTRA[@]}"
)

{
  echo "REPO_ROOT=$REPO_ROOT"
  echo "ENGINE=$ENGINE MODE=$MODE PROFILE=$PROFILE"
  echo "MODEL_SIZE=$MODEL_SIZE MODEL_PATH=$MODEL_PATH"
  echo "DATASET=$DATASET TRAIN_FILE=$TRAIN_FILE VAL_FILE=$VAL_FILE"
  echo "SEARCH_URLS=${SEARCH_URLS[*]} SEARCH_TOPK=$SEARCH_TOPK SEARCH_TIMEOUT=$SEARCH_TIMEOUT SEARCH_LOG_REQUESTS=$SEARCH_LOG_REQUESTS"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES N_GPUS_PER_NODE=$N_GPUS_PER_NODE TP_SIZE=$TP_SIZE"
  echo "SHARED_RETRIEVER=$SHARED_RETRIEVER FREE_GPU_MAX_USED_MIB=$FREE_GPU_MAX_USED_MIB"
  echo "TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE VAL_BATCH_SIZE=$VAL_BATCH_SIZE GROUP_SIZE=$GROUP_SIZE"
  echo "TOTAL_EPOCHS=$TOTAL_EPOCHS MAX_STEPS=$MAX_STEPS MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
  echo "PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE SAVE_FREQ=$SAVE_FREQ TEST_FREQ=$TEST_FREQ"
  echo "ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE ROLLOUT_TOP_P=$ROLLOUT_TOP_P ROLLOUT_TOP_K=$ROLLOUT_TOP_K"
  echo "VAL_TEMPERATURE=$VAL_TEMPERATURE VAL_TOP_P=$VAL_TOP_P VAL_TOP_K=$VAL_TOP_K VAL_DO_SAMPLE=$VAL_DO_SAMPLE"
  echo "KL_LOSS_COEF=$KL_LOSS_COEF ENTROPY_COEFF=$ENTROPY_COEFF CLIP_RATIO=$CLIP_RATIO SIMILARITY_THRESH=$SIMILARITY_THRESH"
  echo "INVALID_ACTION_PENALTY_COEF=$INVALID_ACTION_PENALTY_COEF TERMINAL_REWARD_WEIGHT=$TERMINAL_REWARD_WEIGHT"
  echo "PPO_MICRO_BATCH_SIZE_PER_GPU=$PPO_MICRO_BATCH_SIZE_PER_GPU ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
  echo "ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION"
  echo "DENSE_REWARD_WEIGHT=$DENSE_REWARD_WEIGHT HBRR_REWARD_WEIGHT=$HBRR_REWARD_WEIGHT"
  echo "OMEGA_ENV=$OMEGA_ENV OMEGA_FCSR=$OMEGA_FCSR LAMBDA_FALLBACK=$LAMBDA_FALLBACK"
  echo "RUN_NAME=$RUN_NAME CKPT_DIR=$CKPT_DIR LOG_FILE=$LOG_FILE WANDB_DIR=$WANDB_DIR"
  printf 'RESOLVED_COMMAND:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
} | tee -a "$LOG_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

if [[ "$PROFILE" == "eval" ]]; then
  EVAL_REQUIRE_ARGS=()
  [[ "$DATASET" == "all" ]] && EVAL_REQUIRE_ARGS+=(--require-all-seven)
  "$PYTHON_BIN" -m agent_system.evaluation.searchqa_eval \
    --log "$LOG_FILE" \
    --model-size "$MODEL_SIZE" \
    --json-output "$LOG_DIR/searchqa_eval_summary.json" \
    --csv-output "$LOG_DIR/searchqa_eval_summary.csv" \
    --latex-output "$LOG_DIR/searchqa_eval_summary.tex" \
    "${EVAL_REQUIRE_ARGS[@]}"
fi
