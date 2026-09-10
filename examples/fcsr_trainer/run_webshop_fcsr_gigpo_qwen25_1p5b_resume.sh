#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_1p5b.sh [ENGINE] [MODE] [KEY=VALUE ...] [hydra.override=...]

Examples:
  # Start from scratch.
  unset RESUME_STEP RESUME_CKPT RESUME_FROM_PATH RUN_NAME LOG_DIR LOG_FILE
  bash examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_1p5b.sh vllm fcsr TOTAL_EPOCHS=200

  # Resume from a checkpoint under CKPT_DIR.
  bash examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_1p5b.sh vllm fcsr RESUME_STEP=300 TOTAL_EPOCHS=450

  # Resume from an explicit checkpoint path.
  bash examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_1p5b.sh vllm fcsr \
    RESUME_CKPT=/path/to/global_step_300 TOTAL_EPOCHS=450
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

ENGINE="${ENGINE:-vllm}"
MODE="${MODE:-fcsr}"
HYDRA_EXTRA=()

if [[ $# -gt 0 && "$1" != *=* ]]; then
  if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
  fi
  ENGINE="$1"
  shift
fi

if [[ $# -gt 0 && "$1" != *=* ]]; then
  MODE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
  esac

  if [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    export "$1"
  else
    HYDRA_EXTRA+=("$1")
  fi
  shift
done

cd "${REPO_ROOT}"

# Machine/runtime settings.
CONDA_ENV="${CONDA_ENV:-/root/data/conda/envs/verl-agent-webshop}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/root/data/zh/huggingface/Qwen2.5-7B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-/root/data/verl-agent/text/train.parquet}"
VAL_FILE="${VAL_FILE:-/root/data/verl-agent/text/test.parquet}"

export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-online}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
ulimit -u "${ULIMIT_U:-65536}" || true

setup_java_runtime() {
  local conda_jdk="${CONDA_ENV_PREFIX}/lib/jvm"
  local candidate

  for candidate in "${JAVA_HOME:-}" "${conda_jdk}" /usr/lib/jvm/java-11-openjdk-amd64 /usr/lib/jvm/java-17-openjdk-amd64 /usr/lib/jvm/default-java; do
    if [[ -n "${candidate}" && -x "${candidate}/bin/javac" ]]; then
      export JAVA_HOME="${candidate}"
      export PATH="${JAVA_HOME}/bin:${PATH}"
      break
    fi
  done

  command -v javac >/dev/null 2>&1 || {
    echo "[error] javac not found. Set JAVA_HOME or install a JDK." >&2
    exit 1
  }

  for candidate in "${JVM_PATH:-}" "${JAVA_HOME:-}/lib/server/libjvm.so" "${conda_jdk}/lib/server/libjvm.so"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      export JVM_PATH="${candidate}"
      export LD_LIBRARY_PATH="$(dirname "${candidate}"):${LD_LIBRARY_PATH:-}"
      return
    fi
  done

  echo "[error] libjvm.so not found. Set JVM_PATH to the absolute libjvm.so path." >&2
  exit 1
}

setup_java_runtime

# Experiment settings. Keep these scalar and edit/override them directly.
PROJECT_NAME="${PROJECT_NAME:-verl_agent_webshop}"
RUN_NAME="${RUN_NAME:-webshop_fcsr_gigpo_qwen25_7b_g8_bs16}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/fcsr_gigpo_runs/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"

# Defaults for resuming the interrupted WebShop 7B curve.
RESUME_STEP="${RESUME_STEP-160}"
export KEEP_WANDB_RESUME="${KEEP_WANDB_RESUME-1}"
export WANDB_RUN_ID="${WANDB_RUN_ID-3yaskey8}"
export WANDB_RESUME="${WANDB_RESUME-must}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-300}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-10}"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "$(dirname "${TRAIN_FILE}")" "$(dirname "${VAL_FILE}")"

test -x "${PYTHON_BIN}" || { echo "[error] missing python: ${PYTHON_BIN}" >&2; exit 1; }
test -d "${MODEL_PATH}" || { echo "[error] missing model path: ${MODEL_PATH}" >&2; exit 1; }

if [[ "${PREPARE_DATA:-auto}" == "true" || ( "${PREPARE_DATA:-auto}" == "auto" && ( ! -s "${TRAIN_FILE}" || ! -s "${VAL_FILE}" ) ) ]]; then
  "${PYTHON_BIN}" -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 128
else
  echo "[skip] using existing local parquet data: $(dirname "${TRAIN_FILE}")"
fi

test -s "${TRAIN_FILE}" || { echo "[error] missing train parquet: ${TRAIN_FILE}" >&2; exit 1; }
test -s "${VAL_FILE}" || { echo "[error] missing val parquet: ${VAL_FILE}" >&2; exit 1; }

RESUME_ARGS=(trainer.resume_mode=disable)
if [[ -n "${RESUME_STEP:-}" && -z "${RESUME_CKPT:-}" && -z "${RESUME_FROM_PATH:-}" ]]; then
  RESUME_CKPT="${CKPT_DIR}/global_step_${RESUME_STEP}"
fi

if [[ -n "${RESUME_CKPT:-}" || -n "${RESUME_FROM_PATH:-}" ]]; then
  RESUME_FROM_PATH="${RESUME_FROM_PATH:-${RESUME_CKPT}}"
  test -d "${RESUME_FROM_PATH}" || { echo "[error] missing checkpoint: ${RESUME_FROM_PATH}" >&2; exit 1; }
  test -f "${RESUME_FROM_PATH}/data.pt" || { echo "[error] checkpoint incomplete, missing data.pt: ${RESUME_FROM_PATH}" >&2; exit 1; }
  RESUME_ARGS=(trainer.resume_mode=resume_path trainer.resume_from_path="${RESUME_FROM_PATH}")
fi

case "${MODE}" in
  pure)
    MODE_ARGS=(
      algorithm.adv_estimator=gigpo
      ++reward_method=rewardflow
      ++algorithm.gigpo.step_advantage_w=0.8
    )
    ;;
  fcsr)
    MODE_ARGS=(
      algorithm.adv_estimator=fcsr_gigpo
      ++reward_method=fcsr_lite
      ++algorithm.fcsr_gigpo.omega_fcsr=0.1
      ++algorithm.fcsr_gigpo.lambda_fallback=0.02
      ++algorithm.filter_groups.enable=false
    )
    ;;
  conservative)
    MODE_ARGS=(
      algorithm.adv_estimator=fcsr_gigpo
      ++reward_method=fcsr_lite
      ++algorithm.fcsr_gigpo.omega_fcsr=0.05
      ++algorithm.fcsr_gigpo.lambda_fallback=0.0
      ++algorithm.filter_groups.enable=false
    )
    ;;
  dynamic)
    MODE_ARGS=(
      algorithm.adv_estimator=fcsr_gigpo
      ++reward_method=fcsr_lite
      ++algorithm.fcsr_gigpo.omega_fcsr=0.1
      ++algorithm.fcsr_gigpo.lambda_fallback=0.02
      ++algorithm.filter_groups.enable=true
    )
    ;;
  *)
    echo "[error] unknown MODE=${MODE}. Use pure, fcsr, conservative, or dynamic." >&2
    exit 2
    ;;
esac

RAY_INIT_ARGS=()
if [[ "${USE_EXISTING_RAY:-0}" == "1" ]]; then
  if [[ -n "${RAY_ADDRESS:-}" ]]; then
    RAY_INIT_ARGS=(++ray_init.address="${RAY_ADDRESS}")
  fi
else
  unset RAY_ADDRESS
  RAY_INIT_ARGS=(++ray_init.address=local)
fi

if [[ "${RESET_RAY:-0}" == "1" ]] && command -v ray >/dev/null 2>&1; then
  ray stop --force || true
fi

if [[ "${KEEP_WANDB_RESUME:-0}" != "1" ]]; then
  unset WANDB_RUN_ID
  unset WANDB_RESUME
fi

{
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "JAVA_HOME=${JAVA_HOME}"
  echo "JVM_PATH=${JVM_PATH}"
  echo "RAY_ADDRESS=${RAY_ADDRESS:-<unset>}"
  echo "RAY_INIT_ADDRESS=${RAY_INIT_ARGS[*]:-auto}"
  echo "ENGINE=${ENGINE}"
  echo "MODE=${MODE}"
  echo "RUN_NAME=${RUN_NAME}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "CKPT_DIR=${CKPT_DIR}"
  echo "RESUME_STEP=${RESUME_STEP:-<none>}"
  echo "RESUME_FROM_PATH=${RESUME_FROM_PATH:-<none>}"
  echo "KEEP_WANDB_RESUME=${KEEP_WANDB_RESUME:-0}"
  echo "WANDB_RUN_ID=${WANDB_RUN_ID:-<unset>}"
  echo "WANDB_RESUME=${WANDB_RESUME:-<unset>}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "TRAIN_FILE=${TRAIN_FILE}"
  echo "VAL_FILE=${VAL_FILE}"
  echo "TOTAL_EPOCHS=${TOTAL_EPOCHS}"
  echo "SAVE_FREQ=${SAVE_FREQ}"
  echo "TEST_FREQ=${TEST_FREQ}"

  "${PYTHON_BIN}" -m verl.trainer.main_ppo \
    "${MODE_ARGS[@]}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.mode=mean_norm \
    ++algorithm.fcsr_gigpo.gamma_env=0.95 \
    ++algorithm.fcsr_gigpo.gamma_fcsr=0.95 \
    ++algorithm.fcsr_gigpo.omega_env=0.8 \
    ++algorithm.fcsr_gigpo.norm_type=none \
    ++algorithm.fcsr_gigpo.adv_clip=5.0 \
    ++algorithm.fcsr_gigpo.min_anchor_group_size=2 \
    ++algorithm.fcsr_gigpo.min_distinct_actions=2 \
    ++algorithm.fcsr_gigpo.enable_similarity=false \
    ++algorithm.fcsr_gigpo.similarity_thresh=0.90 \
    ++algorithm.fcsr_gigpo.use_env_step_advantage=true \
    ++algorithm.fcsr_gigpo.use_fcsr_step_residual=true \
    ++algorithm.fcsr_gigpo.use_fcsr_non_anchor_fallback=true \
    ++algorithm.fcsr_gigpo.webshop_state_key_include_recovered_options=true \
    ++algorithm.filter_groups.max_num_gen_batches=10 \
    ++use_diagnostics=true \
    ++use_information_potential=true \
    ++reward_clip_min=-0.2 \
    ++reward_clip_max=0.2 \
    ++fcsr_strict_clip_min=-0.2 \
    ++fcsr_strict_clip_max=0.2 \
    ++use_action_format_reward=true \
    ++format_reward_apply=terminal \
    ++format_reward_weight=0.03 \
    ++format_reward_max=0.05 \
    ++format_reward_require_not_invalid=true \
    ++monitor_kl_estimators=true \
    ++kl_monitor_interval=1 \
    ++kl_monitor_clip_exp_max=1000000.0 \
    ++kl_monitor_log_sequence_stats=true \
    ++kl_monitor_log_success_failure_split=true \
    ++webshop_strict_gate.enabled=true \
    ++webshop_strict_gate.allow_search_reward=false \
    ++webshop_strict_gate.allow_navigation_positive_reward=false \
    ++webshop_strict_gate.allow_generic_click_positive_reward=false \
    ++webshop_strict_gate.allow_goal_option_prefix=true \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.2 \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','wandb']" \
    trainer.project_name=verl_agent_webshop \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_before_train=False \
    trainer.default_local_dir="${CKPT_DIR}" \
    "${RESUME_ARGS[@]}" \
    "${RAY_INIT_ARGS[@]}" \
    "${HYDRA_EXTRA[@]}"
} 2>&1 | tee -a "${LOG_FILE}"
