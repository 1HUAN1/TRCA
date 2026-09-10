#!/usr/bin/env bash
set -euo pipefail

ENGINE=${1:-vllm}
MODE=${2:-fcsr}
HYDRA_EXTRA=("${@:3}")

if [[ "$ENGINE" != "vllm" ]]; then
  echo "Only vllm engine is supported by this AlfWorld GiGPO+FCSR launcher." >&2
  exit 2
fi

if [[ "$MODE" != "fcsr" ]]; then
  echo "Only fcsr mode is allowed here. Refusing MODE=$MODE to avoid RewardFlow/pure baseline drift." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
cd "$PROJECT_ROOT"

export PYTHON_BIN=${PYTHON_BIN:-/root/data/conda/envs/verl-agent/bin/python}
export RAY_BIN=${RAY_BIN:-/root/data/conda/envs/verl-agent/bin/ray}
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/agent_system/environments/env_package/alfworld:${PYTHONPATH:-}"

export MODEL_PATH=${MODEL_PATH:-/root/data/zh/huggingface/Qwen2.5-1.5B-Instruct}
export DATA_DIR=${DATA_DIR:-/root/data/verl-agent/text}
export TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
export VAL_FILE=${VAL_FILE:-$DATA_DIR/test.parquet}

# AlfWorld success-first recipe. Defaults are conservative, but the tmux wrapper
# can override these for smoke runs without touching WebShop.
export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
export PREP_VAL_DATA_SIZE=$VAL_DATA_SIZE
export GROUP_SIZE=${GROUP_SIZE:-8}

export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export TP_SIZE=${TP_SIZE:-2}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.1}
export ALFWORLD_MAX_STEPS=${ALFWORLD_MAX_STEPS:-25}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-300}
export SAVE_FREQ=${SAVE_FREQ:-30}
export TEST_FREQ=${TEST_FREQ:-10}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

export RUN_NAME=${RUN_NAME:-alfworld_fcsr_gigpo_qwen25_1p5b_from0_to300_$(date +%Y%m%d_%H%M%S)}
export CKPT_DIR=${CKPT_DIR:-$PROJECT_ROOT/checkpoints/verl_agent_alfworld/$RUN_NAME}
export LOG_DIR=${LOG_DIR:-$PROJECT_ROOT/outputs/alfworld_fcsr_gigpo_runs/$RUN_NAME}
export WANDB_MODE=${WANDB_MODE:-offline}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export MONITOR_KL_ESTIMATORS=${MONITOR_KL_ESTIMATORS:-true}
export KL_MONITOR_INTERVAL=${KL_MONITOR_INTERVAL:-1}
unset PYTORCH_CUDA_ALLOC_CONF

export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-32}}
export ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.4}
export REWARD_CLIP_MIN=${REWARD_CLIP_MIN:-0.0}
export REWARD_CLIP_MAX=${REWARD_CLIP_MAX:-1.0}
export ALFWORLD_ACTION_GATE_ENABLED=${ALFWORLD_ACTION_GATE_ENABLED:-false}
export ALFWORLD_PROMPT_STYLE=${ALFWORLD_PROMPT_STYLE:-think_action}
export ALFWORLD_ALLOW_BARE_ADMISSIBLE=${ALFWORLD_ALLOW_BARE_ADMISSIBLE:-false}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
export CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
export ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}
export ACTOR_INVALID_ACTION_PENALTY_COEF=${ACTOR_INVALID_ACTION_PENALTY_COEF:-0.1}
export OMEGA_ENV=${OMEGA_ENV:-0.8}
export OMEGA_FCSR=${OMEGA_FCSR:-1.0}
export LAMBDA_FALLBACK=${LAMBDA_FALLBACK:-0.02}
export FCSR_NORM_TYPE=${FCSR_NORM_TYPE:-none}
export DENSE_REWARD_WEIGHT=${DENSE_REWARD_WEIGHT:-0.10}
export HBRR_REWARD_WEIGHT=${HBRR_REWARD_WEIGHT:-0.05}
export PENALTY_INVALID=${PENALTY_INVALID:-0.2}
export PENALTY_NOOP=${PENALTY_NOOP:-0.1}
export PENALTY_REPEAT=${PENALTY_REPEAT:-0.1}
export ALFWORLD_GATE_INVALID_CLAMP=${ALFWORLD_GATE_INVALID_CLAMP:--0.2}
export ALFWORLD_GATE_NOOP_REPEAT_CLAMP=${ALFWORLD_GATE_NOOP_REPEAT_CLAMP:--0.1}
export HBRR_REDUNDANT_PENALTY=${HBRR_REDUNDANT_PENALTY:-0.15}
export HBRR_BOTTLENECK_PENALTY=${HBRR_BOTTLENECK_PENALTY:-0.4}
export ALFWORLD_DEBUG_DUMP=${ALFWORLD_DEBUG_DUMP:-true}
export ALFWORLD_DEBUG_DUMP_DIR=${ALFWORLD_DEBUG_DUMP_DIR:-$LOG_DIR/alfworld_debug_samples}
export ALFWORLD_DEBUG_DUMP_INTERVAL=${ALFWORLD_DEBUG_DUMP_INTERVAL:-1}
export ALFWORLD_DEBUG_DUMP_MAX_SAMPLES=${ALFWORLD_DEBUG_DUMP_MAX_SAMPLES:-20}

unset RESUME_STEP
unset RESUME_CKPT
unset RESUME_FROM_PATH
unset WANDB_RUN_ID
unset WANDB_RESUME
unset RAY_ADDRESS

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  echo "Unsetting PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF because vLLM CuMemAllocator is incompatible with expandable_segments:True" >&2
  unset PYTORCH_CUDA_ALLOC_CONF
fi

mkdir -p "$LOG_DIR" "$CKPT_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing MODEL_PATH: $MODEL_PATH" >&2
  exit 1
fi

if [[ -z "${ALFWORLD_DATA:-}" ]]; then
  if [[ -d /root/data/.cache/alfworld ]]; then
    export ALFWORLD_DATA=/root/data/.cache/alfworld
  else
    export ALFWORLD_DATA=/root/.cache/alfworld
  fi
fi

if [[ ! -d "$ALFWORLD_DATA" ]]; then
  echo "Missing ALFWORLD_DATA: $ALFWORLD_DATA" >&2
  echo "Expected data under /root/data/.cache/alfworld or /root/.cache/alfworld" >&2
  exit 1
fi

TRAIN_GAME_COUNT=$(find "$ALFWORLD_DATA/json_2.1.1/train" -name 'game.tw-pddl' 2>/dev/null | wc -l | tr -d ' ')
VALID_GAME_COUNT=$(find "$ALFWORLD_DATA/json_2.1.1/valid_seen" -name 'game.tw-pddl' 2>/dev/null | wc -l | tr -d ' ')
if [[ "${TRAIN_GAME_COUNT:-0}" == "0" || "${VALID_GAME_COUNT:-0}" == "0" ]]; then
  echo "AlfWorld game files not found under ALFWORLD_DATA=$ALFWORLD_DATA" >&2
  echo "train game.tw-pddl count: ${TRAIN_GAME_COUNT:-0}" >&2
  echo "valid_seen game.tw-pddl count: ${VALID_GAME_COUNT:-0}" >&2
  echo "The environment would initialize with 0 games; fix ALFWORLD_DATA before training." >&2
  exit 1
fi

echo "ALFWORLD_DATA=$ALFWORLD_DATA"
echo "AlfWorld train games: $TRAIN_GAME_COUNT valid_seen games: $VALID_GAME_COUNT"

PREPARE_MODE=${PREPARE_ALFWORLD_DATA:-auto}
NEED_PREPARE=0
if [[ "$PREPARE_MODE" == "1" || "$PREPARE_MODE" == "true" ]]; then
  NEED_PREPARE=1
elif [[ "$PREPARE_MODE" == "auto" && ( ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ) ]]; then
  NEED_PREPARE=1
fi

if [[ "$NEED_PREPARE" == "1" ]]; then
  PREPROCESS_SCRIPT=${ALFWORLD_PREPROCESS_SCRIPT:-examples/data_preprocess/alfworld.py}
  if [[ ! -f "$PREPROCESS_SCRIPT" ]]; then
    echo "Missing AlfWorld preprocess script: $PREPROCESS_SCRIPT" >&2
    echo "Set PREPARE_ALFWORLD_DATA=0 if parquet files already exist." >&2
    exit 1
  fi
  "$PYTHON_BIN" "$PREPROCESS_SCRIPT" \
    --local_dir "$DATA_DIR" \
    --train_size "$TRAIN_DATA_SIZE" \
    --val_size "$PREP_VAL_DATA_SIZE"
fi

if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  echo "Missing AlfWorld parquet data: $TRAIN_FILE or $VAL_FILE" >&2
  exit 1
fi

if [[ "${RESET_RAY:-0}" == "1" ]]; then
  "$RAY_BIN" stop --force || true
fi

LOGGER=${LOGGER:-"['console','wandb']"}

if [[ "${WANDB_MODE:-}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ ! -f "$HOME/.netrc" ]] || ! grep -q "api.wandb.ai" "$HOME/.netrc"; then
    echo "WARNING: WANDB_MODE=online but no WANDB_API_KEY or ~/.netrc login was detected." >&2
    echo "Run 'wandb login' once, or export WANDB_API_KEY before launching." >&2
  fi
fi

exec "$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=fcsr_gigpo \
  algorithm.gigpo.mode=mean_std_norm \
  algorithm.gamma=0.95 \
  ++algorithm.fcsr_gigpo.gamma_env=0.95 \
  ++algorithm.fcsr_gigpo.gamma_fcsr=0.95 \
  ++algorithm.fcsr_gigpo.omega_env="$OMEGA_ENV" \
  ++algorithm.fcsr_gigpo.omega_fcsr="$OMEGA_FCSR" \
  ++algorithm.fcsr_gigpo.lambda_fallback="$LAMBDA_FALLBACK" \
  ++algorithm.fcsr_gigpo.norm_type="$FCSR_NORM_TYPE" \
  ++algorithm.fcsr_gigpo.adv_clip=5.0 \
  ++algorithm.fcsr_gigpo.min_anchor_group_size=2 \
  ++algorithm.fcsr_gigpo.min_distinct_actions=2 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_DATA_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=2048 \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef="$ACTOR_INVALID_ACTION_PENALTY_COEF" \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=2 \
  env.max_steps="$ALFWORLD_MAX_STEPS" \
  env.rollout.n="$GROUP_SIZE" \
  env.resources_per_worker.num_cpus="$NUM_CPUS_PER_ENV_WORKER" \
  ++env.alfworld.parallel_backend=ray \
  ++env.alfworld.ray_get_timeout_s=0 \
  ++env.alfworld.prompt_style="$ALFWORLD_PROMPT_STYLE" \
  ++env.alfworld.allow_bare_admissible="$ALFWORLD_ALLOW_BARE_ADMISSIBLE" \
  ++env.alfworld.debug_dump="$ALFWORLD_DEBUG_DUMP" \
  ++env.alfworld.debug_dump_dir="$ALFWORLD_DEBUG_DUMP_DIR" \
  ++env.alfworld.debug_dump_interval="$ALFWORLD_DEBUG_DUMP_INTERVAL" \
  ++env.alfworld.debug_dump_max_samples="$ALFWORLD_DEBUG_DUMP_MAX_SAMPLES" \
  ++reward_method=fcsr_lite \
  ++use_diagnostics=true \
  ++use_failure_rewriting=true \
  ++use_information_potential=false \
  ++use_action_format_reward=true \
  ++format_reward_apply=terminal \
  ++format_reward_weight=0.03 \
  ++format_reward_max=0.05 \
  ++format_reward_require_not_invalid=true \
  ++reward_normalization=none \
  ++reward_clip_min="$REWARD_CLIP_MIN" \
  ++reward_clip_max="$REWARD_CLIP_MAX" \
  ++dense_reward_weight="$DENSE_REWARD_WEIGHT" \
  ++hbrr_reward_weight="$HBRR_REWARD_WEIGHT" \
  ++iprr_reward_weight=0.0 \
  ++terminal_reward_weight=1.0 \
  ++penalty_invalid="$PENALTY_INVALID" \
  ++penalty_noop="$PENALTY_NOOP" \
  ++penalty_repeat="$PENALTY_REPEAT" \
  ++hbrr_redundant_penalty="$HBRR_REDUNDANT_PENALTY" \
  ++hbrr_bottleneck_penalty="$HBRR_BOTTLENECK_PENALTY" \
  ++alfworld_training_chain=standard \
  ++alfworld_progress_gated_reward=false \
  ++alfworld_action_gate_enabled="$ALFWORLD_ACTION_GATE_ENABLED" \
  ++alfworld_gate_invalid_clamp="$ALFWORLD_GATE_INVALID_CLAMP" \
  ++alfworld_gate_noop_repeat_clamp="$ALFWORLD_GATE_NOOP_REPEAT_CLAMP" \
  trainer.project_name=verl_agent_alfworld \
  trainer.experiment_name="$RUN_NAME" \
  trainer.logger="$LOGGER" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  ++ray_init.address=local \
  "${HYDRA_EXTRA[@]}"
