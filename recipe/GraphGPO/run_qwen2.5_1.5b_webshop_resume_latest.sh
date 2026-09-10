#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/data/zh/gigpo+fcsr_verl
CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
CONDA_ENV=/root/data/conda/envs/verl-agent-webshop
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/verl_agent_webshop/graphgpo_qwen2.5_1.5b_webshop}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/graphgpo_qwen2.5_1.5b_webshop}"
SAVE_FREQ="${SAVE_FREQ:-65}"
ALLOW_FRESH_START="${ALLOW_FRESH_START:-0}"
ENGINE="${ENGINE:-vllm}"
WORLD_SIZE=4

if pgrep -f 'recipe\.GraphGPO\.main_graphgpo|recipe/GraphGPO/main_graphgpo\.py' >/dev/null 2>&1; then
  echo "Refusing to start: a GraphGPO training process is already running." >&2
  exit 2
fi

if [[ ! -r "$CONDA_SH" ]]; then
  echo "Missing conda activation script: $CONDA_SH" >&2
  exit 2
fi

for data_file in /root/data/verl-agent/text/train.parquet /root/data/verl-agent/text/test.parquet; do
  if [[ ! -s "$data_file" ]]; then
    echo "Missing training data: $data_file" >&2
    exit 2
  fi
done

mkdir -p "$CHECKPOINT_ROOT" "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/resume_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

latest_step=-1
latest_checkpoint=""
shopt -s nullglob
for candidate in "$CHECKPOINT_ROOT"/global_step_*; do
  [[ -d "$candidate" ]] || continue
  step="${candidate##*/global_step_}"
  [[ "$step" =~ ^[0-9]+$ ]] || continue
  if (( step > latest_step )); then
    latest_step=$step
    latest_checkpoint=$candidate
  fi
done

resume_args=()
if [[ -n "$latest_checkpoint" ]]; then
  actor_dir="$latest_checkpoint/actor"
  [[ -s "$latest_checkpoint/data.pt" ]] || { echo "Incomplete checkpoint: missing data.pt in $latest_checkpoint" >&2; exit 3; }
  [[ -d "$actor_dir" ]] || { echo "Incomplete checkpoint: missing actor directory in $latest_checkpoint" >&2; exit 3; }

  for prefix in model optim extra_state; do
    count=$(find "$actor_dir" -maxdepth 1 -type f -name "${prefix}_world_size_${WORLD_SIZE}_rank_*.pt" | wc -l)
    if [[ "$count" -ne "$WORLD_SIZE" ]]; then
      echo "Incomplete/incompatible checkpoint: expected ${WORLD_SIZE} ${prefix} shards, found ${count} in $actor_dir" >&2
      exit 3
    fi
  done

  echo "Resume mode: checkpoint"
  echo "Resume checkpoint: $latest_checkpoint"
  resume_args+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=$latest_checkpoint" "trainer.val_before_train=False")
else
  if [[ "$ALLOW_FRESH_START" != "1" ]]; then
    echo "No compatible checkpoint found under $CHECKPOINT_ROOT." >&2
    echo "This experiment previously used trainer.save_freq=-1, so step 64 was not saved." >&2
    echo "Set ALLOW_FRESH_START=1 once to start from the base model and create checkpoints every ${SAVE_FREQ} steps." >&2
    exit 4
  fi
  echo "Resume mode: fresh start (explicitly allowed)"
  echo "No compatible checkpoint exists; the unsaved step-64 state cannot be recovered."
  resume_args+=("trainer.resume_mode=disable" "trainer.val_before_train=True")
fi

echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Checkpoint interval: every $SAVE_FREQ steps"
echo "Log file: $LOG_FILE"

cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HOME=/root/data/zh/huggingface
export WANDB_DIR="$REPO_ROOT/recipe/GraphGPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m recipe.GraphGPO.main_graphgpo \
  algorithm.adv_estimator=graphgpo \
  algorithm.gamma=0.20 \
  algorithm.graphgpo.step_advantage_w=1.0 \
  algorithm.graphgpo.episode_advantage_w=1.0 \
  algorithm.graphgpo.mode=mean_std_norm \
  algorithm.graphgpo.normalize_distance=False \
  algorithm.use_kl_in_reward=False \
  data.train_files=/root/data/verl-agent/text/train.parquet \
  data.val_files=/root/data/verl-agent/text/test.parquet \
  data.train_batch_size=16 \
  data.val_batch_size=128 \
  data.max_prompt_length=5120 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=/root/data/zh/huggingface/Qwen2.5-1.5B-Instruct \
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
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  env.env_name=webshop/WebAgentTextEnv \
  env.seed=0 \
  env.max_steps=15 \
  env.webshop.observation_mode=text_rich \
  env.rollout.n=8 \
  env.resources_per_worker.num_cpus=0.1 \
  trainer.critic_warmup=0 \
  'trainer.logger=[console,wandb]' \
  trainer.project_name=verl_agent_webshop \
  trainer.experiment_name=graphgpo_qwen2.5_1.5b_webshop \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=5 \
  trainer.total_epochs=150 \
  trainer.default_local_dir="$CHECKPOINT_ROOT" \
  "${resume_args[@]}" \
  "$@"
