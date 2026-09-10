#!/usr/bin/env bash
set -euo pipefail

repo=/root/data/zh/gigpo+fcsr_verl
current_pid=14587
current_ckpt_dir=/root/data/zh/checkpoints/verl_agent_alfworld/graphgpo_qwen2.5_1.5b_alfworld
seed1_script="$repo/recipe/GraphGPO/run_qwen2.5_1.5b_alfworld_train_seed1.sh"
seed1_session=graphgpo_alfworld_seed1
stamp="$(date +%Y%m%d_%H%M%S)"
watch_log="$repo/outputs/graphgpo_seed1_watch_${stamp}.log"
seed1_log="$repo/outputs/graphgpo_alfworld_seed1_4gpu_${stamp}.log"

mkdir -p "$repo/outputs" "$repo/outputs/graphgpo_wandb" /root/data/zh/checkpoints/verl_agent_alfworld
exec >>"$watch_log" 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

current_is_running() {
  [[ -r "/proc/$current_pid/cmdline" ]] \
    && tr '\0' ' ' < "/proc/$current_pid/cmdline" \
      | grep -q 'recipe.GraphGPO.main_graphgpo.*env.seed=0'
}

seed0_completed() {
  [[ -d "$current_ckpt_dir/global_step_150" ]] && return 0
  [[ -f "$current_ckpt_dir/latest_checkpointed_iteration.txt" ]] || return 1
  [[ "$(tr -cd '0-9' < "$current_ckpt_dir/latest_checkpointed_iteration.txt")" == 150 ]]
}

gpu_idle() {
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '
        {
          gsub(/ /, "", $1); gsub(/ /, "", $2);
          if ($1 > 1024 || $2 > 5) busy=1
        }
        END { exit !(NR == 4 && !busy) }
      '
}

log "watcher started; waiting for seed0 pid=$current_pid to complete"
while current_is_running; do
  step="$(tmux capture-pane -p -t graphgpo_alfworld:0.0 -S -60 2>/dev/null \
    | tr '\r' '\n' \
    | grep -oE 'training/global_step:[0-9.]+' \
    | tail -n 1 \
    | cut -d: -f2 \
    || true)"
  log "seed0 still running; global_step=${step:-unknown}"
  sleep 300
done

log 'seed0 process exited; verifying final checkpoint 150'
for _ in $(seq 1 60); do
  seed0_completed && break
  sleep 10
done
if ! seed0_completed; then
  log 'seed0 did not produce global_step_150; seed1 will not start'
  exit 42
fi

log 'seed0 completion verified'

# Give the completed job time to release Ray/GPU resources naturally.
for _ in $(seq 1 20); do
  gpu_idle && break
  sleep 30
done

if ! gpu_idle; then
  if pgrep -af 'recipe[.]GraphGPO[.]main_graphgpo' >/dev/null; then
    log 'another GraphGPO process is active; waiting instead of stopping Ray'
  else
    log 'stale Ray/GPU workers remain after seed0; stopping the completed run Ray runtime'
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate /root/data/conda/envs/verl-agent
    ray stop --force || true
  fi
fi

while ! gpu_idle; do
  log 'waiting for all four GPUs to become idle'
  sleep 60
done

if pgrep -af 'recipe[.]GraphGPO[.]main_graphgpo' >/dev/null; then
  log 'a GraphGPO process appeared after the idle check; seed1 will not start'
  exit 43
fi
if tmux has-session -t "$seed1_session" 2>/dev/null; then
  log "tmux session $seed1_session already exists; refusing duplicate launch"
  exit 44
fi

tmux new-session -d -s "$seed1_session"
launch="source /root/miniconda3/etc/profile.d/conda.sh && conda activate /root/data/conda/envs/verl-agent && cd '$repo' && export ALFWORLD_DATA=/root/data/.cache/alfworld HF_HOME=/root/data/.cache/huggingface WANDB_DIR='$repo/outputs/graphgpo_wandb' CHECKPOINTS_DIR=/root/data/zh/checkpoints/verl_agent_alfworld CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=online TOKENIZERS_PARALLELISM=false && bash '$seed1_script' 2>&1 | tee -a '$seed1_log'"
tmux send-keys -t "$seed1_session":0.0 "$launch" C-m
log "seed1 launch sent; tmux=$seed1_session log=$seed1_log"

for _ in $(seq 1 180); do
  if ! tmux has-session -t "$seed1_session" 2>/dev/null; then
    log 'seed1 tmux session exited during startup'
    exit 45
  fi

  seed1_pid="$(pgrep -af 'recipe[.]GraphGPO[.]main_graphgpo' \
    | awk '/env[.]seed=1/ {print $1; exit}' \
    || true)"
  pane="$(tmux capture-pane -p -t "$seed1_session":0.0 -S -220 2>/dev/null | tr '\r' '\n' || true)"
  run_url="$(printf '%s\n' "$pane" \
    | grep -oE 'https://wandb[.]ai/[^ ]+/runs/[A-Za-z0-9]+' \
    | tail -n 1 \
    || true)"

  if [[ -n "$seed1_pid" ]] && [[ -n "$run_url" ]] \
      && printf '%s\n' "$pane" | grep -q 'Training Progress:'; then
    log "seed1 startup verified; pid=$seed1_pid wandb=$run_url"
    exit 0
  fi

  if printf '%s\n' "$pane" \
      | grep -Eq 'CUDA out of memory|OutOfMemory|NCCL.*(error|Error)|(^|[^A-Za-z])Killed([^A-Za-z]|$)'; then
    log 'fatal startup error detected in seed1 log'
    exit 46
  fi

  sleep 30
done

log 'seed1 did not reach W&B + Training Progress within 90 minutes'
exit 47
