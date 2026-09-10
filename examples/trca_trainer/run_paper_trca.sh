#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash examples/trca_trainer/run_paper_trca.sh {alfworld|webshop|searchqa} [hydra.override=...]" >&2
}

[[ $# -ge 1 ]] || { usage; exit 2; }
BENCHMARK="$1"
shift
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUBRIC_LIBRARY="${TRCA_RUBRIC_LIBRARY:-${REPO_ROOT}/trca_rubrics/${BENCHMARK}.json}"
[[ -s "${RUBRIC_LIBRARY}" ]] || {
  echo "Missing validated rubric library: ${RUBRIC_LIBRARY}" >&2
  echo "Run examples/trca_trainer/generate_rubrics.sh before training." >&2
  exit 1
}

COMMON=(
  algorithm.adv_estimator=trca
  algorithm.gamma=0.95
  ++algorithm.trca.gamma=0.95
  ++algorithm.trca.benchmark="${BENCHMARK}"
  ++reward_method=trca
  ++trca_rubric_library="${RUBRIC_LIBRARY}"
  ++trca_mix_lambda=0.8
  ++trca_evidence_budget=1.0
  ++trca_invalidity_budget=1.0
  ++trca_execution_budget=1.0
  actor_rollout_ref.actor.optim.lr=8e-7
  actor_rollout_ref.actor.ppo_mini_batch_size=256
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.012
  actor_rollout_ref.actor.entropy_coeff=0.001
  actor_rollout_ref.actor.clip_ratio_low=0.20
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.use_invalid_action_penalty=false
  actor_rollout_ref.rollout.temperature=1.0
  algorithm.use_kl_in_reward=false
  data.train_batch_size=32
  data.max_response_length=512
  env.rollout.n=8
)

case "${BENCHMARK}" in
  alfworld)
    PROJECT_ROOT="${REPO_ROOT}" exec bash "${REPO_ROOT}/examples/fcsr_trainer/run_alfworld_fcsr_gigpo_qwen25_1p5b.sh" vllm fcsr "${COMMON[@]}" "$@"
    ;;
  webshop)
    REPO_ROOT="${REPO_ROOT}" exec bash "${REPO_ROOT}/examples/fcsr_trainer/run_webshop_fcsr_gigpo_qwen25_1p5b.sh" vllm fcsr "${COMMON[@]}" "$@"
    ;;
  searchqa)
    REPO_ROOT="${REPO_ROOT}" exec bash "${REPO_ROOT}/examples/fcsr_trainer/run_search.sh" vllm fgpo full "${COMMON[@]}" "$@"
    ;;
  *) usage; exit 2 ;;
esac
