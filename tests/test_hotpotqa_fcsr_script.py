import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "fcsr_trainer" / "run_search.sh"
LAUNCHER = ROOT / "examples" / "fcsr_trainer" / "launch_searchqa_fgpo_8gpu.sh"
ROLLOUT_LOOP = ROOT / "agent_system" / "multi_turn_rollout" / "rollout_loop.py"


def test_unified_search_fcsr_script_exists_and_is_executable():
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_size > 0
    assert os.access(SCRIPT, os.X_OK)


def test_default_fgpo_branch_keeps_dcsr_and_hbrr_active():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "algorithm.adv_estimator=fcsr_gigpo" in text
    assert "++reward_method=fcsr_lite" in text
    assert "++use_failure_rewriting=true" in text
    assert 'DENSE_REWARD_WEIGHT="${DENSE_REWARD_WEIGHT:-0.10}"' in text
    assert 'HBRR_REWARD_WEIGHT="${HBRR_REWARD_WEIGHT:-0.03}"' in text
    assert 'MODE_ARGS=(algorithm.adv_estimator=fcsr_gigpo' in text
    assert "env.env_name=search" in text


def test_search_launcher_supports_profiles_models_datasets_and_ablations():
    text = SCRIPT.read_text(encoding="utf-8")
    for token in ("full|smoke|eval", "3b)", "7b)", "gigpo)", "dcsr)", "hbrr)", "fgpo)"):
        assert token in text
    for dataset in ("nq", "triviaqa", "popqa", "hotpotqa", "2wiki", "musique", "bamboogle", "all"):
        assert dataset in text
    assert "prepare_searchqa_data" in text
    assert "--validate-only" in text
    assert '--against "$TRAIN_FILE"' in text


def test_full_profile_matches_official_gigpo_search_scale():
    text = SCRIPT.read_text(encoding="utf-8")
    for token in (
        'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"',
        'VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-512}"',
        'DEFAULT_GROUP_SIZE=8',
        'DEFAULT_SIMILARITY_THRESH=0.80',
        'GROUP_SIZE="${GROUP_SIZE:-$DEFAULT_GROUP_SIZE}"',
        'MAX_STEPS="${MAX_STEPS:-4}"',
        'PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-512}"',
        'VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"',
        'TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"',
        'INVALID_ACTION_PENALTY_COEF="${INVALID_ACTION_PENALTY_COEF:-0.01}"',
        'DATASET="${DATASET:-all}"',
        'BASELINE_DATA_ROOT="${BASELINE_DATA_ROOT:-/root/data/searchR1_processed_direct}"',
    ):
        assert token in text


def test_search_launcher_has_safe_preflights_and_does_not_stop_ray():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "preflight_retriever" in text
    assert "preflight_gpus" in text
    assert "REQUIRE_FREE_GPUS" in text
    assert "ray stop" not in text
    assert "git reset" not in text
    assert "outputs/alfworld" not in text.lower()
    assert "outputs/webshop" not in text.lower()


def test_search_rollout_persists_post_action_evidence_for_rewards():
    text = ROLLOUT_LOOP.read_text(encoding="utf-8")
    assert "'input': prompt_with_chat_template" in text
    assert "'observation': obs_content" in text
    assert 'batch.non_tensor_batch["next_observation"] = post_action_obs' in text
    assert 'batch.non_tensor_batch["tool_output"] = post_action_obs' in text
    assert 'gen_batch_output.meta_info["fcsr_reward_metrics"]' in text


def test_eight_gpu_launcher_pins_official_3b_scale_and_records_exit_status():
    text = LAUNCHER.read_text(encoding="utf-8")
    for token in (
        'TRAIN_FILE="${TRAIN_FILE:-/root/data/searchR1_processed_direct/train.parquet}"',
        'VAL_FILE="${VAL_FILE:-/root/data/searchR1_processed_direct/test.parquet}"',
        'N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"',
        'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"',
        'VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-512}"',
        'GROUP_SIZE="${GROUP_SIZE:-8}"',
        'TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"',
        'SIMILARITY_THRESH="${SIMILARITY_THRESH:-0.80}"',
        'printf \'%s\\n\' "$status" > "$LOG_DIR/exit_code"',
    ):
        assert token in text
