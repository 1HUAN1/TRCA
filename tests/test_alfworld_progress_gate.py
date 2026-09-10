from reward_methods.mixer import compute_rewards_for_batch, compute_rewards_for_trajectory
from reward_methods.normalization import FCSRConfig
from reward_methods.schema import trajectory_from_existing_rollout


def _raw_unrelated_take_traj():
    return [
        {
            "input": "Your task is to: examine the box with the desklamp.\ncurrent observation is: You see a lettuce and a box.\nYour admissible actions are: ['take lettuce from table']",
            "output": "<action>take lettuce from table</action>",
            "uid": "u1",
            "traj_uid": "t1",
            "rewards": 0.0,
            "active_masks": True,
            "is_action_valid": True,
        },
        {
            "input": "Your task is to: examine the box with the desklamp.\ncurrent observation is: You are holding lettuce.\nYour admissible actions are: ['inventory']",
            "output": "<action>inventory</action>",
            "uid": "u1",
            "traj_uid": "t1",
            "rewards": 0.0,
            "active_masks": True,
            "is_action_valid": True,
        },
    ]


def test_progress_gate_zeroes_no_progress_positive_reward():
    traj = trajectory_from_existing_rollout(_raw_unrelated_take_traj(), env_name="alfworld/AlfredTWEnv")
    cfg = FCSRConfig(
        reward_method="fcsr_lite",
        alfworld_training_chain="optimized",
        alfworld_progress_gated_reward=True,
        use_failure_rewriting=False,
    )
    compute_rewards_for_trajectory(traj, cfg)
    assert traj.steps[0].reward_components["positive_zeroed_no_progress"] == 1.0
    assert traj.steps[0].reward_components["dcsr_reward"] == 0.0


def test_standard_chain_keeps_existing_fcsr_lite_behavior():
    traj = trajectory_from_existing_rollout(_raw_unrelated_take_traj(), env_name="alfworld/AlfredTWEnv")
    cfg = FCSRConfig(reward_method="fcsr_lite", alfworld_training_chain="standard", use_failure_rewriting=False)
    compute_rewards_for_trajectory(traj, cfg)
    assert "positive_zeroed_no_progress" not in traj.steps[0].reward_components
    assert traj.steps[0].reward_components["dcsr_reward"] > 0.0


def test_rewardflow_batch_dispatch_is_unchanged():
    class Cfg:
        reward_method = "rewardflow"

    batch = [_raw_unrelated_take_traj()]
    updated, metrics = compute_rewards_for_batch(batch, Cfg())
    assert updated is batch
    assert metrics == {}
