from pathlib import Path

from reward_methods.mixer import compute_rewards_for_batch, compute_rewards_for_trajectory
from reward_methods.normalization import FCSRConfig
from reward_methods.schema import trajectory_from_existing_rollout


def _raw_alfworld_traj():
    return [
        {
            "input": "Your task is to: examine the box with the desklamp.\ncurrent observation is: You are in a room.\nYour admissible actions are: ['go to drawer 1']",
            "output": "<action>go to drawer 1</action>",
            "uid": "u1",
            "traj_uid": "t1",
            "rewards": 0.0,
            "active_masks": True,
            "is_action_valid": True,
        },
        {
            "input": "Your task is to: examine the box with the desklamp.\ncurrent observation is: You arrive at drawer 1.\nYour admissible actions are: ['open drawer 1']",
            "output": "<action>open drawer 1</action>",
            "uid": "u1",
            "traj_uid": "t1",
            "rewards": 1.0,
            "active_masks": True,
            "is_action_valid": True,
        },
    ]


def test_fcsr_lite_adds_terminal_anchor():
    cfg = FCSRConfig(reward_method="fcsr_lite", use_failure_rewriting=True)
    traj = trajectory_from_existing_rollout(_raw_alfworld_traj(), env_name="alfworld/AlfredTWEnv")
    compute_rewards_for_trajectory(traj, cfg)
    assert traj.steps[-1].reward_components["terminal_anchor"] == 1.0
    assert traj.steps[-1].mixed_reward > 0.0


def test_fgpo_lambda_convexly_mixes_dcsr_and_hbrr(monkeypatch):
    from reward_methods import mixer

    def fake_dcsr(step, prev, config):
        return 0.8, {"dcsr_reward": 0.8}

    def fake_hbrr(traj, config):
        for step in traj.steps:
            step.reward_components["hbrr_reward"] = -0.2
        return traj

    monkeypatch.setattr(mixer, "compute_dcsr_reward", fake_dcsr)
    monkeypatch.setattr(mixer, "rewrite_rewards_hbrr", fake_hbrr)

    for mix_lambda, expected in [(0.0, 0.8), (1.0, -0.2), (0.25, 0.55)]:
        cfg = FCSRConfig(
            reward_method="fcsr_lite",
            use_failure_rewriting=True,
            use_action_format_reward=False,
            terminal_reward_weight=0.0,
            iprr_reward_weight=0.0,
            reward_clip_min=-1.0,
            reward_clip_max=1.0,
            fgpo_mix_lambda=mix_lambda,
        )
        traj = trajectory_from_existing_rollout(_raw_alfworld_traj(), env_name="alfworld/AlfredTWEnv")
        compute_rewards_for_trajectory(traj, cfg)
        for step in traj.steps:
            assert step.reward_components["fgpo_mix_lambda"] == mix_lambda
            assert abs(step.reward_components["fgpo_reward"] - expected) < 1e-8
            assert abs(step.mixed_reward - expected) < 1e-8


def test_negative_fgpo_lambda_preserves_legacy_weights(monkeypatch):
    from reward_methods import mixer

    monkeypatch.setattr(mixer, "compute_dcsr_reward", lambda step, prev, config: (0.8, {"dcsr_reward": 0.8}))

    def fake_hbrr(traj, config):
        for step in traj.steps:
            step.reward_components["hbrr_reward"] = -0.2
        return traj

    monkeypatch.setattr(mixer, "rewrite_rewards_hbrr", fake_hbrr)
    cfg = FCSRConfig(
        reward_method="fcsr_lite",
        use_failure_rewriting=True,
        use_action_format_reward=False,
        terminal_reward_weight=0.0,
        dense_reward_weight=0.3,
        hbrr_reward_weight=0.7,
        fgpo_mix_lambda=-1.0,
        reward_clip_min=-1.0,
        reward_clip_max=1.0,
    )
    traj = trajectory_from_existing_rollout(_raw_alfworld_traj(), env_name="alfworld/AlfredTWEnv")
    compute_rewards_for_trajectory(traj, cfg)
    expected = 0.3 * 0.8 + 0.7 * -0.2
    assert all(abs(step.mixed_reward - expected) < 1e-8 for step in traj.steps)
    assert all("fgpo_reward" not in step.reward_components for step in traj.steps)


def test_compute_rewards_for_batch_writes_step_rewards_and_dump(tmp_path):
    class Cfg:
        reward_method = "fcsr_lite"
        use_failure_rewriting = True
        debug_dump_rewards = True
        debug_dump_dir = str(tmp_path)

        class env:
            env_name = "alfworld/AlfredTWEnv"

    batch = [_raw_alfworld_traj()]
    updated, metrics = compute_rewards_for_batch(batch, Cfg())
    assert updated[0][0]["step_rewards"] != 0.0
    assert "fcsr_reward_components" in updated[0][0]
    assert Path(metrics["fcsr/debug_dump_path"]).exists()



def test_core_dispatch_decodes_responses_for_fcsr_actions():
    from rewardflow.core_rewardflow import _decode_fcsr_step_outputs

    class Tok:
        def decode(self, response, skip_special_tokens=True):
            assert response == [1, 2, 3]
            return "<action>open drawer 1</action>"

    batch = [[{"raw_prompt": [{"role": "user", "content": "Question: Who wrote Hamlet?"}], "responses": [1, 2, 3]}]]
    errors = _decode_fcsr_step_outputs(batch, Tok())
    assert errors == 0
    assert batch[0][0]["input"] == "Question: Who wrote Hamlet?"
    assert batch[0][0]["output"] == "<action>open drawer 1</action>"


def test_search_dispatch_uses_real_rollout_shape_for_nonzero_first_retrieval_reward():
    from rewardflow.core_rewardflow import apply_rewardflow_propagation

    class Tok:
        def decode(self, response, skip_special_tokens=True):
            assert response == [7]
            return "<search>Hamlet author</search>"

    class Cfg:
        reward_method = "fcsr_lite"
        use_failure_rewriting = True
        dense_reward_weight = 0.10
        hbrr_reward_weight = 0.03

        class env:
            env_name = "search"

    batch = [[{
        "raw_prompt": [{"role": "user", "content": "Question: Who wrote Hamlet?"}],
        "responses": [7],
        "anchor_obs": "Who wrote Hamlet?",
        "tool_output": "<information>Hamlet was written by William Shakespeare.</information>",
        "next_observation": "<information>Hamlet was written by William Shakespeare.</information>",
        "uid": "u-search",
        "traj_uid": "t-search",
        "rewards": 0.0,
        "active_masks": True,
        "is_action_valid": True,
    }]]

    updated, metrics = apply_rewardflow_propagation(batch, Cfg(), Tok(), envs=None, return_metrics=True)
    assert updated[0][0]["fcsr_reward_components"]["dcsr_reward"] > 0.0
    assert metrics["fcsr/dcsr_mean"] > 0.0
    assert metrics["fcsr/decode_error_count"] == 0.0


def test_explicit_empty_answer_tool_output_is_not_backfilled_with_old_evidence():
    raw = [{
        "input": "Question: Who wrote Hamlet?",
        "output": "<answer>Shakespeare</answer>",
        "anchor_obs": "<information>Hamlet was written by William Shakespeare.</information>",
        "tool_output": "",
        "next_observation": "",
        "uid": "u-search",
        "traj_uid": "t-search",
        "rewards": 1.0,
        "active_masks": True,
    }]
    trajectory = trajectory_from_existing_rollout(raw, env_name="search")
    assert trajectory.steps[0].tool_output == ""


def test_search_fcsr_metrics_include_real_retrieval_and_hbrr_signals():
    class Cfg:
        reward_method = "fcsr_lite"
        use_failure_rewriting = True
        dense_reward_weight = 0.10
        hbrr_reward_weight = 0.03

        class env:
            env_name = "search"

    batch = [
        [
            {
                "input": "Question: Who wrote Hamlet?",
                "output": "<search>Hamlet author</search>",
                "tool_output": "<information>Hamlet was written by William Shakespeare.</information>",
                "next_observation": "<information>Hamlet was written by William Shakespeare.</information>",
                "uid": "u-search",
                "traj_uid": "t-search",
                "rewards": 0.0,
                "active_masks": True,
                "is_action_valid": True,
            },
            {
                "input": "Question: Who wrote Hamlet?",
                "output": "<answer>Christopher Marlowe</answer>",
                "tool_output": "",
                "next_observation": "",
                "uid": "u-search",
                "traj_uid": "t-search",
                "rewards": 0.0,
                "active_masks": True,
                "is_action_valid": True,
            },
        ]
    ]
    updated, metrics = compute_rewards_for_batch(batch, Cfg())
    first_components = updated[0][0]["fcsr_reward_components"]
    assert first_components["dcsr_reward"] != 0.0
    assert "hbrr_reward" in first_components
    assert metrics["fcsr/hbrr_mean"] != 0.0
    assert metrics["search/retrieval_success_rate"] == 0.5
    assert metrics["search/tool_error_rate"] == 0.0
    assert metrics["search/evidence_gain_mean"] > 0.0
