import numpy as np
import torch

from gigpo import core_fcsr_gigpo, core_gigpo


def test_fcsr_gigpo_shape_and_padding_mask():
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float32)
    advantages, returns, metrics = core_fcsr_gigpo.compute_fcsr_gigpo_advantage(
        response_mask=mask,
        uid=np.array(["u1", "u1"], dtype=object),
        traj_uid=np.array(["t1", "t2"], dtype=object),
        anchor_obs=np.array(["same search page", "same search page"], dtype=object),
        actions=np.array(["click[correct product]", "click[wrong product]"], dtype=object),
        env_rewards=np.array([1.0, 0.0], dtype=np.float32),
        fcsr_rewards=np.array([0.0, 0.0], dtype=np.float32),
        omega_fcsr=0.0,
        lambda_fallback=0.0,
    )
    assert advantages.shape == (2, 4)
    assert returns.shape == (2, 4)
    assert torch.all(advantages[0, 2:] == 0)
    assert torch.all(advantages[1, 3:] == 0)
    assert metrics["fcsr_gigpo/anchor_coverage"] == 1.0


def test_fcsr_gigpo_degenerates_to_gigpo_on_valid_env_groups():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    token_rewards = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    step_rewards = torch.tensor([1.0, 0.0])
    uid = np.array(["u1", "u1"], dtype=object)
    traj_uid = np.array(["t1", "t2"], dtype=object)
    obs = np.array(["same search page", "same search page"], dtype=object)
    actions = np.array(["click[correct product]", "click[wrong product]"], dtype=object)

    expected, _ = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=token_rewards,
        step_rewards=step_rewards,
        response_mask=mask,
        anchor_obs=obs,
        index=uid,
        traj_index=traj_uid,
        step_advantage_w=1.0,
        mode="mean_norm",
        enable_similarity=False,
    )
    actual, _, _ = core_fcsr_gigpo.compute_fcsr_gigpo_advantage(
        response_mask=mask,
        uid=uid,
        traj_uid=traj_uid,
        anchor_obs=obs,
        actions=actions,
        env_rewards=np.array([1.0, 0.0], dtype=np.float32),
        fcsr_rewards=np.zeros(2, dtype=np.float32),
        omega_env=1.0,
        omega_fcsr=0.0,
        lambda_fallback=0.0,
        norm_type="none",
    )
    assert torch.allclose(actual, expected)


def test_state_key_includes_uid_and_selected_options():
    obs = np.array(["product page with buy now", "product page with buy now"], dtype=object)
    keys = core_fcsr_gigpo.build_webshop_state_keys(
        anchor_obs=obs,
        uid=np.array(["u1", "u2"], dtype=object),
        traj_uid=np.array(["t1", "t2"], dtype=object),
        actions=np.array(["click[b-blue]", "click[b-blue]"], dtype=object),
    )
    assert keys[0] != keys[1]

    option_keys = core_fcsr_gigpo.build_webshop_state_keys(
        anchor_obs=obs,
        uid=np.array(["u1", "u1"], dtype=object),
        traj_uid=np.array(["t1", "t1"], dtype=object),
        actions=np.array(["click[b-blue]", "click[xx-large]"], dtype=object),
    )
    assert option_keys[0] != option_keys[1]


def test_same_action_anchor_group_skips_step_advantages():
    mask = torch.ones(2, 2)
    _, _, metrics, details = core_fcsr_gigpo.compute_fcsr_gigpo_advantage(
        response_mask=mask,
        uid=np.array(["u1", "u1"], dtype=object),
        traj_uid=np.array(["t1", "t2"], dtype=object),
        anchor_obs=np.array(["same page", "same page"], dtype=object),
        actions=np.array(["click[next >]", "click[next >]"], dtype=object),
        env_rewards=np.array([1.0, -1.0], dtype=np.float32),
        fcsr_rewards=np.array([0.1, 0.1], dtype=np.float32),
        return_details=True,
    )
    assert metrics["fcsr_gigpo/skipped_single_action_group"] == 1.0
    assert np.allclose(details["env_step_adv"], 0.0)
    assert np.allclose(details["fcsr_step_adv"], 0.0)


def test_webshop_action_ranking_uses_env_anchor_returns():
    mask = torch.ones(3, 2)
    _, _, _, details = core_fcsr_gigpo.compute_fcsr_gigpo_advantage(
        response_mask=mask,
        uid=np.array(["u1", "u1", "u1"], dtype=object),
        traj_uid=np.array(["t1", "t2", "t3"], dtype=object),
        anchor_obs=np.array(["search results page", "search results page", "search results page"], dtype=object),
        actions=np.array(
            ["click[blue waterproof hiking shoes]", "click[wrong cotton shirt]", "click[next >]"],
            dtype=object,
        ),
        env_rewards=np.array([1.0, 0.0, -1.0], dtype=np.float32),
        fcsr_rewards=np.zeros(3, dtype=np.float32),
        omega_fcsr=0.0,
        lambda_fallback=0.0,
        return_details=True,
    )
    env_step = details["env_step_adv"]
    assert env_step[0] > env_step[1] > env_step[2]


def test_search_anchor_group_consumes_env_and_fcsr_residuals():
    mask = torch.ones(2, 2)
    _, _, metrics, details = core_fcsr_gigpo.compute_fcsr_gigpo_advantage(
        response_mask=mask,
        uid=np.array(["question-1", "question-1"], dtype=object),
        traj_uid=np.array(["rollout-1", "rollout-2"], dtype=object),
        anchor_obs=np.array(["Question: Who wrote Hamlet?", "Question: Who wrote Hamlet?"], dtype=object),
        actions=np.array(["<search>Hamlet author</search>", "<answer>Marlowe</answer>"], dtype=object),
        env_rewards=np.array([1.0, 0.0], dtype=np.float32),
        fcsr_rewards=np.array([0.1, -0.1], dtype=np.float32),
        omega_env=0.8,
        omega_fcsr=0.08,
        lambda_fallback=0.02,
        return_details=True,
    )
    assert metrics["fcsr_gigpo/anchor_coverage"] == 1.0
    assert np.any(np.abs(details["env_step_adv"]) > 0)
    assert np.any(np.abs(details["fcsr_step_adv"]) > 0)
