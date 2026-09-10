from reward_methods.dcsr import compute_dcsr_reward
from reward_methods.normalization import FCSRConfig
from reward_methods.schema import StepRecord


def test_dcsr_rewards_progress_and_validity():
    cfg = FCSRConfig()
    step = StepRecord(task_id="u", env_name="alfworld", step_id=0)
    step.diagnostic.update(
        {
            "valid_action": True,
            "state_changed": True,
            "progress_delta": 0.3,
            "risk_delta": 0.0,
        }
    )
    reward, comps = compute_dcsr_reward(step, None, cfg)
    assert reward > 0.3
    assert comps["validity_reward"] == cfg.dcsr_beta_validity


def test_dcsr_penalizes_invalid_noop_repeat():
    cfg = FCSRConfig()
    step = StepRecord(task_id="u", env_name="alfworld", step_id=0)
    step.diagnostic.update(
        {
            "valid_action": False,
            "state_changed": False,
            "progress_delta": 0.0,
            "risk_delta": 0.5,
            "no_op": True,
            "repeated_action": True,
        }
    )
    reward, comps = compute_dcsr_reward(step, None, cfg)
    assert reward < 0.0
    assert comps["invalid_penalty"] < 0.0
    assert comps["noop_penalty"] < 0.0

