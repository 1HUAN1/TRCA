from reward_methods.hbrr import classify_failure_type, rewrite_rewards_hbrr
from reward_methods.normalization import FCSRConfig
from reward_methods.schema import StepRecord, TrajectoryRecord


def test_hbrr_classifies_progress_before_unknown():
    cfg = FCSRConfig(use_failure_rewriting=True)
    step = StepRecord(task_id="u", env_name="alfworld", step_id=0)
    step.diagnostic.update({"progress_delta": 0.2, "risk": 0.0})
    assert classify_failure_type(step, None, cfg) == "progress"


def test_hbrr_rewrite_adds_components_and_penalizes_bottleneck_failure():
    cfg = FCSRConfig(use_failure_rewriting=True)
    good = StepRecord(task_id="u", env_name="alfworld", step_id=0)
    bad = StepRecord(task_id="u", env_name="alfworld", step_id=1)
    good.diagnostic.update({"subgoal_progress": 0.5, "progress_delta": 0.3, "risk": 0.0})
    bad.diagnostic.update({"subgoal_progress": 0.0, "progress_delta": -0.5, "risk": 0.9, "no_op": True})
    traj = TrajectoryRecord("u", "alfworld", [good, bad], final_success=False)
    rewrite_rewards_hbrr(traj, cfg)
    assert "hbrr_reward" in good.reward_components
    assert "hbrr_reward" in bad.reward_components
    bottlenecks = [step.reward_components["hbrr_bottleneck"] for step in traj.steps]
    assert sum(bottlenecks) == 1.0
    assert bad.diagnostic["failure_type"] == "redundant"

