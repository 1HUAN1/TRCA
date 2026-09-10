"""Hindsight Bottleneck-Recovery Reward."""

from __future__ import annotations

from reward_methods.normalization import FCSRConfig, clip, safe_float, sigmoid
from reward_methods.schema import StepRecord, TrajectoryRecord


FAILURE_TYPES = {
    "progress",
    "exploratory",
    "recoverable",
    "irreversible",
    "redundant",
    "premature",
    "recovery",
    "unknown",
}


def classify_failure_type(step: StepRecord, prev_step: StepRecord | None, config: FCSRConfig) -> str:
    d = step.diagnostic
    if safe_float(d.get("deadlock_risk", 0.0), 0.0) >= config.deadlock_threshold:
        return "irreversible"
    if d.get("premature_finish") or d.get("premature_buy"):
        return "premature"
    if d.get("repeated_action") or d.get("no_op"):
        return "redundant"
    if safe_float(d.get("progress_delta", 0.0), 0.0) > config.progress_positive_threshold:
        return "progress"
    if safe_float(d.get("evidence_gain", 0.0), 0.0) > 0 or safe_float(d.get("attribute_coverage", 0.0), 0.0) > 0:
        return "exploratory"
    if prev_step is not None:
        prev_risk = safe_float(prev_step.diagnostic.get("risk", 0.0), 0.0)
        curr_risk = safe_float(d.get("risk", 0.0), 0.0)
        if prev_risk > config.recovery_risk_threshold and curr_risk < prev_risk:
            return "recovery"
    if d.get("valid_action") is False:
        return "recoverable"
    return "unknown"


def _hind_q(step: StepRecord, config: FCSRConfig) -> float:
    d = step.diagnostic
    progress = safe_float(d.get("subgoal_progress", 0.0), 0.0)
    risk = safe_float(d.get("risk", 0.0), 0.0)
    info = max(safe_float(d.get("evidence_coverage", 0.0), 0.0), safe_float(d.get("attribute_coverage", 0.0), 0.0))
    verifier = safe_float(d.get("verifier_signal", 0.0), 0.0)
    return sigmoid(
        config.hbrr_w_progress * progress
        - config.hbrr_w_risk * risk
        + config.hbrr_w_info * info
        + config.hbrr_w_verifier * verifier
    )


def rewrite_rewards_hbrr(traj: TrajectoryRecord, config: FCSRConfig) -> TrajectoryRecord:
    """Add HBRR reward components in-place."""
    if not config.use_failure_rewriting:
        for step in traj.steps:
            step.reward_components.setdefault("hbrr_reward", 0.0)
            step.reward_components.setdefault("hbrr_raw", 0.0)
        return traj
    if not traj.steps:
        return traj

    q = [_hind_q(step, config) for step in traj.steps]
    delta_q = []
    for i, _step in enumerate(traj.steps):
        if i + 1 < len(traj.steps):
            delta_q.append(q[i + 1] - q[i])
        else:
            terminal = 1.0 if traj.final_success else 0.0
            delta_q.append(terminal - q[i])

    bottleneck_idx = None
    if traj.final_success is False and delta_q:
        bottleneck_idx = min(range(len(delta_q)), key=lambda idx: delta_q[idx])

    for i, step in enumerate(traj.steps):
        prev_step = traj.steps[i - 1] if i > 0 else None
        ftype = classify_failure_type(step, prev_step, config)
        step.diagnostic["failure_type"] = ftype
        step.diagnostic["recoverability"] = q[i]

        d = step.diagnostic
        info_gain = max(
            safe_float(d.get("evidence_gain", 0.0), 0.0),
            safe_float(d.get("attribute_coverage", 0.0), 0.0),
            safe_float(d.get("subgoal_progress", 0.0), 0.0),
        )

        raw = (
            config.hbrr_alpha_delta_q * delta_q[i]
            + config.hbrr_beta_progress * float(ftype == "progress")
            + config.hbrr_gamma_exploratory * info_gain * float(ftype == "exploratory")
            + config.hbrr_eta_recovery * max(delta_q[i], 0.0) * float(ftype == "recovery")
            - config.hbrr_bottleneck_penalty * float(i == bottleneck_idx)
            - config.hbrr_irreversible_penalty * float(ftype == "irreversible")
            - config.hbrr_redundant_penalty * float(ftype == "redundant")
            - config.hbrr_premature_penalty * float(ftype == "premature")
        )

        reward = clip(raw, config.reward_clip_min, config.reward_clip_max)
        step.reward_components["hbrr_q_hind"] = q[i]
        step.reward_components["hbrr_delta_q"] = delta_q[i]
        step.reward_components["hbrr_bottleneck"] = float(i == bottleneck_idx)
        step.reward_components["hbrr_raw"] = raw
        step.reward_components["hbrr_reward"] = reward

    return traj

