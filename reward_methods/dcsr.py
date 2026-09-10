"""Diagnostic-Causal Step Reward."""

from __future__ import annotations

from reward_methods.normalization import FCSRConfig, clip, safe_float
from reward_methods.schema import StepRecord


def compute_dcsr_reward(step: StepRecord, prev_step: StepRecord | None, config: FCSRConfig) -> tuple[float, dict[str, float]]:
    d = step.diagnostic

    valid_action = d.get("valid_action")
    state_changed = d.get("state_changed")
    invalid = valid_action is False

    progress_delta = safe_float(d.get("progress_delta", 0.0), 0.0)
    risk_delta = safe_float(d.get("risk_delta", 0.0), 0.0)
    deadlock_risk = safe_float(d.get("deadlock_risk", 0.0), 0.0)

    no_op = bool(d.get("no_op", False))
    repeated = bool(d.get("repeated_action", False))
    premature = bool(d.get("premature_finish", False) or d.get("premature_buy", False))

    if valid_action is True and state_changed is True:
        validity_reward = 1.0
    elif valid_action is None and state_changed is True:
        validity_reward = 0.5
    else:
        validity_reward = 0.0

    prev_risk = 0.0
    if prev_step is not None:
        prev_risk = safe_float(prev_step.diagnostic.get("risk", 0.0), 0.0)

    recovery_reward = 1.0 if (prev_risk >= config.recovery_risk_threshold and progress_delta > 0) else 0.0

    raw = (
        config.dcsr_alpha_progress * progress_delta
        + config.dcsr_beta_validity * validity_reward
        + config.dcsr_eta_recovery * recovery_reward
        - config.dcsr_lambda_risk * max(risk_delta, 0.0)
        - config.penalty_invalid * float(invalid)
        - config.penalty_noop * float(no_op)
        - config.penalty_repeat * float(repeated)
        - config.penalty_premature * float(premature)
        - config.penalty_deadlock * deadlock_risk
    )

    reward = clip(raw, config.reward_clip_min, config.reward_clip_max)
    components = {
        "dcsr_raw": raw,
        "progress_reward": config.dcsr_alpha_progress * progress_delta,
        "validity_reward": config.dcsr_beta_validity * validity_reward,
        "recovery_reward": config.dcsr_eta_recovery * recovery_reward,
        "risk_penalty": -config.dcsr_lambda_risk * max(risk_delta, 0.0),
        "invalid_penalty": -config.penalty_invalid * float(invalid),
        "noop_penalty": -config.penalty_noop * float(no_op),
        "repeat_penalty": -config.penalty_repeat * float(repeated),
        "premature_penalty": -config.penalty_premature * float(premature),
        "deadlock_penalty": -config.penalty_deadlock * deadlock_risk,
        "dcsr_reward": reward,
    }
    return reward, components

