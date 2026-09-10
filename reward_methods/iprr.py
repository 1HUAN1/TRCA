"""Lightweight WebShop information-potential reward."""

from __future__ import annotations

from reward_methods.normalization import FCSRConfig, clip, safe_float
from reward_methods.schema import StepRecord
from reward_methods.webshop_strict import classify_webshop_action


def _potential(step: StepRecord, config: FCSRConfig) -> float:
    d = step.diagnostic
    attribute = safe_float(d.get("attribute_coverage", 0.0), 0.0)
    evidence = safe_float(d.get("evidence_coverage", 0.0), 0.0)
    option = 0.0
    candidate = 0.0
    if "webshop" in step.env_name.lower():
        gate = classify_webshop_action(step, cfg=config)
        if gate.category == "allowed_product_click":
            candidate = 1.0
        elif gate.category == "allowed_goal_option":
            option = 1.0
    else:
        option = 1.0 if step.action_type == "click" and any(token in step.action.lower() for token in ("option", "color", "size", "style")) else 0.0
        candidate = 1.0 if step.action_type == "click" else 0.0
    premature = 1.0 if d.get("premature_buy", False) else 0.0
    return (
        config.iprr_w_attribute * attribute
        + config.iprr_w_candidate * candidate
        + config.iprr_w_option * option
        + 0.25 * evidence
        - config.iprr_w_premature * premature
    )


def compute_iprr_reward(step: StepRecord, prev_step: StepRecord | None, config: FCSRConfig) -> tuple[float, dict[str, float]]:
    current = _potential(step, config)
    previous = _potential(prev_step, config) if prev_step is not None else 0.0
    raw = current - config.iprr_gamma * previous
    if step.diagnostic.get("premature_buy", False):
        raw -= config.iprr_w_wrong_attribute * max(0.0, 0.35 - safe_float(step.diagnostic.get("attribute_coverage", 0.0), 0.0))
    reward = clip(raw, config.reward_clip_min, config.reward_clip_max)
    return reward, {"iprr_potential": current, "iprr_raw": raw, "iprr_reward": reward}

