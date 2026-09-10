"""Strict positive-reward gate for WebShop FCSR signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import re

from reward_methods.normalization import clip, safe_float
from reward_methods.schema import StepRecord, TrajectoryRecord, infer_action_type, safe_text


NAVIGATION_TARGETS = {
    "next >",
    "< prev",
    "prev",
    "previous",
    "back to search",
    "description",
    "features",
    "reviews",
}
GENERIC_PRODUCT_WORDS = {
    "product",
    "item",
    "option",
    "color",
    "size",
    "style",
    "brand",
    "price",
    "review",
    "reviews",
    "rating",
    "ratings",
    "buy",
    "webshop",
    "search",
    "find",
    "with",
    "for",
    "and",
    "the",
}
OPTION_WORDS = {
    "black",
    "blue",
    "brown",
    "gray",
    "grey",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
    "beige",
    "small",
    "medium",
    "large",
    "x-large",
    "xx-large",
    "xxx-large",
    "one size",
}


@dataclass(frozen=True)
class WebShopGateResult:
    allowed: bool
    reason: str
    category: str


def canonical_action(action: Any) -> str:
    return " ".join(safe_text(action).lower().strip().split())


def extract_click_target(action: Any) -> str:
    text = canonical_action(action)
    match = re.match(r"click\s*\[(.*?)\]\s*$", text, flags=re.IGNORECASE)
    if match:
        return " ".join(match.group(1).strip().split())
    return ""


def normalize_label(label: Any) -> str:
    text = canonical_action(label)
    text = re.sub(r"[^a-z0-9_<> -]+", " ", text)
    return " ".join(text.split())


def reason_safe_label(label: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", normalize_label(label)).strip("_")
    return safe or "option"


def _tokenize(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]+", safe_text(text).lower())
        if token not in GENERIC_PRODUCT_WORDS and len(token) > 1
    }


def goal_terms(step: StepRecord) -> set[str]:
    return _tokenize(step.diagnostic.get("task_instruction", ""))


def _cfg_bool(cfg: Any, name: str, default: bool) -> bool:
    value = getattr(cfg, name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _is_navigation_target(target: str) -> bool:
    normalized = normalize_label(target)
    return normalized in NAVIGATION_TARGETS or normalized in {"next", "back", "description", "features", "reviews"}


def _is_option_target(target: str) -> bool:
    normalized = normalize_label(target)
    if not normalized:
        return False
    parts = normalized.split()
    if normalized in OPTION_WORDS:
        return True
    if re.match(r"^[a-z]-[a-z0-9][a-z0-9 -]*$", normalized):
        return True
    if any(marker in parts for marker in ("option", "color", "size", "style")):
        return True
    return len(parts) <= 2 and any(word in parts for word in OPTION_WORDS)


def _option_matches_goal(target: str, step: StepRecord, cfg: Any) -> bool:
    normalized = normalize_label(target)
    task = safe_text(step.diagnostic.get("task_instruction", "")).lower()
    if normalized and normalized in task:
        return True
    if _cfg_bool(cfg, "webshop_allow_goal_option_prefix", True):
        suffix = normalized.split("-", 1)[-1].strip()
        if suffix and suffix in task:
            return True
    return bool(_tokenize(normalized) & goal_terms(step))


def _product_click_goal_match(target: str, step: StepRecord) -> bool:
    target_terms = _tokenize(target)
    if target_terms & goal_terms(step):
        return True
    obs_terms = _tokenize(f"{step.observation} {step.tool_output}")
    return bool(target_terms and target_terms <= obs_terms and len(obs_terms & goal_terms(step)) >= 2)


def _candidate_click_targets(step: StepRecord) -> list[str]:
    actions = step.diagnostic.get("admissible_actions", [])
    targets = []
    if not isinstance(actions, Iterable) or isinstance(actions, (str, bytes)):
        return targets
    for action in actions:
        target = extract_click_target(action)
        if not target:
            continue
        if _is_navigation_target(target) or _is_option_target(target):
            continue
        if "buy" in target:
            continue
        targets.append(target)
    return targets


def _single_result_goal_match(target: str, step: StepRecord) -> bool:
    candidates = _candidate_click_targets(step)
    if len(candidates) > 1:
        return False
    if candidates and normalize_label(target) != normalize_label(candidates[0]):
        return False
    text_terms = _tokenize(f"{target} {step.observation} {step.tool_output}")
    return len(text_terms & goal_terms(step)) >= 2


def is_buy_action(step: StepRecord) -> bool:
    action = canonical_action(step.action)
    target = extract_click_target(action)
    action_type = step.action_type or infer_action_type(action)
    return action_type == "buy" or target == "buy now" or action == "buy now" or "buy now" in action


def buy_ready(step: StepRecord) -> bool:
    if bool(step.final_success):
        return True
    if safe_float(step.env_reward, 0.0) > 0.0:
        return True
    raw = step.raw or {}
    return safe_float(raw.get("rewards", raw.get("reward", raw.get("episode_rewards"))), 0.0) > 0.0


def classify_webshop_action(
    step: StepRecord,
    prev_step: StepRecord | None = None,
    traj: TrajectoryRecord | None = None,
    cfg: Any = None,
) -> WebShopGateResult:
    del prev_step, traj
    action = canonical_action(step.action)
    action_type = step.action_type or infer_action_type(action)
    target = extract_click_target(action)

    if action_type == "search":
        if _cfg_bool(cfg, "webshop_allow_search_reward", False):
            return WebShopGateResult(True, "generic_search_allowed", "allowed_search")
        return WebShopGateResult(False, "rejected_generic_search", "rejected_generic_search")

    if is_buy_action(step):
        if buy_ready(step):
            return WebShopGateResult(True, "buy_ready_all_required_options_matched", "allowed_buy_ready")
        return WebShopGateResult(False, "rejected_buy_too_early", "rejected_buy_too_early")

    if action_type != "click":
        return WebShopGateResult(False, "rejected_generic_click", "rejected_generic_click")

    if _is_navigation_target(target):
        if _cfg_bool(cfg, "webshop_allow_navigation_positive_reward", False):
            return WebShopGateResult(True, "navigation_allowed", "allowed_navigation")
        return WebShopGateResult(False, "rejected_navigation", "rejected_navigation")

    if _is_option_target(target):
        if _option_matches_goal(target, step, cfg):
            return WebShopGateResult(True, f"goal_option_{reason_safe_label(target)}_match", "allowed_goal_option")
        return WebShopGateResult(False, "rejected_wrong_option", "rejected_wrong_option")

    if _product_click_goal_match(target, step):
        return WebShopGateResult(True, "product_click_goal_match", "allowed_product_click")
    if _single_result_goal_match(target, step):
        return WebShopGateResult(True, "product_click_single_result_goal_match", "allowed_product_click")

    if _cfg_bool(cfg, "webshop_allow_generic_click_positive_reward", False):
        return WebShopGateResult(True, "generic_click_allowed", "allowed_generic_click")
    return WebShopGateResult(False, "rejected_generic_click", "rejected_generic_click")


def strict_gate_reward(
    reward: float,
    step: StepRecord,
    prev_step: StepRecord | None = None,
    traj: TrajectoryRecord | None = None,
    cfg: Any = None,
) -> tuple[float, WebShopGateResult]:
    result = classify_webshop_action(step, prev_step=prev_step, traj=traj, cfg=cfg)
    lower = safe_float(getattr(cfg, "fcsr_strict_clip_min", -0.2), -0.2)
    upper = safe_float(getattr(cfg, "fcsr_strict_clip_max", 0.2), 0.2)
    value = safe_float(reward, 0.0)
    if value > 0.0 and not result.allowed:
        return 0.0, result
    return clip(value, lower, upper), result


def selected_options_from_actions(actions: Iterable[Any]) -> tuple[str, ...]:
    selected = []
    for action in actions:
        target = extract_click_target(action)
        if target and _is_option_target(target):
            selected.append(normalize_label(target))
    return tuple(sorted(set(selected)))
