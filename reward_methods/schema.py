"""Backward-compatible trajectory schema for step-level reward methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import re


SUCCESS_KEYS = ("success", "is_success", "task_success", "won")
VALIDITY_KEYS = ("is_action_valid", "action_valid", "valid_action", "valid", "is_valid")
ALFWORLD_PROJECTION_DIAGNOSTIC_KEYS = {
    "alfworld_raw_format_valid": "raw_format_valid",
    "alfworld_tag_format_valid": "tag_format_valid",
    "alfworld_action_tag_found": "action_tag_found",
    "alfworld_has_action_tag": "has_action_tag",
    "alfworld_multi_action": "multi_action",
    "alfworld_think_tag_found": "think_tag_found",
    "alfworld_think_tag_required": "think_tag_required",
    "alfworld_contains_chinese": "contains_chinese",
    "alfworld_bare_action_candidate": "bare_action_candidate",
    "alfworld_bare_command_parseable": "bare_command_parseable",
    "alfworld_bare_admissible_exact_match": "bare_admissible_exact_match",
    "alfworld_admissible_exact_match": "admissible_exact_match",
    "alfworld_matched_admissible_action": "matched_admissible_action",
    "alfworld_repaired_action_valid": "repaired_action_valid",
    "alfworld_executable_action_valid": "executable_action_valid",
    "alfworld_projected_action": "projected_action",
    "alfworld_raw_action_text": "raw_action_text",
    "alfworld_normalized_action": "projection_normalized_action",
    "alfworld_action_source": "action_source",
    "alfworld_invalid_reason": "projection_invalid_reason",
}
ALFWORLD_PROJECTION_BOOL_DIAGNOSTICS = {
    "raw_format_valid",
    "tag_format_valid",
    "action_tag_found",
    "has_action_tag",
    "multi_action",
    "think_tag_found",
    "think_tag_required",
    "contains_chinese",
    "bare_command_parseable",
    "bare_admissible_exact_match",
    "admissible_exact_match",
    "repaired_action_valid",
    "executable_action_valid",
}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion that tolerates tensors, numpy scalars, and None."""
    if value is None:
        return default
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: Optional[bool] = False) -> Optional[bool]:
    """Best-effort bool conversion with support for unknown values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "success"}:
            return True
        if text in {"false", "0", "no", "n", "fail", "failed"}:
            return False
        return default
    try:
        if hasattr(value, "item"):
            value = value.item()
        return bool(value)
    except (TypeError, ValueError):
        return default


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return default
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(safe_text(item) for item in value)
    return str(value)


@dataclass
class StepRecord:
    task_id: str
    env_name: str
    step_id: int

    state: Any = None
    observation: str = ""
    thought: str = ""
    action: str = ""
    action_type: str = ""
    tool_output: str = ""
    next_state: Any = None

    done: bool = False
    final_success: Optional[bool] = None
    final_score: Optional[float] = None
    env_reward: Optional[float] = None

    diagnostic: dict[str, Any] = field(default_factory=dict)
    reward_components: dict[str, float] = field(default_factory=dict)

    dense_reward: float = 0.0
    mixed_reward: float = 0.0
    advantage: Optional[float] = None

    raw: Optional[dict[str, Any]] = field(default=None, repr=False, compare=False)


@dataclass
class TrajectoryRecord:
    task_id: str
    env_name: str
    steps: list[StepRecord] = field(default_factory=list)
    final_success: Optional[bool] = None
    final_score: Optional[float] = None
    terminal_reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _nested_get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in data:
        return data[key]
    info = data.get("infos")
    if isinstance(info, dict) and key in info:
        return info[key]
    return default


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_action(output: Any) -> str:
    text = safe_text(output)
    tagged = _extract_tag(text, "action")
    if tagged:
        return " ".join(tagged.split())
    return " ".join(text.strip().split())


def extract_thought(output: Any) -> str:
    return _extract_tag(safe_text(output), "think")


def extract_task(input_text: Any) -> str:
    text = safe_text(input_text)
    patterns = [
        r"Your task is to:\s*(.*?)(?:\n|$)",
        r"Your question:\s*(.*?)(?:\n|$)",
        r"Task:\s*(.*?)(?:\n|$)",
        r"Instruction:\s*(.*?)(?:\n|$)",
        r"\bQuestion:\s*(.*?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return " ".join(match.group(1).strip().split())
    return ""


def extract_current_observation(input_text: Any) -> str:
    text = safe_text(input_text)
    patterns = [
        r"current observation is:\s*(.*?)(?:\nYour admissible actions|\n\nNow|$)",
        r"Your current observation is:\s*(.*?)(?:\n\nYour task|\nYour task|\nYour admissible actions|\n\nNow|$)",
        r"observation is:\s*(.*?)(?:\nYour admissible actions|\n\nNow|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return " ".join(match.group(1).strip().split())
    return ""


def extract_admissible_actions(input_text: Any) -> list[str]:
    text = safe_text(input_text)
    match = re.search(
        r"admissible actions.*?:\s*(\[.*?\])",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    raw = match.group(1)
    return [" ".join(item.split()) for item in re.findall(r"'([^']+)'|\"([^\"]+)\"", raw) for item in item if item]


def infer_action_type(action: str) -> str:
    normalized = action.strip().lower()
    if not normalized:
        return "empty"
    if normalized.startswith("search") or normalized.startswith("query"):
        return "search"
    if normalized.startswith("click"):
        return "click"
    if "buy" in normalized:
        return "buy"
    if normalized.startswith("go to"):
        return "navigate"
    if normalized.startswith("open"):
        return "open"
    if normalized.startswith("close"):
        return "close"
    if normalized.startswith("take") or normalized.startswith("pick"):
        return "take"
    if normalized.startswith("put") or normalized.startswith("place"):
        return "put"
    if normalized.startswith("look"):
        return "look"
    if normalized.startswith("examine"):
        return "examine"
    if normalized.startswith("inventory"):
        return "inventory"
    if normalized.startswith("heat"):
        return "heat"
    if normalized.startswith("cool"):
        return "cool"
    if normalized.startswith("clean"):
        return "clean"
    if normalized.startswith("toggle") or normalized.startswith("turn"):
        return "toggle"
    return normalized.split(" ", 1)[0]


def step_from_existing_dict(
    raw: dict[str, Any],
    step_id: int,
    env_name: str = "",
    task_id: Optional[str] = None,
) -> StepRecord:
    """Convert an existing rollout step dict into a StepRecord."""
    task_id = task_id or safe_text(raw.get("uid") or raw.get("index") or raw.get("traj_uid") or "unknown")
    input_text = raw.get("input", raw.get("prompt", ""))
    output_text = raw.get("output", raw.get("response", raw.get("action", "")))
    observation = safe_text(
        raw.get("observation")
        or raw.get("anchor_obs")
        or raw.get("state")
        or _nested_get(raw, "observation")
        or extract_current_observation(input_text)
    )
    action = extract_action(output_text)
    valid_action = None
    for key in VALIDITY_KEYS:
        if _nested_get(raw, key, None) is not None:
            valid_action = safe_bool(_nested_get(raw, key), default=None)
            break

    step = StepRecord(
        task_id=task_id,
        env_name=env_name,
        step_id=step_id,
        state=raw.get("anchor_obs", raw.get("state", observation)),
        observation=observation,
        thought=extract_thought(output_text),
        action=action,
        action_type=infer_action_type(action),
        tool_output=safe_text(raw.get("tool_output") or raw.get("next_observation") or _nested_get(raw, "tool_output")),
        next_state=raw.get("next_state", raw.get("next_obs", raw.get("next_observation"))),
        done=bool(raw.get("done", False)),
        final_success=None,
        final_score=None,
        env_reward=safe_float(raw.get("rewards", raw.get("reward")), default=0.0),
        raw=raw,
    )
    step.diagnostic["valid_action"] = valid_action
    step.diagnostic["admissible_actions"] = extract_admissible_actions(input_text)
    step.diagnostic["task_instruction"] = extract_task(input_text)
    for raw_key, diagnostic_key in ALFWORLD_PROJECTION_DIAGNOSTIC_KEYS.items():
        value = _nested_get(raw, raw_key, None)
        if value is None:
            continue
        if diagnostic_key in ALFWORLD_PROJECTION_BOOL_DIAGNOSTICS:
            step.diagnostic[diagnostic_key] = safe_bool(value, default=False)
        else:
            step.diagnostic[diagnostic_key] = safe_text(value)
    return step


def _is_active(raw: dict[str, Any]) -> bool:
    return safe_bool(raw.get("active_masks", True), default=True) is not False


def _infer_terminal_reward(raw_steps: list[dict[str, Any]]) -> float:
    active_steps = [step for step in raw_steps if _is_active(step)]
    if not active_steps:
        active_steps = raw_steps
    if not active_steps:
        return 0.0
    last = active_steps[-1]
    reward = safe_float(last.get("rewards", last.get("reward")), default=0.0)
    if reward != 0.0:
        return reward
    for key in ("episode_rewards", "score", "final_score"):
        if key in last:
            return safe_float(last.get(key), default=0.0)
    return 0.0


def _infer_success(raw_steps: list[dict[str, Any]], terminal_reward: float, threshold: float) -> Optional[bool]:
    for raw in reversed(raw_steps):
        for key in SUCCESS_KEYS:
            if key in raw:
                return safe_bool(raw.get(key), default=None)
            info = raw.get("infos")
            if isinstance(info, dict) and key in info:
                return safe_bool(info.get(key), default=None)
    for raw in reversed(raw_steps):
        if "episode_rewards" in raw:
            return safe_float(raw.get("episode_rewards"), default=0.0) >= threshold
    if terminal_reward != 0.0:
        return terminal_reward >= threshold
    return False


def trajectory_from_existing_rollout(
    raw_steps: list[dict[str, Any]],
    env_name: str = "",
    success_reward_threshold: float = 0.99,
    include_inactive: bool = False,
) -> TrajectoryRecord:
    """Convert a list of existing rollout step dicts into a TrajectoryRecord."""
    active_raw_steps = [step for step in raw_steps if include_inactive or _is_active(step)]
    if not active_raw_steps:
        active_raw_steps = list(raw_steps)
    task_id = "unknown"
    if active_raw_steps:
        task_id = safe_text(active_raw_steps[0].get("uid") or active_raw_steps[0].get("index") or active_raw_steps[0].get("traj_uid") or "unknown")
    terminal_reward = _infer_terminal_reward(active_raw_steps)
    final_success = _infer_success(active_raw_steps, terminal_reward, success_reward_threshold)
    final_score = None
    if active_raw_steps:
        final_score = safe_float(
            active_raw_steps[-1].get("episode_rewards", active_raw_steps[-1].get("score", terminal_reward)),
            default=terminal_reward,
        )
    traj = TrajectoryRecord(
        task_id=task_id,
        env_name=env_name,
        final_success=final_success,
        final_score=final_score,
        terminal_reward=terminal_reward,
        metadata={
            "traj_uid": safe_text(active_raw_steps[0].get("traj_uid")) if active_raw_steps else "",
            "raw_step_count": len(raw_steps),
            "active_step_count": len(active_raw_steps),
        },
    )
    traj.steps = [step_from_existing_dict(raw, i, env_name=env_name, task_id=task_id) for i, raw in enumerate(active_raw_steps)]
    for i, step in enumerate(traj.steps):
        step.final_success = final_success
        step.final_score = final_score
        if i + 1 < len(traj.steps):
            next_step = traj.steps[i + 1]
            step.next_state = next_step.state
            if not step.tool_output and "tool_output" not in step.raw and "next_observation" not in step.raw:
                step.tool_output = next_step.observation
        else:
            if step.next_state is None:
                step.next_state = step.state
            if not step.tool_output and "tool_output" not in step.raw and "next_observation" not in step.raw:
                step.tool_output = safe_text(step.next_state or step.observation)
    return traj


def step_to_json(step: StepRecord) -> dict[str, Any]:
    return {
        "task_id": step.task_id,
        "env_name": step.env_name,
        "step_id": step.step_id,
        "observation": step.observation,
        "thought": step.thought,
        "action": step.action,
        "action_type": step.action_type,
        "tool_output": step.tool_output,
        "done": step.done,
        "final_success": step.final_success,
        "final_score": step.final_score,
        "env_reward": step.env_reward,
        "diagnostic": step.diagnostic,
        "reward_components": step.reward_components,
        "dense_reward": step.dense_reward,
        "mixed_reward": step.mixed_reward,
        "advantage": step.advantage,
    }


def trajectory_to_json(traj: TrajectoryRecord) -> dict[str, Any]:
    return {
        "task_id": traj.task_id,
        "env_name": traj.env_name,
        "final_success": traj.final_success,
        "final_score": traj.final_score,
        "terminal_reward": traj.terminal_reward,
        "metadata": traj.metadata,
        "steps": [step_to_json(step) for step in traj.steps],
    }
