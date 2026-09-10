"""Paper-aligned episode-relative and context-relative TRCA advantage."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

import numpy as np
import torch


def _numpy(value: Any, length: int, default: Any) -> np.ndarray:
    if value is None:
        return np.asarray([default] * length, dtype=object)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=object).reshape(-1)
    if len(array) != length:
        raise ValueError(f"Expected {length} values, got {len(array)}")
    return array


def _floats(value: Any, length: int, default: float = 0.0) -> np.ndarray:
    values = _numpy(value, length, default)
    result = np.zeros(length, dtype=np.float64)
    for index, item in enumerate(values):
        try:
            result[index] = float(item.item() if hasattr(item, "item") else item)
        except (TypeError, ValueError):
            result[index] = default
    return result


def _normalize(values: list[float], epsilon: float) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std())
    if std == 0.0:
        return [0.0] * len(values)
    return ((array - float(array.mean())) / (std + epsilon)).tolist()


def _stable_state(value: Any, benchmark: str) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    text = " ".join(str(value or "").casefold().split())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _stable_state(parsed, benchmark)
    except (json.JSONDecodeError, TypeError):
        pass
    # Exact observable content is conservative: it may form fewer groups, but
    # never treats two different evidence/selection states as equivalent.
    return text


def compute_trca_advantage(
    response_mask: torch.Tensor,
    uid: Any,
    traj_uid: Any,
    anchor_obs: Any,
    env_rewards: Any,
    trca_rewards: Any,
    terminal_outcomes: Any,
    step_ids: Any = None,
    gamma: float = 0.95,
    benchmark: str = "alfworld",
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Implement Eq. 16-20 and broadcast only over the supplied action mask."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    n = int(response_mask.shape[0])
    task_ids = _numpy(uid, n, "")
    trajectory_ids = _numpy(traj_uid, n, "")
    states = _numpy(anchor_obs, n, "")
    raw_env = _floats(env_rewards, n)
    env = np.zeros(n, dtype=np.float64)
    rubric = _floats(trca_rewards, n)
    outcomes = _floats(terminal_outcomes, n)
    steps = _numpy(step_ids, n, None) if step_ids is not None else np.asarray([None] * n, dtype=object)

    # Duplicated rows introduced for data-parallel divisibility share one transition unit.
    trajectory_rows: dict[str, list[int]] = defaultdict(list)
    for index, trajectory_id in enumerate(trajectory_ids):
        trajectory_rows[str(trajectory_id)].append(index)
    unit_rows: dict[tuple[str, Any], list[int]] = defaultdict(list)
    trajectory_units: dict[str, list[tuple[str, Any]]] = {}
    for trajectory_id, rows in trajectory_rows.items():
        ordered: list[tuple[str, Any]] = []
        for local_order, row in enumerate(rows):
            raw_step = steps[row]
            key_step = int(raw_step) if raw_step is not None and str(raw_step) not in {"", "None"} else local_order
            key = (trajectory_id, key_step)
            unit_rows[key].append(row)
            if key not in ordered:
                ordered.append(key)
        ordered.sort(key=lambda item: item[1])
        trajectory_units[trajectory_id] = ordered

    # The paper's environment signal is one binary outcome on the terminal
    # transition. Do not mix benchmark-specific 0/10 wrappers into TRCA return.
    for units in trajectory_units.values():
        if units:
            terminal_row = unit_rows[units[-1]][0]
            terminal_value = 1.0 if outcomes[terminal_row] > 0.0 else 0.0
            for duplicate in unit_rows[units[-1]]:
                env[duplicate] = terminal_value

    completion_returns = np.zeros(n, dtype=np.float64)
    unit_return: dict[tuple[str, Any], float] = {}
    for trajectory_id, units in trajectory_units.items():
        running = 0.0
        for unit in reversed(units):
            row = unit_rows[unit][0]
            running = env[row] + rubric[row] + gamma * running
            unit_return[unit] = running
            for duplicate in unit_rows[unit]:
                completion_returns[duplicate] = running

    episode_advantage = np.zeros(n, dtype=np.float64)
    task_trajectories: dict[str, list[str]] = defaultdict(list)
    for trajectory_id, rows in trajectory_rows.items():
        task_id = str(task_ids[rows[0]])
        if trajectory_id not in task_trajectories[task_id]:
            task_trajectories[task_id].append(trajectory_id)
    for trajectory_list in task_trajectories.values():
        normalized = _normalize([outcomes[trajectory_rows[item][0]] for item in trajectory_list], epsilon)
        for trajectory_id, value in zip(trajectory_list, normalized):
            episode_advantage[trajectory_rows[trajectory_id]] = value

    context_advantage = np.zeros(n, dtype=np.float64)
    context_units: dict[tuple[str, str], list[tuple[str, Any]]] = defaultdict(list)
    for unit, rows in unit_rows.items():
        row = rows[0]
        key = (str(task_ids[row]), _stable_state(states[row], benchmark.casefold()))
        if unit not in context_units[key]:
            context_units[key].append(unit)
    comparable_groups = 0
    for units in context_units.values():
        normalized = _normalize([unit_return[item] for item in units], epsilon)
        if any(value != 0.0 for value in normalized):
            comparable_groups += 1
        for unit, value in zip(units, normalized):
            for row in unit_rows[unit]:
                context_advantage[row] = value

    final = episode_advantage + context_advantage
    device, dtype = response_mask.device, torch.float32
    mask = response_mask.to(dtype=dtype)
    scalar_adv = torch.tensor(final, device=device, dtype=dtype).unsqueeze(-1)
    scalar_ret = torch.tensor(completion_returns, device=device, dtype=dtype).unsqueeze(-1)
    advantages = scalar_adv * mask
    returns = scalar_ret * mask
    metrics = {
        "trca/episode_advantage_abs_mean": float(np.mean(np.abs(episode_advantage))) if n else 0.0,
        "trca/context_advantage_abs_mean": float(np.mean(np.abs(context_advantage))) if n else 0.0,
        "trca/final_advantage_abs_mean": float(np.mean(np.abs(final))) if n else 0.0,
        "trca/comparable_context_groups": float(comparable_groups),
        "trca/action_token_rate": float(mask.mean().item()) if mask.numel() else 0.0,
        "trca/raw_env_reward_abs_max": float(np.max(np.abs(raw_env))) if n else 0.0,
        "trca/gamma": float(gamma),
    }
    return advantages, returns, metrics
