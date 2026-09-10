"""FCSR-GiGPO advantage computation.

This keeps GiGPO's env-return signal as the primary objective and uses strict
WebShop-gated FCSR only as a small residual.
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any
import re

import numpy as np
import torch

from reward_methods.schema import extract_action, safe_text
from reward_methods.webshop_strict import extract_click_target, normalize_label, selected_options_from_actions


def _to_numpy(values: Any, length: int, default: Any = "") -> np.ndarray:
    if values is None:
        return np.array([default for _ in range(length)], dtype=object)
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 0:
        arr = np.array([arr.item() for _ in range(length)], dtype=object)
    return arr


def _to_float_array(values: Any, length: int, default: float = 0.0) -> np.ndarray:
    arr = _to_numpy(values, length, default)
    out = np.zeros(length, dtype=np.float32)
    for i in range(length):
        value = arr[i] if i < len(arr) else default
        try:
            if hasattr(value, "item"):
                value = value.item()
            out[i] = float(value)
        except (TypeError, ValueError):
            out[i] = float(default)
    return out


def _normalize_observation(obs: Any) -> str:
    text = safe_text(obs).lower()
    text = re.sub(r"your admissible actions.*", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _page_type(obs: Any) -> str:
    text = safe_text(obs).lower()
    if "buy now" in text or ("description" in text and "reviews" in text):
        return "product_page"
    if "search results" in text or "results for" in text or "next >" in text or "< prev" in text:
        return "search_page"
    if "search" in text and "webshop" in text:
        return "search_home"
    return "other"


def _product_id(obs: Any) -> str:
    text = safe_text(obs)
    for pattern in (
        r"\b(?:asin|product id|item id)\s*[:#]?\s*([A-Z0-9]{6,16})\b",
        r"\b([A-Z][A-Z0-9]{9})\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return ""


def _search_page_index(obs: Any) -> str:
    text = safe_text(obs).lower()
    match = re.search(r"\bpage\s+(\d+)\b", text)
    if match:
        return match.group(1)
    return ""


def _normalized_actions(outputs: Any, length: int) -> np.ndarray:
    arr = _to_numpy(outputs, length, "")
    actions = []
    for value in arr:
        action = extract_action(value)
        actions.append(" ".join(action.lower().strip().split()))
    return np.array(actions, dtype=object)


def build_webshop_state_keys(
    anchor_obs: Any,
    uid: Any,
    traj_uid: Any,
    actions: Any | None = None,
    include_recovered_options: bool = True,
) -> np.ndarray:
    n = len(uid)
    obs_arr = _to_numpy(anchor_obs, n, "")
    uid_arr = _to_numpy(uid, n, "")
    traj_arr = _to_numpy(traj_uid, n, "")
    action_arr = _normalized_actions(actions, n)
    history: dict[str, list[str]] = defaultdict(list)
    keys = []
    for i in range(n):
        traj = str(traj_arr[i])
        selected = selected_options_from_actions(history[traj]) if include_recovered_options else ()
        obs = obs_arr[i]
        key = (
            str(uid_arr[i]),
            _normalize_observation(obs),
            _page_type(obs),
            _product_id(obs),
            _search_page_index(obs),
            tuple(selected),
        )
        keys.append(key)
        history[traj].append(action_arr[i])
    out = np.empty(len(keys), dtype=object)
    out[:] = keys
    return out


def compute_discounted_returns(rewards: np.ndarray, traj_uid: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    positions_by_traj: dict[str, list[int]] = defaultdict(list)
    for i, traj in enumerate(traj_uid):
        positions_by_traj[str(traj)].append(i)
    for positions in positions_by_traj.values():
        running = 0.0
        for idx in reversed(positions):
            running = float(rewards[idx]) + float(gamma) * running
            returns[idx] = running
    return returns


def _episode_env_advantage(
    env_rewards: np.ndarray,
    uid: np.ndarray,
    traj_uid: np.ndarray,
    norm_type: str,
    epsilon: float,
) -> tuple[np.ndarray, dict[str, float]]:
    traj_returns: dict[str, float] = defaultdict(float)
    traj_to_uid: dict[str, str] = {}
    for reward, u, traj in zip(env_rewards, uid, traj_uid):
        traj_key = str(traj)
        traj_returns[traj_key] += float(reward)
        traj_to_uid[traj_key] = str(u)

    uid_to_returns: dict[str, list[float]] = defaultdict(list)
    for traj, value in traj_returns.items():
        uid_to_returns[traj_to_uid[traj]].append(value)

    uid_mean = {u: float(np.mean(values)) for u, values in uid_to_returns.items()}
    uid_std = {u: float(np.std(values)) for u, values in uid_to_returns.items()}
    adv = np.zeros(len(env_rewards), dtype=np.float32)
    for i, traj in enumerate(traj_uid):
        u = str(uid[i])
        value = traj_returns[str(traj)]
        centered = value - uid_mean[u]
        if norm_type in {"std", "mean_std", "mean_std_norm"}:
            centered = centered / (uid_std[u] + epsilon)
        adv[i] = centered
    return adv, {traj: value for traj, value in traj_returns.items()}


def _similar(a: Any, b: Any, thresh: float) -> bool:
    return SequenceMatcher(None, repr(a), repr(b)).ratio() >= thresh


def _build_anchor_groups(
    state_keys: np.ndarray,
    actions: np.ndarray,
    min_anchor_group_size: int,
    min_distinct_actions: int,
    enable_similarity: bool,
    similarity_thresh: float,
) -> tuple[list[list[int]], np.ndarray, dict[str, float]]:
    groups: list[list[int]] = []
    if not enable_similarity:
        buckets: dict[Any, list[int]] = defaultdict(list)
        for i, key in enumerate(state_keys):
            buckets[repr(key)].append(i)
        groups = list(buckets.values())
    else:
        clusters: list[tuple[Any, list[int]]] = []
        for i, key in enumerate(state_keys):
            placed = False
            for rep, indices in clusters:
                if key[0] == rep[0] and _similar(key, rep, similarity_thresh):
                    indices.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append((key, [i]))
        groups = [indices for _, indices in clusters]

    valid_mask = np.zeros(len(state_keys), dtype=np.float32)
    skipped_small = 0
    skipped_single_action = 0
    distinct_counts = []
    eligible = 0
    for indices in groups:
        distinct = {str(actions[i]) for i in indices}
        distinct_counts.append(len(distinct))
        if len(indices) < min_anchor_group_size:
            skipped_small += 1
            continue
        if len(distinct) < min_distinct_actions:
            skipped_single_action += 1
            continue
        eligible += 1
        valid_mask[indices] = 1.0

    metrics = {
        "fcsr_gigpo/anchor_groups": float(len(groups)),
        "fcsr_gigpo/eligible_anchor_groups": float(eligible),
        "fcsr_gigpo/skipped_small_group": float(skipped_small),
        "fcsr_gigpo/skipped_single_action_group": float(skipped_single_action),
        "fcsr_gigpo/avg_anchor_group_size": float(np.mean([len(g) for g in groups])) if groups else 0.0,
        "fcsr_gigpo/distinct_actions_mean": float(np.mean(distinct_counts)) if distinct_counts else 0.0,
    }
    return groups, valid_mask, metrics


def _group_centered_advantage(
    values: np.ndarray,
    groups: list[list[int]],
    valid_mask: np.ndarray,
    norm_type: str,
    epsilon: float,
) -> np.ndarray:
    adv = np.zeros_like(values, dtype=np.float32)
    for indices in groups:
        if not indices or valid_mask[indices[0]] <= 0.0:
            continue
        group_values = values[indices].astype(np.float32)
        centered = group_values - float(np.mean(group_values))
        if norm_type in {"std", "mean_std", "mean_std_norm"}:
            centered = centered / (float(np.std(group_values)) + epsilon)
        adv[indices] = centered
    return adv


def _uid_centered(values: np.ndarray, uid: np.ndarray, norm_type: str, epsilon: float) -> np.ndarray:
    adv = np.zeros_like(values, dtype=np.float32)
    by_uid: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(uid):
        by_uid[str(u)].append(i)
    for indices in by_uid.values():
        group_values = values[indices].astype(np.float32)
        centered = group_values - float(np.mean(group_values))
        if norm_type in {"std", "mean_std", "mean_std_norm"}:
            centered = centered / (float(np.std(group_values)) + epsilon)
        adv[indices] = centered
    return adv


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_fcsr_gigpo_advantage(
    response_mask: torch.Tensor,
    uid: Any,
    traj_uid: Any,
    anchor_obs: Any,
    actions: Any | None,
    env_rewards: Any,
    fcsr_rewards: Any | None = None,
    gamma_env: float = 0.95,
    gamma_fcsr: float = 0.95,
    omega_env: float = 0.8,
    omega_fcsr: float = 0.1,
    lambda_fallback: float = 0.02,
    norm_type: str = "none",
    adv_clip: float = 5.0,
    min_anchor_group_size: int = 2,
    min_distinct_actions: int = 2,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.90,
    use_env_step_advantage: bool = True,
    use_fcsr_step_residual: bool = True,
    use_fcsr_non_anchor_fallback: bool = True,
    webshop_state_key_include_recovered_options: bool = True,
    epsilon: float = 1e-6,
    return_details: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]] | tuple[torch.Tensor, torch.Tensor, dict[str, float], dict[str, Any]]:
    n = response_mask.shape[0]
    uid_arr = _to_numpy(uid, n, "")
    traj_arr = _to_numpy(traj_uid, n, "")
    action_arr = _normalized_actions(actions, n)
    env = _to_float_array(env_rewards, n, 0.0)
    fcsr = _to_float_array(fcsr_rewards, n, 0.0)
    fcsr = np.clip(fcsr, -0.2, 0.2)
    norm = str(norm_type or "none").lower()

    episode_adv, traj_returns = _episode_env_advantage(env, uid_arr, traj_arr, norm, epsilon)
    env_returns = compute_discounted_returns(env, traj_arr, gamma_env)
    fcsr_returns = compute_discounted_returns(fcsr, traj_arr, gamma_fcsr)
    state_keys = build_webshop_state_keys(
        anchor_obs=anchor_obs,
        uid=uid_arr,
        traj_uid=traj_arr,
        actions=action_arr,
        include_recovered_options=webshop_state_key_include_recovered_options,
    )
    groups, valid_anchor_mask, group_metrics = _build_anchor_groups(
        state_keys=state_keys,
        actions=action_arr,
        min_anchor_group_size=int(min_anchor_group_size),
        min_distinct_actions=int(min_distinct_actions),
        enable_similarity=bool(enable_similarity),
        similarity_thresh=float(similarity_thresh),
    )

    env_step_adv = (
        _group_centered_advantage(env_returns, groups, valid_anchor_mask, norm, epsilon)
        if use_env_step_advantage
        else np.zeros(n, dtype=np.float32)
    )
    fcsr_step_adv = (
        _group_centered_advantage(fcsr_returns, groups, valid_anchor_mask, norm, epsilon)
        if use_fcsr_step_residual
        else np.zeros(n, dtype=np.float32)
    )
    fcsr_step_adv = np.clip(fcsr_step_adv, -2.0, 2.0)
    fallback_adv = (
        _uid_centered(fcsr_returns, uid_arr, norm, epsilon)
        if use_fcsr_non_anchor_fallback
        else np.zeros(n, dtype=np.float32)
    )
    non_anchor = 1.0 - valid_anchor_mask
    final_step_adv_unclipped = (
        episode_adv
        + float(omega_env) * env_step_adv
        + float(omega_fcsr) * fcsr_step_adv
        + float(lambda_fallback) * non_anchor * fallback_adv
    )
    final_step_adv = np.clip(final_step_adv_unclipped, -float(adv_clip), float(adv_clip)).astype(np.float32)

    mask = response_mask.to(dtype=torch.float32)
    step_tensor = torch.tensor(final_step_adv, dtype=torch.float32, device=response_mask.device)
    advantages = step_tensor.unsqueeze(-1).tile([1, response_mask.shape[-1]]) * mask
    returns = advantages.clone()

    terminal_success = np.array([1.0 if traj_returns.get(str(traj), 0.0) > 0.0 else 0.0 for traj in traj_arr], dtype=np.float32)
    metrics = dict(group_metrics)
    metrics.update(
        {
            "fcsr_gigpo/anchor_coverage": float(np.mean(valid_anchor_mask)) if n else 0.0,
            "fcsr_gigpo/env_step_adv_mean": _mean(env_step_adv),
            "fcsr_gigpo/env_step_adv_std": _std(env_step_adv),
            "fcsr_gigpo/fcsr_step_adv_mean": _mean(fcsr_step_adv),
            "fcsr_gigpo/fcsr_step_adv_std": _std(fcsr_step_adv),
            "fcsr_gigpo/final_adv_mean": _mean(final_step_adv),
            "fcsr_gigpo/final_adv_std": _std(final_step_adv),
            "fcsr_gigpo/final_adv_clip_frac": float(np.mean(np.abs(final_step_adv_unclipped) > float(adv_clip))) if n else 0.0,
            "fcsr_gigpo/non_anchor_fallback_rate": float(np.mean(non_anchor)) if use_fcsr_non_anchor_fallback and n else 0.0,
            "fcsr_gigpo/fallback_positive_rate": float(np.mean((non_anchor * fallback_adv) > 0.0)) if use_fcsr_non_anchor_fallback and n else 0.0,
            "fcsr_gigpo/corr_env_step_adv_terminal_success": _corr(env_step_adv, terminal_success),
            "fcsr_gigpo/corr_fcsr_step_adv_terminal_success": _corr(fcsr_step_adv, terminal_success),
        }
    )
    details = {
        "episode_adv": episode_adv,
        "env_step_adv": env_step_adv,
        "fcsr_step_adv": fcsr_step_adv,
        "fallback_adv": fallback_adv,
        "valid_anchor_mask": valid_anchor_mask,
        "state_keys": state_keys,
        "final_step_adv": final_step_adv,
    }
    if return_details:
        return advantages, returns, metrics, details
    return advantages, returns, metrics
