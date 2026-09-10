"""Reward dispatch and FCSR-lite mixing."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any
import json
import time

from reward_methods.alfworld_gate import alfworld_gate_reward
from reward_methods.dcsr import compute_dcsr_reward
from reward_methods.debug import dump_reward_trajectories
from reward_methods.diagnostics import parse_diagnostics
from reward_methods.hbrr import rewrite_rewards_hbrr
from reward_methods.iprr import compute_iprr_reward
from reward_methods.normalization import FCSRConfig, clip, reward_method_from_config, safe_float, normalize_rewards_in_trajectory
from reward_methods.schema import TrajectoryRecord, safe_text, trajectory_from_existing_rollout
from reward_methods.webshop_strict import strict_gate_reward
from trca.rewards import compute_trca_rewards


SUPPORTED_REWARD_METHODS = {"outcome_only", "dcsr", "fcsr_lite", "webshop_iprr_lite", "trca"}


def _ensure_diagnostics(traj: TrajectoryRecord, config: FCSRConfig) -> None:
    if not config.use_diagnostics:
        return
    start_time = time.perf_counter()
    for i, step in enumerate(traj.steps):
        prev = traj.steps[i - 1] if i > 0 else None
        step.diagnostic.update(parse_diagnostics(step, prev, traj, config))
    traj.metadata["timing_s/diagnostics"] = time.perf_counter() - start_time


def _is_optimized_alfworld(traj: TrajectoryRecord, config: FCSRConfig) -> bool:
    return "alf" in traj.env_name.lower() and str(config.alfworld_training_chain).strip().lower() == "optimized"


def _has_real_alfworld_progress(step: Any, prev_step: Any | None, config: FCSRConfig) -> bool:
    progress_delta = safe_float(step.diagnostic.get("progress_delta", 0.0), 0.0)
    if progress_delta > safe_float(config.progress_positive_threshold, 0.05):
        return True
    if bool(step.diagnostic.get("completion_critical_action", False)):
        return True
    if prev_step is not None:
        prev_risk = safe_float(prev_step.diagnostic.get("risk", 0.0), 0.0)
        risk = safe_float(step.diagnostic.get("risk", 0.0), 0.0)
        if prev_risk - risk > safe_float(config.recovery_risk_threshold, 0.3):
            return True
    if safe_float(step.diagnostic.get("subgoal_progress", 0.0), 0.0) > 0.0 and progress_delta > safe_float(config.progress_positive_threshold, 0.05):
        return True
    return False


def _gate_alfworld_positive_reward(step: Any, prev_step: Any | None, reward: float, config: FCSRConfig) -> float:
    has_real_progress = _has_real_alfworld_progress(step, prev_step, config)
    step.reward_components["has_real_progress"] = float(bool(has_real_progress))
    if reward > 0.0 and not has_real_progress:
        step.reward_components["positive_zeroed_no_progress"] = 1.0
        return 0.0
    step.reward_components["positive_zeroed_no_progress"] = 0.0
    return reward


def _compute_dcsr_trajectory(traj: TrajectoryRecord, config: FCSRConfig) -> TrajectoryRecord:
    _ensure_diagnostics(traj, config)
    gate_positive = bool(config.alfworld_progress_gated_reward and _is_optimized_alfworld(traj, config))
    for i, step in enumerate(traj.steps):
        prev = traj.steps[i - 1] if i > 0 else None
        dcsr_reward, comps = compute_dcsr_reward(step, prev, config)
        step.reward_components.update(comps)
        if gate_positive:
            dcsr_reward = _gate_alfworld_positive_reward(step, prev, dcsr_reward, config)
            step.reward_components["dcsr_reward"] = dcsr_reward
        step.dense_reward = dcsr_reward
        step.reward_components.setdefault("hbrr_reward", 0.0)
        step.reward_components.setdefault("iprr_reward", 0.0)
        step.reward_components["terminal_anchor"] = 0.0
        step.reward_components["fcsr_raw"] = dcsr_reward
        step.reward_components["fcsr_mixed_reward"] = dcsr_reward
        step.mixed_reward = dcsr_reward
    return traj


def _action_format_reward_eligible(step: Any, config: FCSRConfig) -> bool:
    if not bool(step.diagnostic.get("action_format_valid", False)):
        return False
    if config.format_reward_require_not_invalid and step.diagnostic.get("valid_action") is False:
        return False
    return True


def _trajectory_format_reward(traj: TrajectoryRecord, config: FCSRConfig) -> tuple[float, float]:
    if not traj.steps:
        return 0.0, 0.0
    eligible = [_action_format_reward_eligible(step, config) for step in traj.steps]
    format_rate = float(sum(1 for item in eligible if item) / len(eligible))
    apply_mode = str(config.format_reward_apply or "terminal").strip().lower()
    if not config.use_action_format_reward or apply_mode != "terminal":
        return format_rate, 0.0
    reward = min(config.format_reward_weight * format_rate, config.format_reward_max)
    return format_rate, max(0.0, reward)


def _compute_outcome_only(traj: TrajectoryRecord, config: FCSRConfig) -> TrajectoryRecord:
    terminal = safe_float(traj.terminal_reward, 0.0)
    if terminal == 0.0 and traj.final_success:
        terminal = 1.0
    for i, step in enumerate(traj.steps):
        raw = config.terminal_reward_weight * terminal if i == len(traj.steps) - 1 else 0.0
        reward = clip(raw, config.reward_clip_min, config.reward_clip_max)
        step.reward_components.update(
            {
                "dcsr_reward": 0.0,
                "hbrr_reward": 0.0,
                "iprr_reward": 0.0,
                "terminal_anchor": raw,
                "fcsr_raw": raw,
                "fcsr_mixed_reward": reward,
            }
        )
        step.dense_reward = 0.0
        step.mixed_reward = reward
    return traj


def compute_fcsr_lite_rewards(traj: TrajectoryRecord, config: FCSRConfig) -> TrajectoryRecord:
    _compute_dcsr_trajectory(traj, config)
    if config.use_failure_rewriting:
        traj = rewrite_rewards_hbrr(traj, config)
    else:
        for step in traj.steps:
            step.reward_components.setdefault("hbrr_reward", 0.0)
            step.reward_components.setdefault("hbrr_raw", 0.0)

    if config.use_information_potential and "webshop" in traj.env_name.lower():
        for i, step in enumerate(traj.steps):
            prev = traj.steps[i - 1] if i > 0 else None
            iprr_reward, comps = compute_iprr_reward(step, prev, config)
            step.reward_components.update(comps)
    else:
        for step in traj.steps:
            step.reward_components.setdefault("iprr_reward", 0.0)

    terminal = safe_float(traj.terminal_reward, 0.0)
    if terminal == 0.0 and traj.final_success:
        terminal = 1.0
    format_rate, terminal_format_reward = _trajectory_format_reward(traj, config)
    last_idx = len(traj.steps) - 1
    for i, step in enumerate(traj.steps):
        dcsr = safe_float(step.reward_components.get("dcsr_reward", 0.0), 0.0)
        hbrr = safe_float(step.reward_components.get("hbrr_reward", 0.0), 0.0)
        iprr = safe_float(step.reward_components.get("iprr_reward", 0.0), 0.0)
        mix_lambda = safe_float(config.fgpo_mix_lambda, -1.0)
        if 0.0 <= mix_lambda <= 1.0:
            fgpo_reward = (1.0 - mix_lambda) * dcsr + mix_lambda * hbrr
            raw = fgpo_reward + config.iprr_reward_weight * iprr
            step.reward_components["fgpo_mix_lambda"] = mix_lambda
            step.reward_components["fgpo_reward"] = fgpo_reward
        else:
            raw = (
                config.dense_reward_weight * dcsr
                + config.hbrr_reward_weight * hbrr
                + config.iprr_reward_weight * iprr
            )
        terminal_anchor = 0.0
        format_reward = 0.0
        if i == last_idx:
            terminal_anchor = config.terminal_reward_weight * terminal
            format_reward = terminal_format_reward
            raw += terminal_anchor + format_reward
        mixed = clip(raw, config.reward_clip_min, config.reward_clip_max)
        step.reward_components["terminal_anchor"] = terminal_anchor
        step.reward_components["format_rate"] = format_rate if i == last_idx else 0.0
        step.reward_components["format_reward"] = format_reward
        step.reward_components["fcsr_raw"] = raw
        step.reward_components["fcsr_mixed_reward"] = mixed
        step.mixed_reward = mixed

    if config.reward_normalization == "trajectory":
        normalize_rewards_in_trajectory(traj, field="mixed_reward", config=config)
    return traj


def compute_webshop_iprr_lite_rewards(traj: TrajectoryRecord, config: FCSRConfig) -> TrajectoryRecord:
    config = FCSRConfig(**{**config.__dict__, "use_information_potential": True})
    if config.iprr_reward_weight == 0.0:
        config = FCSRConfig(**{**config.__dict__, "iprr_reward_weight": 0.25})
    return compute_fcsr_lite_rewards(traj, config)


def compute_rewards_for_trajectory(traj: TrajectoryRecord, config: FCSRConfig | Any) -> TrajectoryRecord:
    if not isinstance(config, FCSRConfig):
        config = FCSRConfig.from_config(config)
    method = config.reward_method
    if method == "outcome_only":
        return _compute_outcome_only(traj, config)
    if method == "dcsr":
        return _compute_dcsr_trajectory(traj, config)
    if method == "fcsr_lite":
        return compute_fcsr_lite_rewards(traj, config)
    if method == "webshop_iprr_lite":
        return compute_webshop_iprr_lite_rewards(traj, config)
    if method == "trca":
        return compute_trca_rewards(traj, config)
    raise ValueError(f"Unsupported reward_method={method}. Expected one of {sorted(SUPPORTED_REWARD_METHODS)} or rewardflow.")


def _env_name_from_config(config: Any) -> str:
    env = getattr(config, "env", None)
    if env is None and isinstance(config, dict):
        env = config.get("env")
    if isinstance(env, dict):
        return str(env.get("env_name", ""))
    try:
        return str(getattr(env, "env_name"))
    except Exception:
        return ""


def _write_rewards_to_raw(traj: TrajectoryRecord, config: FCSRConfig) -> None:
    if config.reward_method == "trca":
        for step in traj.steps:
            if step.raw is None:
                continue
            env_reward = safe_float(
                step.raw.get("env_reward_original", step.raw.get("rewards", step.raw.get("reward", step.env_reward))),
                0.0,
            )
            reward = safe_float(step.mixed_reward, 0.0)
            foundational = safe_float(step.reward_components.get("trca_foundational", 0.0), 0.0)
            breakthrough = safe_float(step.reward_components.get("trca_breakthrough", 0.0), 0.0)
            step.raw["env_reward_original"] = env_reward
            step.raw["trca_score"] = reward
            step.raw["trca_foundational"] = foundational
            step.raw["trca_breakthrough"] = breakthrough
            # Backward-compatible aliases for existing diagnostics only. No
            # terminal anchor, strict gate, clipping, or reward normalization.
            step.raw["fcsr_score"] = reward
            step.raw["fcsr_score_strict"] = reward
            step.raw["step_rewards"] = reward
            step.raw["step_rewards_abs"] = abs(reward)
            if config.log_reward_components:
                step.raw["fcsr_reward_components"] = dict(step.reward_components)
        return
    use_strict_gate = bool(config.webshop_strict_gate_enabled and "webshop" in traj.env_name.lower())
    use_alfworld_gate = bool(config.alfworld_action_gate_enabled and "alf" in traj.env_name.lower())
    for i, step in enumerate(traj.steps):
        if step.raw is None:
            continue
        env_reward = safe_float(
            step.raw.get("env_reward_original", step.raw.get("rewards", step.raw.get("reward", step.env_reward))),
            0.0,
        )
        fcsr_score = safe_float(step.mixed_reward, 0.0)
        fcsr_score_strict = fcsr_score
        gate_reason = "strict_gate_disabled"
        gate_allowed = True
        gate_category = "disabled"
        alfworld_gate_reason = "alfworld_gate_disabled"
        alfworld_gate_allowed = True
        alfworld_gate_category = "disabled"
        alfworld_invalid_clamped = False
        alfworld_noop_repeat_clamped = False
        alfworld_positive_allowed = False
        alfworld_terminal_preserved = False
        if use_strict_gate:
            prev_step = traj.steps[i - 1] if i > 0 else None
            fcsr_score_strict, gate = strict_gate_reward(fcsr_score, step, prev_step=prev_step, traj=traj, cfg=config)
            gate_reason = gate.reason
            gate_allowed = gate.allowed
            gate_category = gate.category
            format_reward = safe_float(step.reward_components.get("format_reward", 0.0), 0.0)
            if format_reward > 0.0 and fcsr_score > 0.0 and not gate_allowed:
                fcsr_score_strict = clip(
                    fcsr_score_strict + format_reward,
                    config.fcsr_strict_clip_min,
                    config.fcsr_strict_clip_max,
                )
        elif use_alfworld_gate:
            prev_step = traj.steps[i - 1] if i > 0 else None
            gate = alfworld_gate_reward(fcsr_score, step, prev_step=prev_step, traj=traj, cfg=config)
            fcsr_score_strict = gate.reward
            alfworld_gate_reason = gate.reason
            alfworld_gate_allowed = gate.allowed
            alfworld_gate_category = gate.category
            alfworld_invalid_clamped = gate.invalid_clamped
            alfworld_noop_repeat_clamped = gate.noop_repeat_clamped
            alfworld_positive_allowed = gate.positive_allowed
            alfworld_terminal_preserved = gate.terminal_preserved
        reward = fcsr_score_strict if (use_strict_gate or use_alfworld_gate) else fcsr_score
        step.reward_components["fcsr_score"] = fcsr_score
        step.reward_components["fcsr_score_strict"] = fcsr_score_strict
        step.reward_components["webshop_gate_allowed"] = float(bool(gate_allowed))
        step.reward_components["alfworld_gate_allowed"] = float(bool(alfworld_gate_allowed))
        step.reward_components["alfworld_gate_invalid_clamped"] = float(bool(alfworld_invalid_clamped))
        step.reward_components["alfworld_gate_noop_repeat_clamped"] = float(bool(alfworld_noop_repeat_clamped))
        step.reward_components["alfworld_gate_positive_allowed"] = float(bool(alfworld_positive_allowed))
        step.reward_components["alfworld_gate_terminal_preserved"] = float(bool(alfworld_terminal_preserved))
        step.raw["env_reward_original"] = env_reward
        step.raw["fcsr_score"] = fcsr_score
        step.raw["fcsr_score_strict"] = fcsr_score_strict
        step.raw["webshop_gate_reason"] = gate_reason
        step.raw["webshop_gate_allowed"] = bool(gate_allowed)
        step.raw["webshop_gate_category"] = gate_category
        step.raw["alfworld_gate_reason"] = alfworld_gate_reason
        step.raw["alfworld_gate_allowed"] = bool(alfworld_gate_allowed)
        step.raw["alfworld_gate_category"] = alfworld_gate_category
        step.raw["step_rewards"] = reward
        step.raw["step_rewards_abs"] = reward
        if config.log_reward_components:
            step.raw["fcsr_reward_components"] = dict(step.reward_components)
            step.raw["fcsr_diagnostic"] = dict(step.diagnostic)


def _set_inactive_defaults(raw_traj: list[dict[str, Any]]) -> None:
    for raw in raw_traj:
        if "step_rewards" not in raw:
            reward = safe_float(raw.get("rewards", raw.get("reward", 0.0)), 0.0)
            raw["step_rewards"] = reward
            raw["step_rewards_abs"] = reward


def _metric_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _is_generic_positive_action(reason: str) -> bool:
    return reason in {
        "rejected_navigation",
        "rejected_generic_search",
        "rejected_generic_click",
        "rejected_wrong_option",
        "rejected_buy_too_early",
    }


def summarize_reward_trajectories(trajectories: list[TrajectoryRecord], method: str, config: FCSRConfig | None = None) -> dict[str, float | str]:
    summary_config = config or FCSRConfig()
    rewards = []
    dcsr = []
    hbrr = []
    fgpo = []
    fgpo_formula_errors = []
    iprr = []
    terminal = []
    format_rewards = []
    env_original = []
    fcsr_scores = []
    fcsr_scores_strict = []
    trca_foundational = []
    trca_breakthrough = []
    gate_reasons: Counter[str] = Counter()
    invalid = 0
    noop = 0
    repeated = 0
    premature = 0
    search_steps = 0
    search_query_valid = 0
    search_retrieval_success = 0
    search_tool_error = 0
    search_repeated_query = 0
    search_question_entity = 0
    search_bridge_candidate = 0
    search_final_answer = 0
    search_answer_format_valid = 0
    search_source_novelty = []
    search_evidence_overlap = []
    search_evidence_gain = []
    search_evidence_coverage = []
    failure_types: dict[str, int] = defaultdict(int)
    steps = 0
    successes = 0
    generic_positive = 0
    gate_positive = 0
    action_format_valid = 0
    action_format_reward_eligible = 0
    alfworld_projection_steps = 0
    alfworld_raw_format_valid = 0
    alfworld_tag_format_valid = 0
    alfworld_bare_admissible_exact_match = 0
    alfworld_admissible_exact_match = 0
    alfworld_repaired_action_valid = 0
    alfworld_executable_action_valid = 0
    alfworld_env_action_sent_empty = 0
    alfworld_multi_action = 0
    alfworld_gate_reasons: Counter[str] = Counter()
    positive_zeroed = []
    goal_relevance = []
    completion_critical = []
    timing_values: dict[str, list[float]] = defaultdict(list)
    task_success: dict[str, list[float]] = defaultdict(list)
    task_lengths: dict[str, list[float]] = defaultdict(list)
    task_invalid: dict[str, int] = defaultdict(int)
    task_noop: dict[str, int] = defaultdict(int)
    task_repeat: dict[str, int] = defaultdict(int)
    task_steps: dict[str, int] = defaultdict(int)
    for traj in trajectories:
        successes += int(bool(traj.final_success))
        for key, value in traj.metadata.items():
            if str(key).startswith("timing_s/"):
                timing_values[str(key)].append(safe_float(value, 0.0))
        traj_task = "unknown"
        for step in traj.steps:
            candidate_task = str(step.diagnostic.get("alfworld_task_type", "unknown"))
            if candidate_task and candidate_task != "unknown":
                traj_task = candidate_task
                break
        if traj_task != "unknown":
            task_success[traj_task].append(float(bool(traj.final_success)))
            task_lengths[traj_task].append(float(len(traj.steps)))
        for step in traj.steps:
            steps += 1
            raw = step.raw or {}
            info = raw.get("infos", {})
            if isinstance(info, dict):
                for key, value in info.items():
                    if str(key).startswith("timing_s/"):
                        timing_values[str(key)].append(safe_float(value, 0.0))
            reward = safe_float(step.mixed_reward, 0.0)
            strict_reward = safe_float(raw.get("fcsr_score_strict", reward), 0.0)
            reason = str(raw.get("webshop_gate_reason", "strict_gate_disabled"))
            rewards.append(reward)
            fcsr_scores.append(safe_float(raw.get("fcsr_score", reward), 0.0))
            fcsr_scores_strict.append(strict_reward)
            trca_foundational.append(safe_float(step.reward_components.get("trca_foundational", 0.0), 0.0))
            trca_breakthrough.append(safe_float(step.reward_components.get("trca_breakthrough", 0.0), 0.0))
            env_original.append(safe_float(raw.get("env_reward_original", step.env_reward), 0.0))
            gate_reasons[reason] += 1
            if reason != "strict_gate_disabled" and strict_reward > 0.0:
                gate_positive += 1
            if strict_reward > 0.0 and _is_generic_positive_action(reason):
                generic_positive += 1
            dcsr.append(safe_float(step.reward_components.get("dcsr_reward", 0.0), 0.0))
            hbrr.append(safe_float(step.reward_components.get("hbrr_reward", 0.0), 0.0))
            if "fgpo_reward" in step.reward_components:
                fgpo_value = safe_float(step.reward_components.get("fgpo_reward", 0.0), 0.0)
                mix_lambda = safe_float(step.reward_components.get("fgpo_mix_lambda", -1.0), -1.0)
                fgpo.append(fgpo_value)
                fgpo_formula_errors.append(
                    abs(fgpo_value - ((1.0 - mix_lambda) * dcsr[-1] + mix_lambda * hbrr[-1]))
                )
            iprr.append(safe_float(step.reward_components.get("iprr_reward", 0.0), 0.0))
            terminal.append(safe_float(step.reward_components.get("terminal_anchor", 0.0), 0.0))
            format_rewards.append(safe_float(step.reward_components.get("format_reward", 0.0), 0.0))
            action_format_valid += int(bool(step.diagnostic.get("action_format_valid", False)))
            action_format_reward_eligible += int(_action_format_reward_eligible(step, summary_config))
            if "raw_format_valid" in step.diagnostic or "admissible_exact_match" in step.diagnostic:
                alfworld_projection_steps += 1
                projected_action = safe_text(step.diagnostic.get("projected_action", ""))
                alfworld_raw_format_valid += int(bool(step.diagnostic.get("raw_format_valid", False)))
                alfworld_tag_format_valid += int(bool(step.diagnostic.get("tag_format_valid", False)))
                alfworld_bare_admissible_exact_match += int(bool(step.diagnostic.get("bare_admissible_exact_match", False)))
                alfworld_admissible_exact_match += int(bool(step.diagnostic.get("admissible_exact_match", False)))
                alfworld_repaired_action_valid += int(bool(step.diagnostic.get("repaired_action_valid", False)))
                alfworld_executable_action_valid += int(bool(step.diagnostic.get("executable_action_valid", False)))
                alfworld_env_action_sent_empty += int(projected_action == "")
                alfworld_multi_action += int(bool(step.diagnostic.get("multi_action", False)))
            alfworld_gate_reasons[str(raw.get("alfworld_gate_category", "disabled"))] += 1
            invalid += int(step.diagnostic.get("valid_action") is False)
            noop += int(bool(step.diagnostic.get("no_op", False)))
            repeated += int(bool(step.diagnostic.get("repeated_action", False)))
            if "search" in step.env_name.lower():
                search_steps += 1
                search_query_valid += int(bool(step.diagnostic.get("query_valid", False)))
                search_retrieval_success += int(bool(step.diagnostic.get("retrieval_success", False)))
                search_tool_error += int(bool(step.diagnostic.get("tool_error", False)))
                search_repeated_query += int(bool(step.diagnostic.get("repeated_query", False)))
                search_question_entity += int(bool(step.diagnostic.get("query_contains_question_entity", False)))
                search_bridge_candidate += int(bool(step.diagnostic.get("query_contains_bridge_candidate", False)))
                search_final_answer += int(bool(step.diagnostic.get("final_answer_present", False)))
                search_answer_format_valid += int(bool(step.diagnostic.get("answer_format_valid", False)))
                search_source_novelty.append(safe_float(step.diagnostic.get("source_novelty", 0.0), 0.0))
                search_evidence_overlap.append(safe_float(step.diagnostic.get("evidence_overlap", 0.0), 0.0))
                search_evidence_gain.append(safe_float(step.diagnostic.get("evidence_gain", 0.0), 0.0))
                search_evidence_coverage.append(safe_float(step.diagnostic.get("evidence_coverage", 0.0), 0.0))
            positive_zeroed.append(safe_float(step.reward_components.get("positive_zeroed_no_progress", 0.0), 0.0))
            goal_relevance.append(safe_float(step.diagnostic.get("goal_relevance", 0.0), 0.0))
            completion_critical.append(float(bool(step.diagnostic.get("completion_critical_action", False))))
            step_task = str(step.diagnostic.get("alfworld_task_type", "unknown"))
            if step_task != "unknown":
                task_steps[step_task] += 1
                task_invalid[step_task] += int(step.diagnostic.get("valid_action") is False)
                task_noop[step_task] += int(bool(step.diagnostic.get("no_op", False)))
                task_repeat[step_task] += int(bool(step.diagnostic.get("repeated_action", False)))
            premature += int(bool(step.diagnostic.get("premature_finish", False) or step.diagnostic.get("premature_buy", False)))
            failure_types[str(step.diagnostic.get("failure_type", "unknown"))] += 1
    gate_steps = steps - gate_reasons["strict_gate_disabled"]
    alfworld_gate_steps = steps - alfworld_gate_reasons["disabled"]
    metrics: dict[str, float | str] = {
        "event": "fcsr_batch_metrics",
        "reward_method": method,
        "fcsr/trajectories": float(len(trajectories)),
        "fcsr/steps": float(steps),
        "fcsr/final_success_rate": float(successes / len(trajectories)) if trajectories else 0.0,
        "fcsr/reward_mean": _metric_mean(rewards),
        "fcsr/reward_positive_rate": float(sum(1 for reward in rewards if reward > 0) / len(rewards)) if rewards else 0.0,
        "fcsr/dcsr_mean": _metric_mean(dcsr),
        "fcsr/hbrr_mean": _metric_mean(hbrr),
        "fcsr/fgpo_reward_mean": _metric_mean(fgpo),
        "fcsr/fgpo_mix_lambda": safe_float(summary_config.fgpo_mix_lambda, -1.0),
        "fcsr/fgpo_formula_max_abs_error": max(fgpo_formula_errors) if fgpo_formula_errors else 0.0,
        "fcsr/iprr_mean": _metric_mean(iprr),
        "fcsr/terminal_anchor_mean": _metric_mean(terminal),
        "diagnostic/action_format_valid_rate": float(action_format_valid / steps) if steps else 0.0,
        "diagnostic/action_format_reward_eligible_rate": float(action_format_reward_eligible / steps) if steps else 0.0,
        "alfworld/raw_format_valid_ratio": float(alfworld_raw_format_valid / alfworld_projection_steps) if alfworld_projection_steps else 0.0,
        "alfworld/tag_format_valid_ratio": float(alfworld_tag_format_valid / alfworld_projection_steps) if alfworld_projection_steps else 0.0,
        "alfworld/bare_admissible_exact_match_ratio": float(alfworld_bare_admissible_exact_match / alfworld_projection_steps) if alfworld_projection_steps else 0.0,
        "alfworld/admissible_exact_match_ratio": float(alfworld_admissible_exact_match / alfworld_projection_steps) if alfworld_projection_steps else 0.0,
        "alfworld/repaired_action_valid_ratio": float(alfworld_repaired_action_valid / alfworld_projection_steps) if alfworld_projection_steps else 0.0,
        "alfworld/executable_action_valid_ratio": float(alfworld_executable_action_valid / alfworld_projection_steps) if alfworld_projection_steps else 0.0,
        "alfworld/env_action_sent_empty_ratio": float(alfworld_env_action_sent_empty / alfworld_projection_steps) if alfworld_projection_steps else 0.0,
        "alfworld/multi_action_rate": float(alfworld_multi_action / alfworld_projection_steps) if alfworld_projection_steps else 0.0,
        "diagnostic/positive_zeroed_no_progress_rate": float(sum(1 for value in positive_zeroed if value > 0.0) / len(positive_zeroed)) if positive_zeroed else 0.0,
        "diagnostic/goal_relevance_mean": _metric_mean(goal_relevance),
        "diagnostic/completion_critical_action_rate": _metric_mean(completion_critical),
        "reward/format_reward_mean": _metric_mean(format_rewards),
        "reward/format_reward_max": max(format_rewards) if format_rewards else 0.0,
        "reward/format_to_terminal_ratio": float(sum(format_rewards) / max(1e-8, abs(sum(terminal)))) if terminal else 0.0,
        "fcsr/invalid_rate": float(invalid / steps) if steps else 0.0,
        "fcsr/noop_rate": float(noop / steps) if steps else 0.0,
        "fcsr/repeated_rate": float(repeated / steps) if steps else 0.0,
        "fcsr/repeat_rate": float(repeated / steps) if steps else 0.0,
        "fcsr/premature_rate": float(premature / steps) if steps else 0.0,
        "search/query_valid_rate": float(search_query_valid / search_steps) if search_steps else 0.0,
        "search/retrieval_success_rate": float(search_retrieval_success / search_steps) if search_steps else 0.0,
        "search/tool_error_rate": float(search_tool_error / search_steps) if search_steps else 0.0,
        "search/repeated_query_rate": float(search_repeated_query / search_steps) if search_steps else 0.0,
        "search/query_contains_question_entity_rate": float(search_question_entity / search_steps) if search_steps else 0.0,
        "search/query_contains_bridge_candidate_rate": float(search_bridge_candidate / search_steps) if search_steps else 0.0,
        "search/final_answer_present_rate": float(search_final_answer / search_steps) if search_steps else 0.0,
        "search/answer_format_valid_rate": float(search_answer_format_valid / search_steps) if search_steps else 0.0,
        "search/source_novelty_mean": _metric_mean(search_source_novelty),
        "search/evidence_overlap_mean": _metric_mean(search_evidence_overlap),
        "search/evidence_gain_mean": _metric_mean(search_evidence_gain),
        "search/evidence_coverage_mean": _metric_mean(search_evidence_coverage),
        "reward/env_reward_original_mean": _metric_mean(env_original),
        "reward/fcsr_score_mean": _metric_mean(fcsr_scores),
        "reward/fcsr_score_strict_mean": _metric_mean(fcsr_scores_strict),
        "reward/fcsr_score_strict_positive_rate": float(sum(1 for reward in fcsr_scores_strict if reward > 0) / len(fcsr_scores_strict)) if fcsr_scores_strict else 0.0,
        "trca/foundational_mean": _metric_mean(trca_foundational),
        "trca/breakthrough_mean": _metric_mean(trca_breakthrough),
        "webshop_gate/positive_rate": float(gate_positive / gate_steps) if gate_steps else 0.0,
        "webshop_gate/rejected_navigation": float(gate_reasons["rejected_navigation"] / gate_steps) if gate_steps else 0.0,
        "webshop_gate/rejected_generic_search": float(gate_reasons["rejected_generic_search"] / gate_steps) if gate_steps else 0.0,
        "webshop_gate/rejected_generic_click": float(gate_reasons["rejected_generic_click"] / gate_steps) if gate_steps else 0.0,
        "webshop_gate/rejected_wrong_option": float(gate_reasons["rejected_wrong_option"] / gate_steps) if gate_steps else 0.0,
        "webshop_gate/rejected_buy_too_early": float(gate_reasons["rejected_buy_too_early"] / gate_steps) if gate_steps else 0.0,
        "webshop_gate/allowed_product_click": float((gate_reasons["product_click_goal_match"] + gate_reasons["product_click_single_result_goal_match"]) / gate_steps) if gate_steps else 0.0,
        "webshop_gate/allowed_goal_option": float(sum(count for reason, count in gate_reasons.items() if reason.startswith("goal_option_")) / gate_steps) if gate_steps else 0.0,
        "webshop_gate/allowed_buy_ready": float(gate_reasons["buy_ready_all_required_options_matched"] / gate_steps) if gate_steps else 0.0,
        "webshop_gate/generic_action_positive_rate": float(generic_positive / gate_steps) if gate_steps else 0.0,
        "alfworld_gate/invalid_clamped": float(alfworld_gate_reasons["invalid_clamped"] / alfworld_gate_steps) if alfworld_gate_steps else 0.0,
        "alfworld_gate/noop_repeat_clamped": float(alfworld_gate_reasons["noop_repeat_clamped"] / alfworld_gate_steps) if alfworld_gate_steps else 0.0,
        "alfworld_gate/positive_allowed": float(alfworld_gate_reasons["positive_allowed"] / alfworld_gate_steps) if alfworld_gate_steps else 0.0,
        "alfworld_gate/terminal_preserved": float(alfworld_gate_reasons["terminal_preserved"] / alfworld_gate_steps) if alfworld_gate_steps else 0.0,
    }
    for ftype, count in sorted(failure_types.items()):
        metrics[f"fcsr/failure_type/{ftype}"] = float(count / steps) if steps else 0.0
    for key, values in sorted(timing_values.items()):
        metrics[key] = _metric_mean(values)
    for task in ("pick", "pick_two", "look", "clean", "heat", "cool", "put"):
        if task in task_success:
            metrics[f"alfworld_task/{task}_success"] = _metric_mean(task_success[task])
            metrics[f"alfworld_task/{task}_avg_length"] = _metric_mean(task_lengths[task])
        if task in task_steps:
            denom = max(1, task_steps[task])
            metrics[f"alfworld_task/{task}_invalid_rate"] = float(task_invalid[task] / denom)
            metrics[f"alfworld_task/{task}_noop_rate"] = float(task_noop[task] / denom)
            metrics[f"alfworld_task/{task}_repeat_rate"] = float(task_repeat[task] / denom)
    return metrics

def compute_rewards_for_batch(total_batch_list: list[list[dict[str, Any]]], config: Any) -> tuple[list[list[dict[str, Any]]], dict[str, float | str]]:
    start_time = time.perf_counter()
    method = reward_method_from_config(config)
    if method == "rewardflow":
        return total_batch_list, {}
    fcfg = FCSRConfig.from_config(config)
    if fcfg.reward_method not in SUPPORTED_REWARD_METHODS:
        raise ValueError(f"Unsupported reward_method={fcfg.reward_method}")
    env_name = _env_name_from_config(config)
    trajectories = [
        trajectory_from_existing_rollout(raw_traj, env_name=env_name, success_reward_threshold=fcfg.success_reward_threshold)
        for raw_traj in total_batch_list
    ]
    for traj in trajectories:
        compute_rewards_for_trajectory(traj, fcfg)
        _write_rewards_to_raw(traj, fcfg)
    for raw_traj in total_batch_list:
        _set_inactive_defaults(raw_traj)
    metrics = summarize_reward_trajectories(trajectories, method, fcfg)
    metrics["timing_s/fcsr_reward"] = time.perf_counter() - start_time
    if fcfg.debug_dump_rewards:
        path = dump_reward_trajectories(trajectories, fcfg.debug_dump_dir, prefix=method)
        metrics["fcsr/debug_dump_path"] = path
    print("FCSR_METRIC_JSON: " + json.dumps(metrics, sort_keys=True))
    return total_batch_list, metrics
