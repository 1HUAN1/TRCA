"""Small numeric utilities and config normalization for FCSR-lite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    try:
        if hasattr(value, "item"):
            value = value.item()
        return bool(value)
    except (TypeError, ValueError):
        return default


def _lookup(container: Any, name: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(name, default)
    if hasattr(container, "get"):
        try:
            value = container.get(name, default)
            if value is not None:
                return value
        except Exception:
            pass
    try:
        value = getattr(container, name)
        if value is not None:
            return value
    except Exception:
        pass
    return default


def config_get(config: Any, name: str, default: Any = None) -> Any:
    """Read a config value from top level, fcsr block, then algorithm block."""
    value = _lookup(config, name, None)
    if value is not None:
        return value
    for block_name in ("fcsr", "algorithm"):
        block = _lookup(config, block_name, None)
        value = _lookup(block, name, None)
        if value is not None:
            return value
    nested_aliases = {
        "webshop_strict_gate_enabled": ("webshop_strict_gate", "enabled"),
        "webshop_allow_search_reward": ("webshop_strict_gate", "allow_search_reward"),
        "webshop_allow_navigation_positive_reward": ("webshop_strict_gate", "allow_navigation_positive_reward"),
        "webshop_allow_generic_click_positive_reward": ("webshop_strict_gate", "allow_generic_click_positive_reward"),
        "webshop_allow_goal_option_prefix": ("webshop_strict_gate", "allow_goal_option_prefix"),
    }
    if name in nested_aliases:
        block_name, option_name = nested_aliases[name]
        block = _lookup(config, block_name, None)
        value = _lookup(block, option_name, None)
        if value is not None:
            return value
        for parent_name in ("fcsr", "algorithm"):
            parent = _lookup(config, parent_name, None)
            block = _lookup(parent, block_name, None)
            value = _lookup(block, option_name, None)
            if value is not None:
                return value
    return default


def reward_method_from_config(config: Any) -> str:
    method = config_get(config, "reward_method", "rewardflow")
    return str(method or "rewardflow").strip().lower()


@dataclass(frozen=True)
class FCSRConfig:
    reward_method: str = "rewardflow"
    alfworld_training_chain: str = "standard"
    alfworld_progress_gated_reward: bool = False
    alfworld_action_gate_enabled: bool = False
    alfworld_gate_invalid_clamp: float = -0.1
    alfworld_gate_noop_repeat_clamp: float = -0.05
    alfworld_goal_relevance_threshold: float = 0.35
    alfworld_ray_get_timeout_s: float = 0.0
    use_diagnostics: bool = True
    use_failure_rewriting: bool = False
    use_information_potential: bool = False
    use_counterfactual: bool = False
    use_fclar: bool = False
    use_action_format_reward: bool = True
    format_reward_apply: str = "terminal"
    format_reward_weight: float = 0.03
    format_reward_max: float = 0.05
    format_reward_require_not_invalid: bool = True

    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0
    reward_normalization: str = "none"
    group_normalization: bool = True

    terminal_reward_weight: float = 1.0
    dense_reward_weight: float = 0.5
    hbrr_reward_weight: float = 0.5
    iprr_reward_weight: float = 0.0
    # Opt-in paper lambda. Negative values preserve the legacy independent
    # dense/HBRR weighting used by existing runs.
    fgpo_mix_lambda: float = -1.0

    # Paper-aligned TRCA configuration. The rubric library is generated once
    # before training and then loaded by the deterministic runtime evaluator.
    trca_rubric_dir: str = "trca_rubrics"
    trca_rubric_library: str = ""
    trca_mix_lambda: float = 0.8
    trca_evidence_budget: float = 1.0
    trca_invalidity_budget: float = 1.0
    trca_execution_budget: float = 1.0

    episode_adv_weight: float = 0.5
    step_adv_weight: float = 0.5

    dcsr_alpha_progress: float = 1.0
    dcsr_beta_validity: float = 0.1
    dcsr_eta_recovery: float = 0.2
    dcsr_lambda_risk: float = 0.5

    penalty_invalid: float = 0.2
    penalty_noop: float = 0.1
    penalty_repeat: float = 0.1
    penalty_premature: float = 0.4
    penalty_deadlock: float = 0.7

    progress_positive_threshold: float = 0.05
    recovery_risk_threshold: float = 0.3
    deadlock_threshold: float = 0.7
    repeat_window: int = 3

    hbrr_alpha_delta_q: float = 0.5
    hbrr_beta_progress: float = 0.1
    hbrr_gamma_exploratory: float = 0.1
    hbrr_eta_recovery: float = 0.2
    hbrr_bottleneck_penalty: float = 0.4
    hbrr_irreversible_penalty: float = 0.6
    hbrr_redundant_penalty: float = 0.15
    hbrr_premature_penalty: float = 0.4

    hbrr_w_progress: float = 1.0
    hbrr_w_risk: float = 1.0
    hbrr_w_info: float = 0.5
    hbrr_w_verifier: float = 0.0

    iprr_gamma: float = 0.99
    iprr_w_attribute: float = 1.0
    iprr_w_candidate: float = 0.5
    iprr_w_option: float = 0.5
    iprr_w_premature: float = 1.0
    iprr_w_wrong_attribute: float = 0.5

    webshop_strict_gate_enabled: bool = False
    webshop_allow_search_reward: bool = False
    webshop_allow_navigation_positive_reward: bool = False
    webshop_allow_generic_click_positive_reward: bool = False
    webshop_allow_goal_option_prefix: bool = True
    fcsr_strict_clip_min: float = -0.2
    fcsr_strict_clip_max: float = 0.2

    debug_dump_rewards: bool = False
    debug_dump_dir: str = "debug_reward_dumps"
    log_reward_components: bool = True
    log_reward_hacking_metrics: bool = True

    critical_top_k: int = 2
    counterfactual_num_actions: int = 0
    counterfactual_num_rollouts: int = 0
    success_reward_threshold: float = 0.99

    @classmethod
    def from_config(cls, config: Any) -> "FCSRConfig":
        values: dict[str, Any] = {}
        for name in cls.__annotations__:
            default = getattr(cls, name)
            raw = config_get(config, name, default)
            if isinstance(default, bool):
                values[name] = safe_bool(raw, default)
            elif isinstance(default, int) and not isinstance(default, bool):
                values[name] = int(safe_float(raw, float(default)))
            elif isinstance(default, float):
                values[name] = safe_float(raw, float(default))
            else:
                values[name] = str(raw)
        values["reward_method"] = str(values["reward_method"]).lower()
        return cls(**values)


def normalize_rewards_in_trajectory(traj: Any, field: str = "mixed_reward", config: FCSRConfig | None = None) -> Any:
    """Normalize rewards in a single trajectory to bounded zero-mean scale."""
    if config is None:
        config = FCSRConfig()
    rewards = [safe_float(getattr(step, field, 0.0), 0.0) for step in traj.steps]
    if len(rewards) <= 1:
        return traj
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(max(variance, 1e-8))
    for step in traj.steps:
        value = (safe_float(getattr(step, field, 0.0), 0.0) - mean) / std
        setattr(step, field, clip(value, config.reward_clip_min, config.reward_clip_max))
        step.reward_components[f"{field}_normalized"] = getattr(step, field)
    return traj
