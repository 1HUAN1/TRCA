"""AlfWorld-only action validity gate for FCSR-lite rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reward_methods.normalization import safe_float


@dataclass(frozen=True)
class AlfWorldGateResult:
    reward: float
    reason: str
    category: str
    allowed: bool
    invalid_clamped: bool = False
    noop_repeat_clamped: bool = False
    positive_allowed: bool = False
    terminal_preserved: bool = False


def _is_terminal_success(step: Any, traj: Any) -> bool:
    if not bool(getattr(traj, "final_success", False)):
        return False
    steps = getattr(traj, "steps", []) or []
    is_last = bool(steps and step is steps[-1])
    terminal_anchor = safe_float(getattr(step, "reward_components", {}).get("terminal_anchor", 0.0), 0.0)
    return bool(is_last and terminal_anchor > 0.0)


def alfworld_gate_reward(score: float, step: Any, prev_step: Any | None = None, traj: Any | None = None, cfg: Any | None = None) -> AlfWorldGateResult:
    """Clamp unsafe AlfWorld step rewards without touching terminal success.

    This gate is deliberately conservative: it never fuzzy-repairs actions, and
    it treats projection exact-admissible failure as invalid even if the raw env
    feedback is ambiguous.
    """

    del prev_step
    diag = getattr(step, "diagnostic", {}) or {}
    score = safe_float(score, 0.0)
    invalid_clamp = min(safe_float(getattr(cfg, "alfworld_gate_invalid_clamp", -0.1), -0.1), 0.0)
    noop_repeat_clamp = min(safe_float(getattr(cfg, "alfworld_gate_noop_repeat_clamp", -0.05), -0.05), 0.0)
    terminal_preserved = bool(traj is not None and _is_terminal_success(step, traj))
    valid_action = diag.get("valid_action")
    exact_match = diag.get("admissible_exact_match")
    invalid = bool(valid_action is False or exact_match is False)
    noop_or_repeat = bool(diag.get("no_op", False) or diag.get("repeated_action", False))

    if terminal_preserved:
        return AlfWorldGateResult(
            reward=score,
            reason="terminal_success_preserved",
            category="terminal_preserved",
            allowed=True,
            positive_allowed=score > 0.0,
            terminal_preserved=True,
        )

    if invalid:
        return AlfWorldGateResult(
            reward=min(score, invalid_clamp),
            reason=str(diag.get("projection_invalid_reason") or "invalid_or_not_admissible"),
            category="invalid_clamped",
            allowed=False,
            invalid_clamped=True,
        )

    if noop_or_repeat:
        reason = "repeat" if bool(diag.get("repeated_action", False)) else "noop"
        return AlfWorldGateResult(
            reward=min(score, noop_repeat_clamp),
            reason=reason,
            category="noop_repeat_clamped",
            allowed=False,
            noop_repeat_clamped=True,
        )

    return AlfWorldGateResult(
        reward=score,
        reason="allowed",
        category="positive_allowed" if score > 0.0 else "allowed_non_positive",
        allowed=True,
        positive_allowed=score > 0.0,
    )
