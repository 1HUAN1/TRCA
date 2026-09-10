"""Rule-based diagnostics for FCSR-lite."""

from __future__ import annotations

from typing import Any
import re

from reward_methods.action_format import is_action_format_valid, safe_normalize_action_text, safe_repair_action
from reward_methods.normalization import FCSRConfig, clip, safe_float
from reward_methods.schema import StepRecord, TrajectoryRecord, infer_action_type, safe_bool, safe_text
from reward_methods.webshop_strict import classify_webshop_action


INVALID_PHRASES = (
    "invalid",
    "not a valid action",
    "can't",
    "cannot",
    "nothing happens",
    "unknown action",
    "you can't",
    "you cannot",
)
NOOP_ACTIONS = {"", "noop", "no op", "no-op", "wait", "pass", "none", "null", "n/a", "do nothing"}
SEARCH_ERROR_PHRASES = (
    "connection refused",
    "connection error",
    "failed to connect",
    "http error",
    "no search results found",
    "search request failed",
    "unknown api state",
    "timed out",
    "timeout",
    "service unavailable",
    "internal server error",
    "retrieval error",
)
ALFWORLD_ACTIONS = {"navigate", "open", "close", "take", "put", "look", "examine", "inventory", "heat", "cool", "clean", "toggle"}
STOP_WORDS = {
    "the",
    "a",
    "an",
    "with",
    "to",
    "in",
    "on",
    "of",
    "and",
    "or",
    "your",
    "task",
    "is",
    "be",
    "from",
    "into",
    "using",
    "for",
}


def default_diagnostic() -> dict[str, Any]:
    return {
        "valid_action": None,
        "state_changed": None,
        "no_op": False,
        "repeated_action": False,
        "tool_error": False,
        "subgoal_progress": 0.0,
        "progress_delta": 0.0,
        "risk": 0.0,
        "risk_delta": 0.0,
        "deadlock_risk": 0.0,
        "evidence_gain": 0.0,
        "evidence_coverage": 0.0,
        "attribute_coverage": 0.0,
        "premature_finish": False,
        "premature_buy": False,
        "recoverability": 0.0,
        "failure_type": "unknown",
        "verifier_signal": 0.0,
        "action_format_valid": None,
        "action_format_type": "",
        "action_parse_error": "",
        "raw_action": "",
        "normalized_action": "",
        "action_repaired": False,
        "action_repair_reason": "",
        "alfworld_task_type": "unknown",
        "goal_object_match": False,
        "goal_receptacle_match": False,
        "required_tool_match": False,
        "target_action_match": False,
        "goal_relevance": 0.0,
        "simple_action_type": "unknown",
        "completion_critical_action": False,
    }


def canonical_action(action: str) -> str:
    return " ".join(safe_text(action).lower().strip().split())


def _contains_any(text: str, phrases: tuple[str, ...] | set[str]) -> bool:
    lower = safe_text(text).lower()
    return any(phrase in lower for phrase in phrases)


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", safe_text(text).lower()) if tok not in STOP_WORDS and len(tok) > 2}


def _goal_tokens(step: StepRecord) -> set[str]:
    task = safe_text(step.diagnostic.get("task_instruction", ""))
    return _tokenize(task)


def _is_alfworld_optimized(cfg: FCSRConfig) -> bool:
    return str(getattr(cfg, "alfworld_training_chain", "standard")).strip().lower() == "optimized"


def _phrase_tokens(text: str) -> set[str]:
    return _tokenize(text)


def _extract_phrase_after(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    phrase = match.group(1)
    phrase = re.split(r"\b(?:with|in|on|to|from|inside|into|and|then|using)\b", phrase, maxsplit=1, flags=re.IGNORECASE)[0]
    phrase = re.sub(r"\b(?:a|an|the|some|two|one)\b", " ", phrase, flags=re.IGNORECASE)
    return " ".join(phrase.strip(" .,!?:;").split())


def parse_alfworld_task_instruction(task: str) -> dict[str, Any]:
    """Best-effort AlfWorld task parser used only for diagnostics."""
    text = safe_text(task).lower()
    task_type = "unknown"
    if "two" in text and ("pick" in text or "place" in text or "put" in text):
        task_type = "pick_two"
    elif any(word in text for word in ("look at", "examine")):
        task_type = "look"
    elif "clean" in text:
        task_type = "clean"
    elif "heat" in text:
        task_type = "heat"
    elif "cool" in text:
        task_type = "cool"
    elif any(word in text for word in ("put", "place")):
        task_type = "put"
    elif any(word in text for word in ("pick", "take")):
        task_type = "pick"

    target_action = {
        "pick": "take",
        "pick_two": "take",
        "look": "examine",
        "clean": "clean",
        "heat": "heat",
        "cool": "cool",
        "put": "put",
    }.get(task_type, "")

    target_object = ""
    for pattern in (
        r"(?:examine|look at)\s+(?:a|an|the|some)?\s*([^\\.]+)",
        r"(?:clean|heat|cool|slice)\s+(?:a|an|the|some)?\s*([^\\.]+)",
        r"(?:pick up|pick|take)\s+(?:a|an|the|some|two)?\s*([^\\.]+)",
        r"(?:put|place|move)\s+(?:a|an|the|some)?\s*([^\\.]+)",
    ):
        target_object = _extract_phrase_after(pattern, text)
        if target_object:
            break

    required_tool = _extract_phrase_after(r"(?:with|using)\s+(?:a|an|the|some)?\s*([^\\.]+)", text)
    target_receptacle = _extract_phrase_after(r"(?:in|on|to|into|inside)\s+(?:a|an|the|some)?\s*([^\\.]+)", text)

    return {
        "task_type": task_type,
        "target_action": target_action,
        "target_object": target_object,
        "target_receptacle": target_receptacle,
        "required_tool": required_tool,
        "target_location": target_receptacle,
    }


def _simple_action_type(action: str) -> str:
    text = safe_normalize_action_text(action)
    if text in {"look"}:
        return "look"
    if text == "inventory":
        return "inventory"
    if text.startswith("go to "):
        return "navigation"
    if text.startswith("open ") or text.startswith("close "):
        return "open_close"
    if text.startswith("take ") or text.startswith("pick "):
        return "take"
    if text.startswith("put ") or text.startswith("move ") or text.startswith("place "):
        return "put"
    if text.startswith(("clean ", "heat ", "cool ", "slice ", "toggle ", "use ")):
        return "use_tool"
    if text.startswith("examine "):
        return "goal_action"
    return "unknown"


def _tokens_match(needle: str, haystack: str) -> bool:
    tokens = _phrase_tokens(needle)
    if not tokens:
        return False
    hay_tokens = _phrase_tokens(haystack)
    return tokens.issubset(hay_tokens)


def _apply_alfworld_goal_diagnostics(diag: dict[str, Any], step: StepRecord, cfg: FCSRConfig) -> dict[str, Any]:
    admissible = diag.get("admissible_actions") or []
    repaired, normalized, action_repaired, repair_reason = safe_repair_action(step.action, admissible)
    action_for_match = repaired if action_repaired else normalized
    task_info = parse_alfworld_task_instruction(diag.get("task_instruction", ""))

    action_type = _simple_action_type(action_for_match)
    goal_object_match = _tokens_match(task_info["target_object"], action_for_match)
    goal_receptacle_match = _tokens_match(task_info["target_receptacle"], action_for_match)
    required_tool_match = _tokens_match(task_info["required_tool"], action_for_match)
    target_action_match = False
    if task_info["target_action"]:
        target_action_match = action_for_match.startswith(task_info["target_action"] + " ")
        if task_info["task_type"] == "look":
            target_action_match = target_action_match or action_for_match.startswith("use ") or action_for_match.startswith("toggle ")

    relevance = 0.0
    relevance += 0.45 * float(goal_object_match)
    relevance += 0.25 * float(goal_receptacle_match)
    relevance += 0.30 * float(required_tool_match)
    relevance += 0.20 * float(target_action_match)
    if action_type == "navigation":
        relevance = min(relevance, 0.25)
    if action_type in {"look", "inventory", "open_close"} and not (goal_object_match or required_tool_match):
        relevance = min(relevance, 0.15)
    relevance = clip(relevance, 0.0, 1.0)

    blocked = bool(diag.get("no_op") or diag.get("repeated_action") or diag.get("valid_action") is False)
    completion_critical = bool(
        not blocked
        and action_type not in {"look", "inventory", "navigation", "open_close"}
        and (goal_object_match or goal_receptacle_match or required_tool_match or target_action_match)
    )
    if task_info["task_type"] == "look" and not blocked:
        completion_critical = bool((goal_object_match and target_action_match) or required_tool_match)

    diag.update(
        {
            "raw_action": safe_text(step.action),
            "normalized_action": normalized,
            "action_repaired": bool(action_repaired),
            "action_repair_reason": repair_reason,
            "alfworld_task_type": task_info["task_type"],
            "alfworld_target_action": task_info["target_action"],
            "alfworld_target_object": task_info["target_object"],
            "alfworld_target_receptacle": task_info["target_receptacle"],
            "alfworld_required_tool": task_info["required_tool"],
            "goal_object_match": bool(goal_object_match),
            "goal_receptacle_match": bool(goal_receptacle_match),
            "required_tool_match": bool(required_tool_match),
            "target_action_match": bool(target_action_match),
            "goal_relevance": relevance,
            "simple_action_type": action_type,
            "completion_critical_action": completion_critical,
        }
    )
    return diag


def _state_changed(step: StepRecord, invalid: bool) -> bool:
    if invalid:
        return False
    current = safe_text(step.observation or step.state).strip().lower()
    nxt = safe_text(step.tool_output or step.next_state).strip().lower()
    if not nxt:
        return None
    return current != nxt


def _repeated_action(step: StepRecord, traj: TrajectoryRecord | None, cfg: FCSRConfig) -> bool:
    if traj is None or step.step_id <= 0:
        return False
    current = canonical_action(step.action)
    if not current or current in NOOP_ACTIONS:
        return False
    start = max(0, step.step_id - int(cfg.repeat_window))
    for prev in traj.steps[start:step.step_id]:
        if canonical_action(prev.action) == current:
            return True
    return False


def _base_diagnostic(step: StepRecord, prev_step: StepRecord | None, traj: TrajectoryRecord | None, cfg: FCSRConfig) -> dict[str, Any]:
    diag = default_diagnostic()
    diag.update(step.diagnostic or {})
    text = " ".join([safe_text(step.observation), safe_text(step.tool_output), safe_text(step.next_state)])
    action = canonical_action(step.action)
    inferred_invalid = _contains_any(text, INVALID_PHRASES)
    valid_action = diag.get("valid_action")
    if valid_action is None:
        valid_action = False if inferred_invalid else None
    else:
        valid_action = safe_bool(valid_action, default=None)
    invalid = valid_action is False
    state_changed = _state_changed(step, invalid)
    no_op = invalid or action in NOOP_ACTIONS or (state_changed is False)
    repeated = _repeated_action(step, traj, cfg)
    repaired, normalized, action_repaired, repair_reason = safe_repair_action(step.action, diag.get("admissible_actions") or [])

    action_format_valid, action_format_type, action_parse_error = is_action_format_valid(step.action, step.env_name)

    diag.update(
        {
            "valid_action": valid_action,
            "state_changed": state_changed,
            "no_op": bool(no_op),
            "repeated_action": bool(repeated),
            "tool_error": bool(inferred_invalid),
            "action_format_valid": bool(action_format_valid),
            "action_format_type": action_format_type,
            "action_parse_error": action_parse_error,
            "raw_action": safe_text(step.action),
            "normalized_action": normalized,
            "action_repaired": bool(action_repaired),
            "action_repair_reason": repair_reason,
        }
    )
    return diag


def _finish_without_success(step: StepRecord, traj: TrajectoryRecord | None) -> bool:
    action = canonical_action(step.action)
    if not any(token in action for token in ("finish", "done", "submit", "stop", "answer")):
        return False
    return not bool(traj and traj.final_success)


def parse_alfworld_diagnostics(
    step: StepRecord,
    prev_step: StepRecord | None,
    traj: TrajectoryRecord | None,
    cfg: FCSRConfig,
) -> dict[str, Any]:
    diag = _base_diagnostic(step, prev_step, traj, cfg)
    optimized = _is_alfworld_optimized(cfg)
    if optimized:
        diag = _apply_alfworld_goal_diagnostics(diag, step, cfg)
    diagnostic_action = step.action
    if diag.get("admissible_exact_match") is True and diag.get("projected_action"):
        diagnostic_action = safe_text(diag.get("projected_action"))
    action = canonical_action(diag.get("normalized_action") if optimized else diagnostic_action)
    action_type = infer_action_type(action) if optimized else infer_action_type(action)
    diag["action_used_for_diagnostics"] = action
    goal_tokens = _goal_tokens(step)
    obs_tokens = _tokenize(step.observation + " " + step.tool_output)
    action_tokens = _tokenize(action)
    goal_overlap = len(goal_tokens & (obs_tokens | action_tokens))
    goal_ratio = goal_overlap / max(1, min(len(goal_tokens), 6))

    progress = 0.0
    evidence_gain = 0.0
    if action_type == "navigate" and diag["state_changed"]:
        progress += 0.12
    if action_type in {"open", "look", "examine"} and diag["state_changed"]:
        progress += 0.18
        evidence_gain += 0.15
    if action_type in {"take", "put", "heat", "cool", "clean", "toggle"} and diag["state_changed"]:
        progress += 0.35
    if action_type == "inventory" and diag["state_changed"]:
        evidence_gain += 0.05
    if goal_ratio > 0:
        progress += min(0.45, 0.12 + 0.40 * goal_ratio)
    if action_type == "examine" and goal_ratio > 0:
        progress += 0.20
    if diag["valid_action"] is False or diag["no_op"]:
        progress = min(progress, 0.02)
        evidence_gain = 0.0
    if optimized:
        relevance = safe_float(diag.get("goal_relevance", 0.0), 0.0)
        completion_critical = bool(diag.get("completion_critical_action", False))
        simple_type = str(diag.get("simple_action_type", "unknown"))
        threshold = safe_float(getattr(cfg, "alfworld_goal_relevance_threshold", 0.35), 0.35)
        if diag["valid_action"] is False or diag["no_op"] or diag["repeated_action"]:
            progress = min(progress, 0.02)
            evidence_gain = 0.0
            diag["completion_critical_action"] = False
        elif not completion_critical:
            if simple_type in {"look", "inventory", "navigation", "open_close"}:
                progress = min(progress, 0.03)
                evidence_gain = min(evidence_gain, 0.05)
            elif relevance < threshold:
                progress = min(progress, 0.03)
        else:
            progress = max(progress, min(1.0, 0.35 + 0.40 * relevance))

    prev_progress = safe_float(prev_step.diagnostic.get("subgoal_progress", 0.0), 0.0) if prev_step else 0.0
    progress = clip(progress, 0.0, 1.0)
    progress_delta = progress - prev_progress
    if progress > 0 and progress_delta <= 0 and diag["state_changed"]:
        progress_delta = min(0.10, progress * 0.25)

    premature = _finish_without_success(step, traj)
    risk = 0.0
    risk += 0.55 * float(diag["valid_action"] is False)
    risk += 0.35 * float(diag["no_op"])
    risk += 0.20 * float(diag["repeated_action"])
    risk += 0.15 * float(action_type == "navigate" and goal_ratio == 0 and step.step_id >= 3)
    risk += 0.40 * float(premature)
    prev_risk = safe_float(prev_step.diagnostic.get("risk", 0.0), 0.0) if prev_step else 0.0

    diag.update(
        {
            "subgoal_progress": progress,
            "progress_delta": clip(progress_delta, -1.0, 1.0),
            "risk": clip(risk, 0.0, 1.0),
            "risk_delta": clip(risk - prev_risk, -1.0, 1.0),
            "evidence_gain": clip(evidence_gain, 0.0, 1.0),
            "evidence_coverage": clip(goal_ratio, 0.0, 1.0),
            "premature_finish": premature,
        }
    )
    return diag


def _extract_webshop_terms(text: str) -> set[str]:
    tokens = _tokenize(text)
    generic = {"find", "product", "item", "price", "review", "rating", "webshop", "shop", "buy"}
    return {tok for tok in tokens if tok not in generic}


def parse_webshop_diagnostics(
    step: StepRecord,
    prev_step: StepRecord | None,
    traj: TrajectoryRecord | None,
    cfg: FCSRConfig,
) -> dict[str, Any]:
    diag = _base_diagnostic(step, prev_step, traj, cfg)
    task_terms = _extract_webshop_terms(safe_text(step.diagnostic.get("task_instruction", "")))
    seen_terms = _extract_webshop_terms(" ".join([step.observation, step.tool_output, step.action]))
    coverage = len(task_terms & seen_terms) / max(1, min(len(task_terms), 8))
    gate = classify_webshop_action(step, prev_step=prev_step, traj=traj, cfg=cfg)

    progress = 0.0
    evidence_gain = 0.0
    if gate.allowed:
        if gate.category == "allowed_product_click":
            progress += 0.40
            evidence_gain += 0.25
        elif gate.category == "allowed_goal_option":
            progress += 0.25
            evidence_gain += 0.10
        elif gate.category == "allowed_buy_ready":
            progress += 0.55
        progress += min(0.20, 0.20 * coverage)

    if diag["valid_action"] is False or diag["no_op"]:
        progress = min(progress, 0.02)
        evidence_gain = 0.0

    premature_buy = gate.reason == "rejected_buy_too_early"
    prev_progress = safe_float(prev_step.diagnostic.get("subgoal_progress", 0.0), 0.0) if prev_step else 0.0
    progress = clip(progress, 0.0, 1.0)
    progress_delta = progress - prev_progress
    if progress > 0 and progress_delta <= 0 and diag["state_changed"]:
        progress_delta = min(0.08, progress * 0.20)

    risk = 0.0
    risk += 0.50 * float(diag["valid_action"] is False)
    risk += 0.30 * float(diag["no_op"])
    risk += 0.15 * float(diag["repeated_action"])
    risk += 0.70 * float(premature_buy)
    risk += 0.10 * float(gate.category.startswith("rejected") and not premature_buy)
    prev_risk = safe_float(prev_step.diagnostic.get("risk", 0.0), 0.0) if prev_step else 0.0
    diag.update(
        {
            "subgoal_progress": progress,
            "progress_delta": clip(progress_delta, -1.0, 1.0),
            "risk": clip(risk, 0.0, 1.0),
            "risk_delta": clip(risk - prev_risk, -1.0, 1.0),
            "evidence_gain": clip(evidence_gain, 0.0, 1.0),
            "evidence_coverage": clip(evidence_gain, 0.0, 1.0),
            "attribute_coverage": clip(coverage, 0.0, 1.0),
            "premature_buy": premature_buy,
            "webshop_gate_reason": gate.reason,
            "webshop_gate_allowed": gate.allowed,
            "webshop_gate_category": gate.category,
        }
    )
    return diag

def _sokoban_deadlock_risk(text: str) -> float:
    rows = [line.rstrip() for line in safe_text(text).splitlines() if line.strip()]
    if len(rows) < 3:
        return 0.0
    width = max(len(row) for row in rows)
    grid = [row.ljust(width) for row in rows]
    boxes = "$B"
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char not in boxes:
                continue
            up = r == 0 or grid[r - 1][c] == "#"
            down = r == len(grid) - 1 or grid[r + 1][c] == "#"
            left = c == 0 or grid[r][c - 1] == "#"
            right = c == width - 1 or grid[r][c + 1] == "#"
            if (up or down) and (left or right):
                return 1.0
    return 0.0


def parse_sokoban_diagnostics(
    step: StepRecord,
    prev_step: StepRecord | None,
    traj: TrajectoryRecord | None,
    cfg: FCSRConfig,
) -> dict[str, Any]:
    diag = _base_diagnostic(step, prev_step, traj, cfg)
    deadlock = _sokoban_deadlock_risk(step.tool_output or step.observation)
    progress = 0.15 if diag["state_changed"] and not diag["no_op"] else 0.0
    prev_progress = safe_float(prev_step.diagnostic.get("subgoal_progress", 0.0), 0.0) if prev_step else 0.0
    risk = clip(0.30 * float(diag["no_op"]) + deadlock, 0.0, 1.0)
    prev_risk = safe_float(prev_step.diagnostic.get("risk", 0.0), 0.0) if prev_step else 0.0
    diag.update(
        {
            "subgoal_progress": progress,
            "progress_delta": clip(progress - prev_progress, -1.0, 1.0),
            "risk": risk,
            "risk_delta": clip(risk - prev_risk, -1.0, 1.0),
            "deadlock_risk": deadlock,
        }
    )
    return diag


def parse_deepresearch_diagnostics(
    step: StepRecord,
    prev_step: StepRecord | None,
    traj: TrajectoryRecord | None,
    cfg: FCSRConfig,
) -> dict[str, Any]:
    diag = _base_diagnostic(step, prev_step, traj, cfg)
    text = " ".join([step.action, step.tool_output, step.observation]).lower()
    evidence_markers = ("http", "source", "citation", "paper", "abstract", "doi", "arxiv", "result")
    evidence_gain = 0.2 if any(marker in text for marker in evidence_markers) else 0.0
    progress = evidence_gain + (0.1 if diag["state_changed"] and not diag["no_op"] else 0.0)
    prev_progress = safe_float(prev_step.diagnostic.get("subgoal_progress", 0.0), 0.0) if prev_step else 0.0
    risk = clip(0.4 * float(diag["no_op"]) + 0.3 * float(diag["repeated_action"]), 0.0, 1.0)
    prev_risk = safe_float(prev_step.diagnostic.get("risk", 0.0), 0.0) if prev_step else 0.0
    diag.update(
        {
            "subgoal_progress": clip(progress, 0.0, 1.0),
            "progress_delta": clip(progress - prev_progress, -1.0, 1.0),
            "risk": risk,
            "risk_delta": clip(risk - prev_risk, -1.0, 1.0),
            "evidence_gain": evidence_gain,
            "evidence_coverage": evidence_gain,
        }
    )
    return diag



def _extract_tag_value(text: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}>\s*(.*?)\s*</{tag}>", safe_text(text), flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return ""
    return " ".join(matches[-1].split())


def _previous_search_queries(traj: TrajectoryRecord | None, step_id: int) -> list[str]:
    if traj is None:
        return []
    queries = []
    for prev in traj.steps[:step_id]:
        query = _extract_tag_value(prev.action, "search")
        if query:
            queries.append(canonical_action(query))
    return queries


def _search_evidence_text(step: StepRecord) -> str:
    """Return post-action retrieval evidence without falling back to the question."""
    return safe_text(step.tool_output or step.next_state)


def _previous_search_evidence_tokens(traj: TrajectoryRecord | None, step_id: int) -> set[str]:
    if traj is None or step_id <= 0:
        return set()
    tokens: set[str] = set()
    for prev in traj.steps[:step_id]:
        tokens |= _tokenize(_search_evidence_text(prev))
    return tokens


def _source_novelty(step: StepRecord, traj: TrajectoryRecord | None) -> float:
    current_tokens = _tokenize(_search_evidence_text(step))
    if not current_tokens:
        return 0.0
    if traj is None or step.step_id <= 0:
        return 1.0
    previous_tokens = _previous_search_evidence_tokens(traj, step.step_id)
    if not previous_tokens:
        return 1.0
    novel = current_tokens - previous_tokens
    return clip(len(novel) / max(1, len(current_tokens)), 0.0, 1.0)


def parse_search_diagnostics(
    step: StepRecord,
    prev_step: StepRecord | None,
    traj: TrajectoryRecord | None,
    cfg: FCSRConfig,
) -> dict[str, Any]:
    explicit_valid_action = step.diagnostic.get("valid_action")
    explicit_tool_error_flag = safe_bool(step.diagnostic.get("tool_error"), default=False)
    diag = _base_diagnostic(step, prev_step, traj, cfg)
    query = _extract_tag_value(step.action, "search")
    answer = _extract_tag_value(step.action, "answer")
    question = safe_text(step.diagnostic.get("task_instruction", ""))
    question_tokens = _tokenize(question)
    query_tokens = _tokenize(query)
    evidence_text = _search_evidence_text(step)
    evidence_tokens = _tokenize(evidence_text)

    mixed_search_and_answer = bool(query and answer)
    query_valid = bool(query and query_tokens and not mixed_search_and_answer)
    final_answer_present = bool(answer)
    answer_format_valid = bool(final_answer_present and not mixed_search_and_answer)
    repeated_query = bool(query and canonical_action(query) in _previous_search_queries(traj, step.step_id))
    explicit_tool_error = _contains_any(evidence_text, SEARCH_ERROR_PHRASES)
    tool_error = bool(explicit_tool_error_flag or explicit_tool_error or (query_valid and not evidence_tokens))
    retrieval_success = bool(query_valid and evidence_tokens and not tool_error)
    source_novelty = _source_novelty(step, traj) if retrieval_success else 0.0

    evidence_overlap = 0.0
    if question_tokens and evidence_tokens:
        evidence_overlap = len(question_tokens & evidence_tokens) / max(1, min(len(question_tokens), 12))
        evidence_overlap = clip(evidence_overlap, 0.0, 1.0)
    query_contains_question_entity = bool(question_tokens and query_tokens and question_tokens & query_tokens)
    previous_evidence_tokens = _previous_search_evidence_tokens(traj, step.step_id)
    bridge_candidates = {token for token in previous_evidence_tokens - question_tokens if len(token) > 3}
    query_contains_bridge_candidate = bool(query_tokens & bridge_candidates)

    previous_evidence = safe_float(prev_step.diagnostic.get("evidence_coverage", 0.0), 0.0) if prev_step else 0.0
    previous_progress = safe_float(prev_step.diagnostic.get("subgoal_progress", 0.0), 0.0) if prev_step else 0.0
    evidence_gain = 0.0
    if retrieval_success and not repeated_query:
        bridge_bonus = 0.10 if query_contains_bridge_candidate else 0.0
        if evidence_overlap > 0.0 or query_contains_bridge_candidate:
            evidence_gain = clip(
                0.05 + 0.25 * evidence_overlap + 0.10 * source_novelty + bridge_bonus,
                0.0,
                0.45,
            )

    if mixed_search_and_answer:
        diag["valid_action"] = False
    elif explicit_valid_action is None and (query_valid or final_answer_present):
        diag["valid_action"] = True
    if final_answer_present:
        diag["state_changed"] = True
        diag["no_op"] = False
    elif query_valid:
        diag["state_changed"] = retrieval_success
        diag["no_op"] = not retrieval_success
    diag["repeated_action"] = bool(diag.get("repeated_action") or repeated_query)

    if diag["valid_action"] is False or diag["no_op"] or repeated_query:
        evidence_gain = 0.0

    evidence_coverage = clip(previous_evidence + evidence_gain, 0.0, 1.0)
    progress = clip(previous_progress + evidence_gain, 0.0, 1.0)
    final_success = bool(traj and traj.final_success)
    if final_answer_present and final_success:
        progress = 1.0
    premature_answer = bool(final_answer_present and not final_success and evidence_coverage < 0.15)

    risk = 0.0
    risk += 0.35 * float(diag["valid_action"] is False)
    risk += 0.20 * float(diag["no_op"])
    risk += 0.25 * float(repeated_query)
    risk += 0.40 * float(tool_error)
    risk += 0.35 * float(premature_answer)
    risk += 0.10 * float(final_answer_present and not answer_format_valid)
    prev_risk = safe_float(prev_step.diagnostic.get("risk", 0.0), 0.0) if prev_step else 0.0

    diag.update(
        {
            "hotpotqa_question_type": "unknown",
            "query_valid": query_valid,
            "query_contains_question_entity": query_contains_question_entity,
            "query_contains_bridge_candidate": query_contains_bridge_candidate,
            "retrieval_success": retrieval_success,
            "source_novelty": source_novelty,
            "evidence_overlap": evidence_overlap,
            "supporting_fact_hit": False,
            "evidence_coverage": evidence_coverage,
            "bridge_progress": 0.10 if query_contains_bridge_candidate and retrieval_success and not repeated_query else 0.0,
            "comparison_progress": 0.0,
            "repeated_query": repeated_query,
            "tool_error": tool_error,
            "premature_answer": premature_answer,
            "premature_finish": premature_answer,
            "final_answer_present": final_answer_present,
            "answer_format_valid": answer_format_valid,
            "answer_f1_proxy": 0.0,
            "subgoal_progress": progress,
            "progress_delta": clip(progress - previous_progress, -1.0, 1.0),
            "risk": clip(risk, 0.0, 1.0),
            "risk_delta": clip(risk - prev_risk, -1.0, 1.0),
            "evidence_gain": evidence_gain,
            "failure_type": "repeated_query" if repeated_query else ("tool_error" if tool_error else "unknown"),
        }
    )
    return diag


def parse_generic_diagnostics(
    step: StepRecord,
    prev_step: StepRecord | None,
    traj: TrajectoryRecord | None,
    cfg: FCSRConfig,
) -> dict[str, Any]:
    diag = _base_diagnostic(step, prev_step, traj, cfg)
    progress = 0.1 if diag["state_changed"] and not diag["no_op"] else 0.0
    prev_progress = safe_float(prev_step.diagnostic.get("subgoal_progress", 0.0), 0.0) if prev_step else 0.0
    risk = clip(0.5 * float(diag["valid_action"] is False) + 0.3 * float(diag["no_op"]) + 0.2 * float(diag["repeated_action"]), 0.0, 1.0)
    prev_risk = safe_float(prev_step.diagnostic.get("risk", 0.0), 0.0) if prev_step else 0.0
    diag.update(
        {
            "subgoal_progress": progress,
            "progress_delta": clip(progress - prev_progress, -1.0, 1.0),
            "risk": risk,
            "risk_delta": clip(risk - prev_risk, -1.0, 1.0),
            "premature_finish": _finish_without_success(step, traj),
        }
    )
    return diag


def parse_diagnostics(
    step: StepRecord,
    prev_step: StepRecord | None,
    traj: TrajectoryRecord | None,
    config: FCSRConfig,
) -> dict[str, Any]:
    """Parse diagnostics with environment-specific lightweight rules."""
    env_name = safe_text(step.env_name).lower()
    if "alf" in env_name:
        return parse_alfworld_diagnostics(step, prev_step, traj, config)
    if "webshop" in env_name:
        return parse_webshop_diagnostics(step, prev_step, traj, config)
    if "search" in env_name:
        return parse_search_diagnostics(step, prev_step, traj, config)
    if "sokoban" in env_name:
        return parse_sokoban_diagnostics(step, prev_step, traj, config)
    if "deep" in env_name:
        return parse_deepresearch_diagnostics(step, prev_step, traj, config)
    return parse_generic_diagnostics(step, prev_step, traj, config)
