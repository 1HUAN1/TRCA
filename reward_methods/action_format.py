"""Action-format checks for lightweight reward regularization."""

from __future__ import annotations

import re
from typing import Any


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    return str(value)


def normalize_action(action: str) -> str:
    text = _safe_text(action).strip()
    match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    return " ".join(text.strip().split())


def safe_normalize_action_text(action: str) -> str:
    """Conservative formatting cleanup that does not do semantic rewriting."""
    text = normalize_action(action)
    text = re.sub(r"^\s*action\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("\"'` \t\r\n")
    text = re.sub(r"[\s\.;,!]+$", "", text).strip()
    return " ".join(text.lower().split())


def safe_repair_action(action: str, admissible_actions: list[str] | tuple[str, ...] | None = None) -> tuple[str, str, bool, str]:
    """Return a safe exact-match repair candidate and diagnostic metadata.

    The repair is allowed only when the normalized text exactly matches one of
    the admissible actions after the same conservative formatting cleanup.
    """
    raw = normalize_action(action)
    normalized = safe_normalize_action_text(action)
    if not normalized:
        return raw, normalized, False, "empty_normalized_action"
    if not admissible_actions:
        return raw, normalized, False, "no_admissible_actions"
    for candidate in admissible_actions:
        if safe_normalize_action_text(candidate) == normalized:
            repaired = normalize_action(candidate)
            changed = repaired != raw
            return repaired, normalized, bool(changed), "normalized_exact_admissible_match"
    return raw, normalized, False, "normalized_not_admissible"


def is_alfworld_action_format_valid(action: str) -> bool:
    text = normalize_action(action).lower()
    if not text:
        return False
    patterns = (
        r"^look$",
        r"^inventory$",
        r"^go to .+$",
        r"^open .+$",
        r"^close .+$",
        r"^take .+ from .+$",
        r"^(put|move) .+ (in|on|to) .+$",
        r"^(clean|heat|cool|slice) .+ with .+$",
        r"^use .+$",
        r"^toggle .+$",
        r"^examine .+$",
    )
    return any(re.match(pattern, text) for pattern in patterns)


def is_webshop_action_format_valid(action: str) -> bool:
    text = normalize_action(action)
    if not text:
        return False
    return bool(re.match(r"^(search|click)\s*\[.+\]\s*$", text, flags=re.IGNORECASE | re.DOTALL))


def is_action_format_valid(action: str, env_name: str) -> tuple[bool, str, str]:
    text = normalize_action(action)
    if not text:
        return False, "empty", "empty_action"

    env = _safe_text(env_name).lower()
    if "webshop" in env:
        match = re.match(r"^(search|click)\s*\[.+\]\s*$", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return True, f"webshop_{match.group(1).lower()}", ""
        return False, "webshop", "webshop_parse_error"

    if "alf" in env:
        lowered = text.lower()
        action_type = lowered.split(" ", 1)[0]
        if lowered.startswith("go to "):
            action_type = "navigate"
        if is_alfworld_action_format_valid(text):
            return True, f"alfworld_{action_type}", ""
        return False, "alfworld", "alfworld_parse_error"

    if "search" in env:
        search_matches = re.findall(r"<search>\s*(.*?)\s*</search>", text, flags=re.IGNORECASE | re.DOTALL)
        answer_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
        has_one_search = len(search_matches) == 1 and bool(search_matches[0].strip())
        has_one_answer = len(answer_matches) == 1 and bool(answer_matches[0].strip())
        if has_one_search and not answer_matches:
            return True, "search_query", ""
        if has_one_answer and not search_matches:
            return True, "search_answer", ""
        return False, "search", "search_parse_error"

    return True, "generic", "unknown_env"
