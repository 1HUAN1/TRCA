"""Canonical observable transition facts for ALFWorld, WebShop, and SearchQA."""

from __future__ import annotations

import json
import re
from typing import Any


_OUTCOME_KEYS = {
    "success", "is_success", "task_success", "won", "episode_rewards", "final_score", "final_reward",
    "task_score", "score", "reward", "rewards", "answer", "answers", "answer_aliases", "ground_truth",
}
_MISSING = object()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _tokens(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_.-]*", _text(value).casefold())


def _safe_public_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if str(key).casefold() not in _OUTCOME_KEYS}


def _first(containers: list[dict[str, Any]], names: tuple[str, ...], default: Any = None) -> Any:
    for container in containers:
        for name in names:
            if name in container and container[name] is not None:
                return container[name]
    return default


def transition_facts(step: Any, previous_step: Any | None = None) -> dict[str, Any]:
    raw = step.raw if isinstance(getattr(step, "raw", None), dict) else {}
    diagnostics = dict(getattr(step, "diagnostic", {}) or {})
    current = _text(getattr(step, "observation", "") or getattr(step, "state", ""))
    following = _text(getattr(step, "tool_output", "") or getattr(step, "next_state", ""))
    raw_output = _text(raw.get("output", raw.get("response", "")))
    action = str(getattr(step, "action", "") or "").strip()
    matched_tag = False
    for tag in ("action", "search", "answer"):
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", raw_output, flags=re.I | re.S)
        if match:
            matched_tag = True
            content = " ".join(match.group(1).split())
            action = content if tag == "action" else f"{tag} {content}"
            break
    action_tokens = _tokens(action)
    action_type = str(getattr(step, "action_type", "") or (action_tokens[0] if action_tokens else ""))
    env_info = _safe_public_mapping(raw.get("infos"))
    admissible = diagnostics.get("admissible_actions", [])
    if not admissible and isinstance(env_info.get("available_actions"), dict):
        admissible = env_info["available_actions"].get("clickables", [])
    admissible = list(admissible) if isinstance(admissible, (list, tuple, set)) else []
    valid = diagnostics.get("valid_action")
    parser_valid = diagnostics.get("tag_format_valid", diagnostics.get("action_format_valid", matched_tag))
    if diagnostics.get("multi_action"):
        parser_valid = False
    previous_action = str(getattr(previous_step, "action", "") or "").strip().casefold() if previous_step else ""
    current_tokens, next_tokens = set(_tokens(current)), set(_tokens(following))
    accepted = None if valid is None else valid is True
    rejected = None if valid is None else valid is False
    changed = None if not following else _text(current).strip() != _text(following).strip()
    repeated = bool(action and previous_action and action.casefold() == previous_action)
    no_op = diagnostics.get("no_op")
    if no_op is None and accepted is True and changed is not None:
        no_op = not changed
    for key, value in raw.items():
        if str(key).casefold() in _OUTCOME_KEYS or key in {"input_ids", "responses", "prompts"}:
            continue
        if key.startswith(("alfworld_", "webshop_", "search_")):
            env_info[key] = value
    sources = [env_info, diagnostics, _safe_public_mapping(raw)]
    # Export only the paper's canonical observable fields. Unknown fields stay
    # absent, so the evaluator cannot confuse an unsupported value with false.
    canonical_env: dict[str, Any] = {}
    aliases = {
        "visible_entities": ("visible_entities", "entities", "objects"),
        "inventory": ("inventory", "inventory_items", "carrying"),
        "object_locations": ("object_locations", "locations"),
        "object_states": ("object_states", "states"),
        "page_type": ("page_type", "page", "webshop_page_type"),
        "product_ids": ("product_ids", "result_ids", "asins"),
        "visible_attributes": ("visible_attributes", "attributes"),
        "available_options": ("available_options", "options"),
        "selected_options": ("selected_options", "selected"),
        "price": ("price", "product_price"),
        "cart_state": ("cart_state", "cart"),
        "purchase_status": ("purchase_status", "purchased"),
        "query": ("query", "search_query"),
        "result_ids": ("result_ids", "search_result_ids"),
        "passage_ids": ("passage_ids", "document_ids", "source_ids"),
        "retrieved_text": ("retrieved_text", "passage", "document"),
        "tool_status": ("tool_status", "status"),
        "answer_submitted": ("answer_submitted", "submitted"),
    }
    for target, names in aliases.items():
        value = _first(sources, names, _MISSING)
        if value is not _MISSING:
            canonical_env[target] = value

    available_actions = env_info.get("available_actions")
    if "available_options" not in canonical_env and isinstance(available_actions, dict):
        clickables = available_actions.get("clickables")
        if isinstance(clickables, (list, tuple, set)):
            canonical_env["available_options"] = list(clickables)

    observable_text = f"{current}\n{following}"
    if "price" not in canonical_env:
        prices = {float(item) for item in re.findall(r"\$\s*(\d+(?:\.\d+)?)", observable_text)}
        if len(prices) == 1:
            canonical_env["price"] = prices.pop()
    if "selected_options" not in canonical_env:
        selected = re.findall(r"you have clicked\s+['\"]?([^'\"\n.,;\]]+)", observable_text, flags=re.I)
        if selected:
            canonical_env["selected_options"] = list(dict.fromkeys(item.strip() for item in selected))

    search_match = re.search(r"<search>\s*(.*?)\s*</search>", raw_output, flags=re.I | re.S)
    if search_match:
        canonical_env["query"] = search_match.group(1).strip()
    if re.search(r"<answer>.*?</answer>", raw_output, flags=re.I | re.S):
        canonical_env["answer_submitted"] = True
    if following and "retrieved_text" not in canonical_env:
        canonical_env["retrieved_text"] = following
    return {
        "action": {
            "text": action,
            "type": action_type,
            "argument": " ".join(action_tokens[1:]),
            "tag_present": matched_tag,
            "parser_valid": bool(parser_valid),
            "admissible": None if not admissible else action.casefold() in {str(item).strip().casefold() for item in admissible},
            "available_actions": admissible,
        },
        "transition": {
            "accepted": accepted,
            "rejected": rejected,
            "state_changed": changed,
            "repeated": repeated,
            "no_op": no_op,
            "tool_error": diagnostics.get("tool_error"),
        },
        "observation": {
            "current_text": current,
            "next_text": following,
            "current_tokens": sorted(current_tokens),
            "next_tokens": sorted(next_tokens),
            "new_tokens": sorted(next_tokens - current_tokens),
            "new_text": " ".join(sorted(next_tokens - current_tokens)),
        },
        "state": {
            "current": getattr(step, "state", None),
            "next": getattr(step, "next_state", None),
        },
        "env": canonical_env,
    }
