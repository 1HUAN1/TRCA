# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List
import os
import re


_LAST_PROJECTION_INFOS: list[dict[str, Any]] = []


def _normalize_action(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"<\|(?:im_end|endoftext)\|>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*action\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`")
    text = re.sub(r"[\s\.;,!。！？；：]+$", "", text)
    return " ".join(text.lower().split())


def _extract_first_action_block(text: str) -> tuple[str, bool, int]:
    matches = re.findall(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return "", False, 0
    return matches[0].strip(), True, len(matches)


def _has_think_block(text: str) -> bool:
    return bool(re.search(r"<think>\s*.*?\s*</think>", text, flags=re.IGNORECASE | re.DOTALL))


def _prompt_style() -> str:
    style = os.environ.get("ALFWORLD_PROMPT_STYLE", "think_action")
    return str(style or "think_action").strip().lower().replace("-", "_")


def _requires_think_block() -> bool:
    return _prompt_style() not in {"action", "actiononly", "action_only", "action_only_bare", "action_only_tagged"}


def _allows_bare_admissible() -> bool:
    flag = os.environ.get("ALFWORLD_ALLOW_BARE_ADMISSIBLE", "")
    flag_enabled = str(flag).strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(flag_enabled or _prompt_style() in {"action_only_bare", "bare", "action_bare"})


def _clean_bare_candidate(text: str) -> tuple[str, bool, str]:
    """Return a safe bare command candidate when the whole response is one action."""

    cleaned = re.sub(r"<\|(?:im_end|endoftext)\|>", "", text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("\"'`")
    if not cleaned:
        return "", False, "empty_action"
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return "", False, "contains_chinese"
    if re.search(r"</?(?:action|think)>", cleaned, flags=re.IGNORECASE):
        return "", False, "unexpected_xml_tag"

    non_empty_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(non_empty_lines) > 1:
        return "", False, "multiple_bare_actions"

    lower = cleaned.lower()
    explanation_patterns = (
        r"\bi think\b",
        r"\bwe should\b",
        r"\bthe best action is\b",
        r"\bbest action is\b",
        r"\bi will\b",
        r"\blet'?s\b",
        r"\blet us\b",
        r"\bbecause\b",
        r"\bfirst\b.*\bthen\b",
    )
    if any(re.search(pattern, lower) for pattern in explanation_patterns):
        return "", False, "explanatory_text"
    if re.search(r"\b(?:and then|then|followed by|after that)\b", lower):
        return "", False, "multiple_bare_actions"
    if ";" in cleaned:
        return "", False, "multiple_bare_actions"

    normalized = _normalize_action(cleaned)
    if not normalized:
        return "", False, "empty_action"
    return cleaned, True, ""


def _match_admissible_action(action_text: str, admissible_actions: list[str] | tuple[str, ...] | None) -> tuple[str, bool]:
    normalized_action = _normalize_action(action_text)
    if not normalized_action or not admissible_actions:
        return action_text, False
    for candidate in admissible_actions:
        if _normalize_action(candidate) == normalized_action:
            return str(candidate), True
    return action_text, False


def get_last_projection_infos() -> list[dict[str, Any]]:
    return list(_LAST_PROJECTION_INFOS)


def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    Project model outputs to AlfWorld actions.

    A model action is executable when it is either:
    1. one clean <action>...</action> block whose content exactly matches an
       admissible command after normalization; or
    2. in action_only_bare mode, the whole raw response is one bare command that
       exactly matches an admissible command after normalization.

    The final command sent to the environment is always the original admissible
    command. There is no fuzzy repair and no fallback action.
    """

    global _LAST_PROJECTION_INFOS
    projected_actions: list[str] = []
    valids: list[int] = []
    infos: list[dict[str, Any]] = []

    for i, raw in enumerate(actions):
        raw_text = "" if raw is None else str(raw)
        action_pool = action_pools[i] if action_pools is not None and i < len(action_pools) else []
        raw_action_text, action_tag_found, action_tag_count = _extract_first_action_block(raw_text)
        multi_action = action_tag_count > 1
        think_tag_found = _has_think_block(raw_text)
        think_required = _requires_think_block()
        bare_allowed = _allows_bare_admissible()
        contains_chinese = bool(re.search(r"[\u4e00-\u9fff]", raw_text))
        bare_action_candidate = ""
        bare_command_parseable = False
        bare_parse_error = ""

        if action_tag_found:
            candidate_action = raw_action_text
            action_source = "tag"
        else:
            bare_action_candidate, bare_command_parseable, bare_parse_error = _clean_bare_candidate(raw_text)
            candidate_action = bare_action_candidate if bare_allowed and bare_command_parseable else ""
            action_source = "bare" if candidate_action else "none"

        normalized_action = _normalize_action(candidate_action)
        matched_action, admissible_exact_match = _match_admissible_action(candidate_action, action_pool)

        tag_format_valid = bool(
            action_tag_found
            and action_tag_count == 1
            and (think_tag_found or not think_required)
            and not contains_chinese
            and normalized_action
        )
        bare_admissible_exact_match = bool(
            not action_tag_found
            and bare_allowed
            and bare_command_parseable
            and admissible_exact_match
            and not contains_chinese
        )
        executable_action_valid = bool(
            admissible_exact_match
            and not contains_chinese
            and (tag_format_valid or bare_admissible_exact_match)
        )
        raw_format_valid = bool(tag_format_valid or bare_admissible_exact_match)
        metric_action_valid = raw_format_valid

        invalid_reasons: list[str] = []
        if not action_tag_found and not (bare_allowed and bare_command_parseable):
            invalid_reasons.append("missing_action_tag")
        if multi_action:
            invalid_reasons.append("multiple_action_tags")
        if think_required and not think_tag_found:
            invalid_reasons.append("missing_think_tag")
        if contains_chinese:
            invalid_reasons.append("contains_chinese")
        if bare_parse_error and not action_tag_found and not bare_admissible_exact_match:
            invalid_reasons.append(bare_parse_error)
        if not normalized_action:
            invalid_reasons.append("empty_action")
        if not admissible_exact_match:
            invalid_reasons.append("not_in_admissible")

        projected_action = matched_action if executable_action_valid else ""
        projected_actions.append(projected_action)
        valids.append(1 if metric_action_valid else 0)
        infos.append(
            {
                "raw_format_valid": raw_format_valid,
                "tag_format_valid": bool(tag_format_valid),
                "action_tag_found": bool(action_tag_found),
                "has_action_tag": bool(action_tag_found),
                "multi_action": bool(multi_action),
                "think_tag_found": bool(think_tag_found),
                "think_tag_required": bool(think_required),
                "contains_chinese": bool(contains_chinese),
                "bare_action_candidate": bare_action_candidate,
                "bare_command_parseable": bool(bare_command_parseable),
                "bare_admissible_exact_match": bool(bare_admissible_exact_match),
                "admissible_exact_match": bool(admissible_exact_match),
                "matched_admissible_action": matched_action if admissible_exact_match else "",
                "repaired_action_valid": bool(executable_action_valid),
                "executable_action_valid": bool(executable_action_valid),
                "is_action_valid": bool(metric_action_valid),
                "projected_action": projected_action,
                "raw_action_text": raw_action_text,
                "normalized_action": normalized_action,
                "action_source": action_source,
                "invalid_reason": ",".join(dict.fromkeys(invalid_reasons)) if invalid_reasons else "",
            }
        )

    _LAST_PROJECTION_INFOS = infos
    alfworld_projection.last_infos = infos
    return projected_actions, valids


alfworld_projection.last_infos = []
