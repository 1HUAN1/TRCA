"""Deterministic benchmark-specific task binding for shared rubric operators."""

from __future__ import annotations

import re
from typing import Any


_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "find", "for", "from", "get", "i", "in",
    "is", "it", "me", "of", "on", "or", "please", "put", "some", "that", "the", "then", "to",
    "what", "which", "who", "with", "would", "your",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_.-]*", text.casefold())


def _key_terms(text: str) -> list[str]:
    return list(dict.fromkeys(token for token in _tokens(text) if len(token) > 2 and token not in _STOP))


def _metadata_bindings(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    candidate = metadata.get("trca_bindings", metadata.get("task_bindings", {}))
    return dict(candidate) if isinstance(candidate, dict) else {}


def _alfworld(instruction: str) -> dict[str, Any]:
    lowered = instruction.casefold()
    operations = [name for name in ("clean", "cool", "heat", "take", "pick", "put", "place", "look", "examine") if re.search(rf"\b{name}\w*\b", lowered)]
    quantity = 2 if re.search(r"\b(two|2)\b", lowered) else 1
    destinations = re.findall(r"\b(?:in|into|on|onto)\s+(?:the\s+|a\s+|an\s+)?([a-z][a-z0-9_-]*)", lowered)
    candidates = re.findall(r"\b(?:clean|cool|heat|take|pick(?: up)?|put|place|examine)\s+(?:two\s+|2\s+|the\s+|a\s+|an\s+|some\s+)?([a-z][a-z0-9_-]*)", lowered)
    targets = [item for item in candidates if item not in {"it", "them"}]
    return {
        "target_entities": list(dict.fromkeys(targets)),
        "destination_receptacles": list(dict.fromkeys(destinations)),
        "required_operations": list(dict.fromkeys(operations)),
        "required_quantity": quantity,
    }


def _webshop(instruction: str) -> dict[str, Any]:
    lowered = instruction.casefold()
    price_match = re.search(
        r"(?:under|below|less than|lower than|at most|no more than)\s*\$?\s*(\d+(?:\.\d+)?)",
        lowered,
    )
    quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", instruction)
    return {
        "requested_terms": _key_terms(instruction),
        "requested_options": list(dict.fromkeys(quoted)),
        "max_price": float(price_match.group(1)) if price_match else None,
        "purchase_required": bool(re.search(r"\b(buy|purchase|order)\b", lowered)),
    }


def _searchqa(instruction: str) -> dict[str, Any]:
    capitalized = re.findall(r"\b(?:[A-Z][\w.-]+(?:\s+[A-Z][\w.-]+){0,4})\b", instruction)
    answer_format = "text"
    if re.search(r"\b(when|what year|date)\b", instruction, flags=re.IGNORECASE):
        answer_format = "date"
    elif re.search(r"\b(how many|number)\b", instruction, flags=re.IGNORECASE):
        answer_format = "number"
    elif re.search(r"\b(who|which person)\b", instruction, flags=re.IGNORECASE):
        answer_format = "person"
    return {
        "target_entities": list(dict.fromkeys(capitalized)),
        "information_need_terms": _key_terms(instruction),
        "query_constraints": [],
        "required_answer_format": answer_format,
    }


def bind_task(benchmark: str, instruction: str, metadata: Any = None) -> dict[str, Any]:
    """Bind task fields without an LLM and without ground-truth answer aliases."""
    benchmark = benchmark.casefold()
    base = {
        "instruction": instruction,
        "instruction_tokens": _tokens(instruction),
        "task_relevant_terms": _key_terms(instruction),
    }
    if benchmark == "alfworld":
        base.update(_alfworld(instruction))
    elif benchmark == "webshop":
        base.update(_webshop(instruction))
    elif benchmark == "searchqa":
        base.update(_searchqa(instruction))
    else:
        raise ValueError(f"Unsupported TRCA benchmark: {benchmark}")
    base.update(_metadata_bindings(metadata))
    # Explicitly remove prohibited completion labels if supplied by upstream metadata.
    for key in ("answer", "answers", "answer_aliases", "ground_truth", "terminal_outcome", "success"):
        base.pop(key, None)
    return base
