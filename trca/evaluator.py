"""Safe deterministic evaluator for TRCA rubric conditions."""

from __future__ import annotations

import re
from typing import Any

from trca.schema import RubricOperator


_MISSING = object()


def dotted_get(container: Any, path: str, default: Any = _MISSING) -> Any:
    value = container
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def _text(value: Any) -> str:
    if value is None or value is _MISSING:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value)
    return str(value).casefold()


def _sequence(value: Any) -> list[Any]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _resolve_rhs(condition: dict[str, Any], facts: dict[str, Any], bindings: dict[str, Any]) -> Any:
    if "binding" in condition:
        return dotted_get(bindings, str(condition["binding"]))
    if "other_fact" in condition:
        return dotted_get(facts, str(condition["other_fact"]))
    return condition.get("value", _MISSING)


def _condition_inputs(condition: dict[str, Any]) -> tuple[set[str], set[str]]:
    facts: set[str] = set()
    bindings: set[str] = set()
    for name in ("all", "any"):
        if name in condition:
            for child in condition[name]:
                child_facts, child_bindings = _condition_inputs(child)
                facts.update(child_facts)
                bindings.update(child_bindings)
            return facts, bindings
    if "not" in condition:
        return _condition_inputs(condition["not"])
    facts.add(str(condition["fact"]))
    if "other_fact" in condition:
        facts.add(str(condition["other_fact"]))
    if "binding" in condition:
        bindings.add(str(condition["binding"]))
    return facts, bindings


def _known(value: Any) -> bool:
    return value is not _MISSING and value is not None


def evaluate_condition(condition: dict[str, Any], facts: dict[str, Any], bindings: dict[str, Any]) -> bool:
    if "all" in condition:
        return all(evaluate_condition(item, facts, bindings) for item in condition["all"])
    if "any" in condition:
        return any(evaluate_condition(item, facts, bindings) for item in condition["any"])
    if "not" in condition:
        return not evaluate_condition(condition["not"], facts, bindings)

    lhs = dotted_get(facts, condition["fact"])
    op = condition["op"]
    if op == "exists":
        return lhs is not _MISSING and lhs is not None and lhs != "" and lhs != [] and lhs != {}
    if lhs is _MISSING:
        return False
    if op == "truthy":
        return bool(lhs)
    if op == "falsy":
        return not bool(lhs)
    rhs = _resolve_rhs(condition, facts, bindings)
    if rhs is _MISSING:
        return False
    if op == "changed":
        return lhs != rhs
    if op == "eq":
        return _text(lhs) == _text(rhs)
    if op == "neq":
        return _text(lhs) != _text(rhs)
    if op == "in":
        return _text(lhs) in {_text(item) for item in _sequence(rhs)}
    if op == "contains":
        return bool(_text(rhs)) and _text(rhs) in _text(lhs)
    if op == "contains_any":
        return any(_text(item) and _text(item) in _text(lhs) for item in _sequence(rhs))
    if op == "contains_all":
        values = [_text(item) for item in _sequence(rhs) if _text(item)]
        return bool(values) and all(item in _text(lhs) for item in values)
    if op == "intersects":
        left = {_text(item) for item in _sequence(lhs) if _text(item)}
        right = {_text(item) for item in _sequence(rhs) if _text(item)}
        return bool(left & right)
    if op == "regex":
        flags = re.IGNORECASE if "i" in str(condition.get("flags", "i")) else 0
        try:
            return re.search(str(rhs), str(lhs), flags=flags) is not None
        except re.error:
            return False
    try:
        left_num, right_num = float(lhs), float(rhs)
    except (TypeError, ValueError):
        return False
    return {"gt": left_num > right_num, "ge": left_num >= right_num, "lt": left_num < right_num, "le": left_num <= right_num}.get(op, False)


def evaluate_operator(operator: RubricOperator, facts: dict[str, Any], bindings: dict[str, Any]) -> int:
    # Unknown values must not become true through ``not`` or ``falsy``.
    condition_facts, condition_bindings = _condition_inputs(operator.condition)
    fact_paths = set(operator.required_facts) | condition_facts
    binding_paths = set(operator.task_binding_fields) | condition_bindings
    if any(not _known(dotted_get(facts, path)) for path in fact_paths):
        return 0
    if any(not _known(dotted_get(bindings, path)) for path in binding_paths):
        return 0
    return int(evaluate_condition(operator.condition, facts, bindings))
