"""Validated schema for API-generated, deterministic TRCA rubric libraries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


CATEGORIES = ("evidence", "invalidity", "execution")
ALLOWED_FACT_PATHS = {
    "action.text", "action.type", "action.argument", "action.tag_present", "action.parser_valid",
    "action.admissible", "action.available_actions", "transition.accepted", "transition.rejected",
    "transition.state_changed", "transition.repeated", "transition.no_op", "transition.tool_error",
    "observation.current_text", "observation.next_text", "observation.current_tokens",
    "observation.next_tokens", "observation.new_tokens", "observation.new_text", "state.current",
    "state.next", "env.visible_entities", "env.inventory", "env.object_locations", "env.object_states",
    "env.page_type", "env.product_ids", "env.visible_attributes", "env.available_options",
    "env.selected_options", "env.price", "env.cart_state", "env.purchase_status", "env.query",
    "env.result_ids", "env.passage_ids", "env.retrieved_text", "env.tool_status",
    "env.answer_submitted",
}
ALLOWED_BINDING_PATHS = {
    "instruction", "instruction_tokens", "task_relevant_terms", "target_entities",
    "destination_receptacles", "required_operations", "required_quantity", "requested_terms",
    "requested_options", "max_price", "purchase_required", "information_need_terms",
    "query_constraints", "required_answer_format",
}
_ATOMIC_OPS = {
    "exists",
    "truthy",
    "falsy",
    "eq",
    "neq",
    "in",
    "contains",
    "contains_any",
    "contains_all",
    "intersects",
    "regex",
    "gt",
    "ge",
    "lt",
    "le",
    "changed",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_condition(condition: Any, path: str = "condition") -> None:
    """Validate the safe condition DSL; arbitrary code is never accepted."""
    _require(isinstance(condition, dict) and condition, f"{path} must be a non-empty object")
    combinators = [name for name in ("all", "any", "not") if name in condition]
    if combinators:
        _require(len(combinators) == 1 and len(condition) == 1, f"{path} must contain one combinator")
        name = combinators[0]
        value = condition[name]
        if name == "not":
            validate_condition(value, f"{path}.not")
        else:
            _require(isinstance(value, list) and value, f"{path}.{name} must be a non-empty list")
            for index, child in enumerate(value):
                validate_condition(child, f"{path}.{name}[{index}]")
        return

    op = condition.get("op")
    _require(op in _ATOMIC_OPS, f"{path}.op must be one of {sorted(_ATOMIC_OPS)}")
    _require(isinstance(condition.get("fact"), str) and condition["fact"], f"{path}.fact is required")
    allowed = {"op", "fact", "value", "binding", "other_fact", "flags"}
    _require(not (set(condition) - allowed), f"{path} has unsupported keys: {sorted(set(condition) - allowed)}")
    if op == "changed":
        _require(isinstance(condition.get("other_fact"), str), f"{path}.other_fact is required for changed")
    elif op not in {"exists", "truthy", "falsy"}:
        sources = int("value" in condition) + int("binding" in condition) + int("other_fact" in condition)
        _require(sources == 1, f"{path} must specify exactly one of value, binding, or other_fact")


def condition_paths(condition: dict[str, Any]) -> tuple[set[str], set[str]]:
    facts: set[str] = set()
    bindings: set[str] = set()
    for name in ("all", "any"):
        if name in condition:
            for child in condition[name]:
                child_facts, child_bindings = condition_paths(child)
                facts.update(child_facts)
                bindings.update(child_bindings)
            return facts, bindings
    if "not" in condition:
        return condition_paths(condition["not"])
    facts.add(str(condition["fact"]))
    if "other_fact" in condition:
        facts.add(str(condition["other_fact"]))
    if "binding" in condition:
        bindings.add(str(condition["binding"]))
    return facts, bindings


@dataclass(frozen=True)
class RubricOperator:
    identifier: str
    category: str
    description: str
    required_facts: tuple[str, ...]
    task_binding_fields: tuple[str, ...]
    condition: dict[str, Any]
    positive_example: str = ""
    negative_example: str = ""
    missing_evidence: str = "return_0"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RubricOperator":
        identifier = str(raw.get("identifier", "")).strip()
        category = str(raw.get("category", "")).strip().lower()
        _require(bool(identifier), "operator identifier is required")
        _require(category in CATEGORIES, f"operator {identifier}: unsupported category {category!r}")
        required_facts = tuple(str(item) for item in raw.get("required_facts", []))
        binding_fields = tuple(str(item) for item in raw.get("task_binding_fields", []))
        _require(bool(required_facts), f"operator {identifier}: required_facts must not be empty")
        condition = raw.get("condition")
        validate_condition(condition, f"operator[{identifier}].condition")
        condition_facts, condition_bindings = condition_paths(condition)
        _require(
            set(required_facts) <= ALLOWED_FACT_PATHS and condition_facts <= ALLOWED_FACT_PATHS,
            f"operator {identifier}: condition uses unsupported canonical facts",
        )
        _require(
            set(binding_fields) <= ALLOWED_BINDING_PATHS and condition_bindings <= ALLOWED_BINDING_PATHS,
            f"operator {identifier}: condition uses unsupported task bindings",
        )
        _require(
            condition_facts <= set(required_facts),
            f"operator {identifier}: every condition fact must be declared in required_facts",
        )
        _require(
            condition_bindings <= set(binding_fields),
            f"operator {identifier}: every condition binding must be declared in task_binding_fields",
        )
        behavior = str(raw.get("missing_evidence", "return_0")).strip().lower().replace(" ", "_")
        _require(behavior in {"return_0", "0", "false"}, f"operator {identifier}: missing evidence must return 0")
        return cls(
            identifier=identifier,
            category=category,
            description=str(raw.get("description", "")).strip(),
            required_facts=required_facts,
            task_binding_fields=binding_fields,
            condition=condition,
            positive_example=str(raw.get("positive_example", "")).strip(),
            negative_example=str(raw.get("negative_example", "")).strip(),
            missing_evidence="return_0",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "category": self.category,
            "description": self.description,
            "required_facts": list(self.required_facts),
            "task_binding_fields": list(self.task_binding_fields),
            "condition": self.condition,
            "positive_example": self.positive_example,
            "negative_example": self.negative_example,
            "missing_evidence": self.missing_evidence,
        }


@dataclass(frozen=True)
class RubricLibrary:
    benchmark: str
    version: str
    operators: tuple[RubricOperator, ...]
    generated_by: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RubricLibrary":
        benchmark = str(raw.get("benchmark", "")).strip().lower()
        _require(benchmark in {"alfworld", "webshop", "searchqa"}, f"unsupported benchmark {benchmark!r}")
        operators = tuple(RubricOperator.from_dict(item) for item in raw.get("operators", []))
        identifiers = [item.identifier for item in operators]
        _require(len(identifiers) == len(set(identifiers)), "operator identifiers must be unique")
        for category in CATEGORIES:
            count = sum(item.category == category for item in operators)
            _require(count == 5, f"{benchmark}: expected 5 {category} operators, got {count}")
        return cls(
            benchmark=benchmark,
            version=str(raw.get("version", "1.0")),
            operators=operators,
            generated_by=dict(raw.get("generated_by", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RubricLibrary":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "version": self.version,
            "generated_by": self.generated_by,
            "operators": [item.to_dict() for item in self.operators],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
