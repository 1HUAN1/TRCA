"""Foundational and Breakthrough rubric rewards from the TRCA paper."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from trca.binding import bind_task
from trca.evaluator import evaluate_operator
from trca.facts import transition_facts
from trca.schema import CATEGORIES, RubricLibrary


def infer_benchmark(env_name: str) -> str:
    name = env_name.casefold()
    if "alf" in name:
        return "alfworld"
    if "webshop" in name or "shop" in name:
        return "webshop"
    if "search" in name or "qa" in name:
        return "searchqa"
    raise ValueError(f"Cannot infer TRCA benchmark from env_name={env_name!r}")


@lru_cache(maxsize=8)
def _load_library(path: str) -> RubricLibrary:
    return RubricLibrary.load(path)


def resolve_library(config: Any, benchmark: str) -> RubricLibrary:
    explicit = str(getattr(config, "trca_rubric_library", "") or "").strip()
    directory = str(getattr(config, "trca_rubric_dir", "trca_rubrics") or "trca_rubrics").strip()
    path = Path(explicit) if explicit else Path(directory) / f"{benchmark}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"TRCA rubric library not found: {path}. Generate it once with "
            f"`python -m trca.api_generate --benchmark {benchmark} --output {path}` before training."
        )
    library = _load_library(str(path.resolve()))
    if library.benchmark != benchmark:
        raise ValueError(f"Rubric benchmark {library.benchmark!r} does not match environment {benchmark!r}")
    return library


def _budgets(config: Any) -> dict[str, float]:
    return {
        "evidence": max(0.0, float(getattr(config, "trca_evidence_budget", 1.0))),
        "invalidity": abs(float(getattr(config, "trca_invalidity_budget", 1.0))),
        "execution": max(0.0, float(getattr(config, "trca_execution_budget", 1.0))),
    }


def reward_from_judgments(
    judgments: dict[str, int],
    library: RubricLibrary,
    coverage: set[str],
    budgets: dict[str, float],
    mix_lambda: float,
) -> tuple[float, float, float, set[str]]:
    """Compute Eq. 4, Eq. 6, and Eq. 7, updating positive coverage causally."""
    counts = {category: sum(item.category == category for item in library.operators) for category in CATEGORIES}
    foundational = 0.0
    breakthrough = 0.0
    new_coverage = set(coverage)
    for operator in library.operators:
        active = int(bool(judgments.get(operator.identifier, 0)))
        if not active:
            continue
        amount = budgets[operator.category] / counts[operator.category]
        foundational += -amount if operator.category == "invalidity" else amount
        if operator.category in {"evidence", "execution"} and operator.identifier not in coverage:
            breakthrough += amount
            new_coverage.add(operator.identifier)
    mixed = (1.0 - mix_lambda) * foundational + mix_lambda * breakthrough
    return foundational, breakthrough, mixed, new_coverage


def compute_trca_rewards(trajectory: Any, config: Any) -> Any:
    """Evaluate one trajectory with no terminal labels and no online LLM calls."""
    benchmark = infer_benchmark(str(getattr(trajectory, "env_name", "")))
    library = resolve_library(config, benchmark)
    instruction = ""
    metadata: dict[str, Any] = {}
    if trajectory.steps:
        first = trajectory.steps[0]
        instruction = str(first.diagnostic.get("task_instruction", "") or "")
        raw = first.raw if isinstance(first.raw, dict) else {}
        metadata = raw.get("task_metadata", raw.get("env_kwargs", {}))
    bindings = bind_task(benchmark, instruction, metadata)
    budgets = _budgets(config)
    mix_lambda = float(getattr(config, "trca_mix_lambda", 0.8))
    if not 0.0 <= mix_lambda <= 1.0:
        raise ValueError(f"trca_mix_lambda must be in [0, 1], got {mix_lambda}")
    coverage: set[str] = set()
    for index, step in enumerate(trajectory.steps):
        facts = transition_facts(step, trajectory.steps[index - 1] if index else None)
        judgments = {operator.identifier: evaluate_operator(operator, facts, bindings) for operator in library.operators}
        foundational, breakthrough, mixed, coverage = reward_from_judgments(
            judgments, library, coverage, budgets, mix_lambda
        )
        step.dense_reward = foundational
        step.mixed_reward = mixed
        step.reward_components.update(
            {
                "trca_foundational": foundational,
                "trca_breakthrough": breakthrough,
                "trca_reward": mixed,
                "trca_mix_lambda": mix_lambda,
                "trca_evidence_hits": float(sum(judgments[item.identifier] for item in library.operators if item.category == "evidence")),
                "trca_invalidity_hits": float(sum(judgments[item.identifier] for item in library.operators if item.category == "invalidity")),
                "trca_execution_hits": float(sum(judgments[item.identifier] for item in library.operators if item.category == "execution")),
                "terminal_anchor": 0.0,
            }
        )
        if isinstance(step.raw, dict):
            step.raw["trca_judgments"] = judgments
            step.raw["trca_task_bindings"] = bindings
    trajectory.metadata["trca_benchmark"] = benchmark
    trajectory.metadata["trca_rubric_version"] = library.version
    return trajectory
