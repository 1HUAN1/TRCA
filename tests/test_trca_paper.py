from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from reward_methods.schema import StepRecord, TrajectoryRecord
from trca.action_mask import action_token_mask
from trca.advantage import compute_trca_advantage
from trca.api_generate import generate_library
from trca.binding import bind_task
from trca.evaluator import evaluate_operator
from trca.facts import transition_facts
from trca.rewards import compute_trca_rewards, reward_from_judgments
from trca.schema import RubricLibrary, RubricOperator


def _operator(category: str, index: int, condition=None, facts=None, bindings=None):
    condition = condition or {"fact": "action.text", "op": "eq", "value": "__never__"}
    return {
        "identifier": f"{category}_{index}",
        "category": category,
        "description": "test operator",
        "required_facts": facts or [condition.get("fact", "transition.accepted")],
        "task_binding_fields": bindings or [],
        "condition": condition,
        "positive_example": "positive",
        "negative_example": "negative",
        "missing_evidence": "return_0",
    }


def _library_dict(benchmark="alfworld"):
    operators = []
    for category in ("evidence", "invalidity", "execution"):
        for index in range(5):
            operators.append(_operator(category, index))
    return {"benchmark": benchmark, "version": "test", "operators": operators}


def test_library_requires_exactly_five_operators_per_category():
    raw = _library_dict()
    RubricLibrary.from_dict(raw)
    raw["operators"].pop()
    with pytest.raises(ValueError, match="expected 5 execution"):
        RubricLibrary.from_dict(raw)


def test_api_generator_makes_one_call_and_validates_the_library(monkeypatch):
    calls = []
    content = json.dumps(_library_dict("webshop"))
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return body

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return Response()

    monkeypatch.setattr("trca.api_generate.urllib.request.urlopen", fake_urlopen)
    library = generate_library("webshop", "https://api.example/v1", "secret", "rubric-model")
    assert len(calls) == 1
    assert calls[0][0].full_url == "https://api.example/v1/chat/completions"
    assert library.benchmark == "webshop"
    assert len(library.operators) == 15


def test_missing_evidence_is_zero_and_arbitrary_code_is_rejected():
    operator = RubricOperator.from_dict(_operator("evidence", 0))
    assert evaluate_operator(operator, {"action": {"text": "anything"}}, {}) == 0
    bad = _operator("evidence", 1, condition={"fact": "x", "op": "python", "value": "import os"})
    with pytest.raises(ValueError, match="condition.op"):
        RubricOperator.from_dict(bad)


def test_unknown_evidence_cannot_be_activated_by_not_or_falsy():
    negated = RubricOperator.from_dict(
        _operator(
            "invalidity",
            0,
            condition={"not": {"fact": "env.price", "op": "le", "binding": "max_price"}},
            facts=["env.price"],
            bindings=["max_price"],
        )
    )
    falsy = RubricOperator.from_dict(
        _operator("invalidity", 1, condition={"fact": "env.price", "op": "falsy"}, facts=["env.price"])
    )
    assert evaluate_operator(negated, {"env": {"price": None}}, {"max_price": 30}) == 0
    assert evaluate_operator(falsy, {"env": {"price": None}}, {}) == 0


def test_schema_rejects_undeclared_or_noncanonical_inputs():
    undeclared = _operator("evidence", 0, facts=["transition.accepted"])
    with pytest.raises(ValueError, match="declared in required_facts"):
        RubricOperator.from_dict(undeclared)
    private_score = _operator(
        "evidence", 1, condition={"fact": "env.task_score", "op": "truthy"}, facts=["env.task_score"]
    )
    with pytest.raises(ValueError, match="unsupported canonical facts"):
        RubricOperator.from_dict(private_score)


def test_foundational_breakthrough_and_first_coverage_formula():
    library = RubricLibrary.from_dict(_library_dict())
    budgets = {"evidence": 1.0, "invalidity": 1.0, "execution": 1.0}
    hits = {"evidence_0": 1, "invalidity_0": 1, "execution_0": 1}
    foundational, breakthrough, mixed, coverage = reward_from_judgments(hits, library, set(), budgets, 0.8)
    assert foundational == pytest.approx(0.2)
    assert breakthrough == pytest.approx(0.4)
    assert mixed == pytest.approx(0.36)
    assert coverage == {"evidence_0", "execution_0"}

    foundational, breakthrough, mixed, coverage2 = reward_from_judgments(hits, library, coverage, budgets, 0.8)
    assert foundational == pytest.approx(0.2)
    assert breakthrough == 0.0
    assert mixed == pytest.approx(0.04)
    assert coverage2 == coverage


def test_runtime_reward_uses_observable_transition_without_terminal_anchor(tmp_path):
    raw = _library_dict()
    raw["operators"][0] = _operator(
        "evidence",
        0,
        condition={"fact": "observation.new_tokens", "op": "intersects", "binding": "task_relevant_terms"},
        facts=["observation.new_tokens"],
        bindings=["task_relevant_terms"],
    )
    raw["operators"][10] = _operator(
        "execution", 0, condition={"fact": "transition.accepted", "op": "truthy"}, facts=["transition.accepted"]
    )
    library_path = tmp_path / "alfworld.json"
    RubricLibrary.from_dict(raw).save(library_path)
    step = StepRecord(
        task_id="task",
        env_name="alfworld/AlfredTWEnv",
        step_id=0,
        observation="empty room",
        tool_output="an apple is visible",
        action="look apple",
        action_type="look",
        env_reward=1.0,
        raw={"output": "<think>x</think><action>look apple</action>", "rewards": 1.0},
    )
    step.diagnostic.update({"task_instruction": "find apple", "valid_action": True})
    trajectory = TrajectoryRecord(task_id="task", env_name=step.env_name, steps=[step], terminal_reward=1.0, final_success=True)
    config = SimpleNamespace(
        trca_rubric_library=str(library_path),
        trca_rubric_dir=str(tmp_path),
        trca_mix_lambda=0.8,
        trca_evidence_budget=1.0,
        trca_invalidity_budget=1.0,
        trca_execution_budget=1.0,
    )
    compute_trca_rewards(trajectory, config)
    assert step.reward_components["trca_foundational"] == pytest.approx(0.4)
    assert step.reward_components["trca_breakthrough"] == pytest.approx(0.4)
    assert step.mixed_reward == pytest.approx(0.4)
    assert step.reward_components["terminal_anchor"] == 0.0


class _CharTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(item) for item in ids if item)


def test_action_mask_excludes_think_tags_action_tags_and_padding():
    text = "<think>hidden</think>\n<action>click item</action>PAD"
    ids = [ord(char) for char in text]
    mask = action_token_mask(ids, _CharTokenizer())
    credited = "".join(char for char, keep in zip(text, mask) if keep)
    assert credited == "click item"
    assert sum(mask) == len("click item")
    assert action_token_mask([ord(char) for char in "<action>unclosed"], _CharTokenizer()) == [0] * len("<action>unclosed")


@pytest.mark.parametrize(
    ("tag", "body"),
    [("action", "\nclick item\n"), ("search", "capital of France"), ("answer", "Paris")],
)
def test_action_mask_supports_all_environment_action_protocols(tag, body):
    text = f"<think>hidden</think><{tag}>{body}</{tag}>"
    ids = [ord(char) for char in text] + [0, 0]
    mask = action_token_mask(ids, _CharTokenizer())
    credited = "".join(chr(item) for item, keep in zip(ids, mask) if keep)
    assert credited.strip() == body.strip()


def test_action_tags_written_inside_think_are_not_credited():
    text = "<think>try <action>bad</action></think><action>good</action>"
    ids = [ord(char) for char in text]
    mask = action_token_mask(ids, _CharTokenizer())
    assert "".join(chr(item) for item, keep in zip(ids, mask) if keep) == "good"


def test_search_actions_and_webshop_observables_use_public_canonical_fields():
    step = StepRecord(
        task_id="task",
        env_name="Webshop",
        step_id=0,
        observation="Product $19.99",
        tool_output="You have clicked red",
        action="click red",
        raw={"infos": {"task_score": 1.0, "available_actions": {"clickables": ["red", "blue"]}}},
    )
    facts = transition_facts(step)
    assert facts["env"]["price"] == pytest.approx(19.99)
    assert facts["env"]["available_options"] == ["red", "blue"]
    assert facts["env"]["selected_options"] == ["red"]
    assert "task_score" not in facts["env"]
    assert bind_task("webshop", "Find a shirt, price lower than 30.00 dollars")["max_price"] == 30.0

    search_step = StepRecord(
        task_id="question",
        env_name="searchqa",
        step_id=0,
        raw={"output": "<think>x</think><search>capital of France</search>"},
    )
    search_facts = transition_facts(search_step)
    assert search_facts["action"]["text"] == "search capital of France"
    assert search_facts["action"]["parser_valid"] is True
    malformed = StepRecord(task_id="question", env_name="searchqa", step_id=0, raw={"output": "<search>missing close"})
    assert transition_facts(malformed)["action"]["parser_valid"] is False


def test_combined_return_and_two_part_advantage_are_broadcast_to_action_only():
    mask = torch.tensor([[0, 1, 1, 0]] * 4)
    advantages, returns, metrics = compute_trca_advantage(
        response_mask=mask,
        uid=np.asarray(["task"] * 4, dtype=object),
        traj_uid=np.asarray(["a", "a", "b", "b"], dtype=object),
        anchor_obs=np.asarray(["kitchen", "inventory: apple", "kitchen", "inventory: apple"], dtype=object),
        env_rewards=np.asarray([0.0, 1.0, 0.0, 0.0]),
        trca_rewards=np.asarray([1.0, 0.0, 0.0, 0.0]),
        terminal_outcomes=np.asarray([1.0, 1.0, 0.0, 0.0]),
        step_ids=np.asarray([0, 1, 0, 1]),
        gamma=0.95,
        benchmark="alfworld",
        epsilon=0.0,
    )
    assert returns[:, 1].tolist() == pytest.approx([1.95, 1.0, 0.0, 0.0])
    assert advantages[:, 1].tolist() == pytest.approx([2.0, 2.0, -2.0, -2.0])
    assert torch.count_nonzero(advantages[:, [0, 3]]) == 0
    assert metrics["trca/comparable_context_groups"] == 2.0


def test_zero_variance_episode_group_keeps_context_signal():
    mask = torch.ones((2, 1), dtype=torch.long)
    advantages, _, _ = compute_trca_advantage(
        response_mask=mask,
        uid=["task", "task"],
        traj_uid=["a", "b"],
        anchor_obs=["same state", "same state"],
        env_rewards=[0.0, 0.0],
        trca_rewards=[1.0, 0.0],
        terminal_outcomes=[0.0, 0.0],
        step_ids=[0, 0],
        gamma=0.95,
        benchmark="alfworld",
        epsilon=0.0,
    )
    assert advantages[:, 0].tolist() == pytest.approx([1.0, -1.0])


def test_trca_uses_binary_terminal_outcome_and_does_not_merge_distinct_contexts():
    mask = torch.ones((2, 1), dtype=torch.long)
    _, returns, metrics = compute_trca_advantage(
        response_mask=mask,
        uid=["task", "task"],
        traj_uid=["a", "b"],
        anchor_obs=["document 1: Paris is in France", "document 1: Berlin is in Germany"],
        env_rewards=[10.0, 10.0],
        trca_rewards=[0.0, 0.0],
        terminal_outcomes=[1.0, 1.0],
        step_ids=[0, 0],
        benchmark="searchqa",
        epsilon=0.0,
    )
    assert returns[:, 0].tolist() == [1.0, 1.0]
    assert metrics["trca/comparable_context_groups"] == 0.0
    assert metrics["trca/raw_env_reward_abs_max"] == 10.0
