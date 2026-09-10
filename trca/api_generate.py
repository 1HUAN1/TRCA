"""One-time API generation of a shared benchmark-level TRCA operator library.

The API is used before training only. Runtime transition evaluation is fully
deterministic and never calls this module.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any

from trca.schema import RubricLibrary


PAPER_PROMPT = """Role. You are an offline rubric-library designer for long-horizon agent training. Construct one shared benchmark-level rubric operator library. The resulting library will be reused across all task instances from the same benchmark.

Core rule. Define reusable rubric operators rather than task-specific or step-specific judgments. Each operator must be implementable as a deterministic Boolean function over canonical environment facts. No LLM judge may be required during rollout collection or policy optimization.

Task instantiation. A deterministic task binder extracts task-specific entities, attributes, constraints, and target conditions from the task instruction and environment metadata. These parameters are bound to the shared benchmark-level operators. Do not generate a new library for each task.

Transition evaluation. Each bound operator evaluates exactly one action-induced transition and returns 0 or 1. Missing, ambiguous, or unsupported observable evidence returns 0. Successful trajectories, terminal outcomes, future transitions, hidden states, benchmark scores, ground-truth answers, and answer aliases are prohibited.

Create exactly five operators in each category:
- evidence: newly revealed task-relevant entities, attributes, locations, relations, products, passages, or supporting facts;
- invalidity: malformed, unavailable, rejected, repeated, no-op, or precondition-violating actions;
- execution: accepted task-required operations, achieved subgoals, transformations, selections, retrieval operations, or intermediate objectives.

Canonical fact paths (use only these paths):
action.text, action.type, action.argument, action.tag_present, action.parser_valid, action.admissible, action.available_actions;
transition.accepted, transition.rejected, transition.state_changed, transition.repeated, transition.no_op, transition.tool_error;
observation.current_text, observation.next_text, observation.current_tokens, observation.next_tokens, observation.new_tokens, observation.new_text;
state.current, state.next;
env.visible_entities, env.inventory, env.object_locations, env.object_states;
env.page_type, env.product_ids, env.visible_attributes, env.available_options, env.selected_options, env.price, env.cart_state, env.purchase_status;
env.query, env.result_ids, env.passage_ids, env.retrieved_text, env.tool_status, env.answer_submitted.

Task binding paths:
instruction, instruction_tokens, task_relevant_terms;
ALFWorld: target_entities, destination_receptacles, required_operations, required_quantity;
WebShop: requested_terms, requested_options, max_price, purchase_required;
SearchQA: target_entities, information_need_terms, query_constraints, required_answer_format.

Safe Boolean condition DSL (JSON only):
- {"all": [condition, ...]}, {"any": [...]}, {"not": condition}
- atomic: {"fact": "path", "op": OP, "value": literal}
- replace value with "binding": "path" or "other_fact": "path".
OP is one of exists, truthy, falsy, eq, neq, in, contains, contains_any, contains_all, intersects, regex, gt, ge, lt, le, changed.
For changed use fact plus other_fact. Never emit source code, expressions, templates, or an operator outside this DSL.

Return one JSON object with keys benchmark, version, operators. Each operator must include identifier, category, description, required_facts, task_binding_fields, condition, positive_example, negative_example, and missing_evidence="return_0". Return exactly 15 operators and no markdown."""


BENCHMARK_INTERFACES = {
    "alfworld": "ALFWorld text actions expose admissible commands, object/receptacle names, inventory, locations, acceptance feedback, and clean/cool/heat state transformations.",
    "webshop": "WebShop exposes page type, product/result identifiers, visible attributes, available and selected options, price fields, navigation feedback, cart state, and purchase readiness.",
    "searchqa": "SearchQA exposes queries, result/passage identifiers, retrieved text, tool status, retrieval-stage changes, source novelty, and answer-submission format; ground-truth aliases are unavailable to operators.",
}


def _endpoint(value: str) -> str:
    endpoint = value.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint += "/chat/completions"
    return endpoint


def _extract_content(response: dict[str, Any]) -> str:
    if "choices" in response:
        content = response["choices"][0]["message"]["content"]
    elif "output_text" in response:
        content = response["output_text"]
    else:
        raise ValueError("API response has neither choices[].message.content nor output_text")
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if isinstance(value.get("library"), dict):
        value = value["library"]
    return value


def generate_library(
    benchmark: str,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: float = 180.0,
) -> RubricLibrary:
    benchmark = benchmark.casefold()
    if benchmark not in BENCHMARK_INTERFACES:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": PAPER_PROMPT},
            {"role": "user", "content": f"Benchmark: {benchmark}\nInterface: {BENCHMARK_INTERFACES[benchmark]}"},
        ],
    }
    request = urllib.request.Request(
        _endpoint(endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Rubric API returned HTTP {error.code}: {detail}") from error
    raw = _parse_json(_extract_content(body))
    raw["benchmark"] = benchmark
    raw.setdefault("version", "1.0")
    raw["generated_by"] = {
        "model": model,
        "endpoint": _endpoint(endpoint),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "temperature": 0,
        "calls_per_benchmark": 1,
    }
    return RubricLibrary.from_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARK_INTERFACES))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default=os.getenv("TRCA_RUBRIC_API_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("TRCA_RUBRIC_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("TRCA_RUBRIC_MODEL", ""))
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    missing = [name for name, value in (("endpoint/TRCA_RUBRIC_API_URL", args.endpoint), ("api-key/TRCA_RUBRIC_API_KEY", args.api_key), ("model/TRCA_RUBRIC_MODEL", args.model)) if not value]
    if missing:
        parser.error("missing " + ", ".join(missing))
    library = generate_library(args.benchmark, args.endpoint, args.api_key, args.model, args.timeout)
    library.save(args.output)
    print(f"Wrote validated {library.benchmark} rubric library with {len(library.operators)} operators to {args.output}")


if __name__ == "__main__":
    main()
