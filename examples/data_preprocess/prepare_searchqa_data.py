#!/usr/bin/env python3
"""Convert local SearchQA records to the schema used by the agent rollout."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from agent_system.evaluation.hotpotqa_eval import coerce_ground_truths


SOURCE_ALIASES = {
    "2wikimultihopqa": "2wiki",
    "2wiki_multihop_qa": "2wiki",
    "trivia_qa": "triviaqa",
}
BASELINE_SOURCE_COUNTS = {
    "train": {"nq": 79168, "hotpotqa": 90447},
    "test": {
        "nq": 3610,
        "triviaqa": 11313,
        "popqa": 14267,
        "hotpotqa": 7405,
        "2wikimultihopqa": 12576,
        "musique": 2417,
        "bamboogle": 125,
    },
}

PROMPT_TEMPLATE = """Answer the given question using the offline search tool when needed.
After each new observation, reason briefly inside <think> and </think>.
Use exactly <search> query </search> to retrieve evidence.
When enough evidence is available, return only the short answer inside <answer> and </answer>.
Do not issue a search and an answer in the same turn.
Question: {question}
"""


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _plain(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    value = _plain(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return [_plain(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [_plain(json.loads(line)) for line in handle if line.strip()]
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("examples", []))
        return [_plain(row) for row in payload]
    raise ValueError(f"Unsupported SearchQA input: {path}")


def _question(row: dict[str, Any]) -> str:
    extra = _as_dict(row.get("extra_info"))
    question = str(row.get("question") or extra.get("question") or "").strip()
    if not question:
        raise ValueError("SearchQA record is missing question")
    if question[-1] != "?":
        question += "?"
    return question


def _answers(row: dict[str, Any]) -> list[str]:
    extra = _as_dict(row.get("extra_info"))
    reward_model = _as_dict(row.get("reward_model"))
    for value in (
        reward_model.get("ground_truth"),
        row.get("golden_answers"),
        row.get("answers"),
        row.get("answer"),
        extra.get("golden_answers"),
    ):
        answers = coerce_ground_truths(_plain(value))
        if answers:
            return answers
    raise ValueError(f"SearchQA record is missing answers: {_question(row)[:100]}")


def normalize_source(row: dict[str, Any], override: str | None = None) -> str:
    extra = _as_dict(row.get("extra_info"))
    source = str(override or extra.get("source_config") or row.get("data_source") or "").strip().lower()
    source = source.removeprefix("mixed_searchqa_")
    if source == "mixed_searchqa":
        source = str(extra.get("source_config") or "").strip().lower()
    source = SOURCE_ALIASES.get(source, source)
    if not source:
        raise ValueError(f"Cannot infer SearchQA source for question: {_question(row)[:100]}")
    return source


def convert_row(
    row: dict[str, Any],
    *,
    split: str,
    index: int,
    source_override: str | None = None,
) -> dict[str, Any]:
    question = _question(row)
    answers = _answers(row)
    source = normalize_source(row, source_override)
    original_extra = _as_dict(row.get("extra_info"))
    source_id = _first_identifier(row.get("id"), original_extra.get("id"))
    record_id = source_id or f"{source}_{split}_{index}"
    ground_truth = {"target": answers}
    env_kwargs = {"ground_truth": ground_truth, "question": question, "data_source": source}
    extra_info = {
        "index": record_id,
        "id": record_id,
        "source_id": source_id or record_id,
        "question": question,
        "split": split,
        "source_config": source,
        "need_tools_kwargs": True,
        "tools_kwargs": {"search": {"create_kwargs": env_kwargs}},
        "golden_answers": answers,
        "tool_protocol": "tagged_action",
    }
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        "ability": row.get("ability") or "fact_reasoning",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
        "metadata": json.dumps(_plain(row.get("metadata")), ensure_ascii=False)
        if _plain(row.get("metadata")) is not None
        else None,
        "env_kwargs": env_kwargs,
    }


def normalized_question(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _first_identifier(*values: Any) -> str:
    for value in values:
        value = _plain(value)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def validate_frame(frame: pd.DataFrame, *, label: str) -> dict[str, Any]:
    required = {"data_source", "prompt", "reward_model", "extra_info", "env_kwargs"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    if frame.empty:
        raise ValueError(f"{label}: dataset is empty")
    questions: list[str] = []
    record_ids: list[str] = []
    stable_record_ids: list[str] = []
    for idx, row in frame.iterrows():
        env_kwargs = _as_dict(row["env_kwargs"])
        extra_info = _as_dict(row["extra_info"])
        metadata = _as_dict(row.get("metadata"))
        question = str(env_kwargs.get("question") or "").strip()
        answers = coerce_ground_truths(_as_dict(env_kwargs.get("ground_truth")))
        if not question or not answers:
            raise ValueError(f"{label}: row {idx} has invalid env_kwargs")
        questions.append(normalized_question(question))
        source = str(row["data_source"]).strip().lower()
        explicit_id = _first_identifier(row.get("id"), extra_info.get("id"), extra_info.get("source_id"), metadata.get("id"))
        if explicit_id:
            record_id = f"{source}:{explicit_id}"
            stable_record_ids.append(record_id)
        else:
            split = str(extra_info.get("split") or "unknown").strip().lower()
            local_index = extra_info.get("index", idx)
            record_id = f"{source}:{split}:{local_index}"
        record_ids.append(record_id)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{label}: duplicate record IDs are not allowed")
    return {
        "rows": len(frame),
        "sources": dict(Counter(str(value) for value in frame["data_source"])),
        "questions": questions,
        "record_ids": record_ids,
        "stable_record_ids": stable_record_ids,
        "stable_id_coverage": len(stable_record_ids) / len(frame),
    }


def _compare_summaries(
    summary: dict[str, Any],
    other_summary: dict[str, Any],
    *,
    against: Path,
    allow_question_overlap: bool,
) -> None:
    id_overlap = set(summary["stable_record_ids"]) & set(other_summary["stable_record_ids"])
    if id_overlap:
        raise ValueError(f"Found {len(id_overlap)} stable record ID overlaps with {against}")
    overlap = set(summary["questions"]) & set(other_summary["questions"])
    summary["question_overlap_with_against"] = len(overlap)
    if overlap and not allow_question_overlap:
        raise ValueError(f"Found {len(overlap)} normalized question overlaps with {against}")
    if overlap:
        print(
            f"[warning] allowing {len(overlap)} normalized question overlaps to preserve the official baseline split",
            file=sys.stderr,
        )


def _public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"questions", "record_ids", "stable_record_ids"}
    }


def validate_baseline_counts(summary: dict[str, Any], split: str) -> None:
    expected = BASELINE_SOURCE_COUNTS[split]
    actual = {str(key): int(value) for key, value in summary["sources"].items()}
    if actual != expected:
        raise ValueError(f"{split}: expected source counts {expected}, got {actual}")
    expected_rows = sum(expected.values())
    if int(summary["rows"]) != expected_rows:
        raise ValueError(f"{split}: expected {expected_rows} rows, got {summary['rows']}")


def validate_eval_source_count(summary: dict[str, Any], source: str) -> None:
    normalized_expected = {
        SOURCE_ALIASES.get(key, key): value for key, value in BASELINE_SOURCE_COUNTS["test"].items()
    }
    expected_rows = normalized_expected[source]
    actual = {SOURCE_ALIASES.get(str(key), str(key)): int(value) for key, value in summary["sources"].items()}
    if actual != {source: expected_rows} or int(summary["rows"]) != expected_rows:
        raise ValueError(f"{source}: expected exactly {expected_rows} rows, got rows={summary['rows']} sources={actual}")


def split_existing_eval_parquet(input_path: Path, output_root: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(input_path)
    validate_frame(frame, label=str(input_path))
    table = pq.read_table(input_path)
    expected = BASELINE_SOURCE_COUNTS["test"]
    outputs = []
    output_root.mkdir(parents=True, exist_ok=True)
    for raw_source, expected_rows in expected.items():
        source = SOURCE_ALIASES.get(raw_source, raw_source)
        subset = table.filter(pc.equal(table["data_source"], raw_source))
        if subset.num_rows != expected_rows:
            raise ValueError(f"{source}: expected {expected_rows} rows, got {subset.num_rows}")
        output = output_root / f"{source}.parquet"
        temporary = output.with_name(output.name + ".tmp")
        pq.write_table(subset, temporary)
        temporary.replace(output)
        outputs.append({"source": source, "rows": subset.num_rows, "output": str(output)})
    return outputs


def convert_files(
    inputs: Iterable[Path],
    output: Path,
    *,
    split: str,
    source_override: str | None = None,
    max_per_source: int = 0,
    against: Path | None = None,
    allow_question_overlap: bool = False,
) -> dict[str, Any]:
    converted: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for input_path in inputs:
        for row in _read_rows(input_path):
            record = convert_row(row, split=split, index=len(converted), source_override=source_override)
            source = record["data_source"]
            if max_per_source > 0 and source_counts[source] >= max_per_source:
                continue
            source_counts[source] += 1
            converted.append(record)

    frame = pd.DataFrame(converted)
    summary = validate_frame(frame, label=str(output))
    if against is not None:
        other = pd.read_parquet(against)
        other_summary = validate_frame(other, label=str(against))
        _compare_summaries(
            summary,
            other_summary,
            against=against,
            allow_question_overlap=allow_question_overlap,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    public_summary = _public_summary(summary)
    public_summary["output"] = str(output)
    return public_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--source")
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--against", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-question-overlap", action="store_true")
    parser.add_argument("--expect-baseline-split", choices=sorted(BASELINE_SOURCE_COUNTS))
    parser.add_argument(
        "--expect-eval-source",
        choices=("nq", "triviaqa", "popqa", "hotpotqa", "2wiki", "musique", "bamboogle"),
    )
    parser.add_argument("--split-output-root", type=Path)
    args = parser.parse_args()

    if args.split_output_root is not None:
        if len(args.input) != 1:
            parser.error("--split-output-root expects exactly one --input")
        summary = {"splits": split_existing_eval_parquet(args.input[0], args.split_output_root)}
    elif args.validate_only:
        if len(args.input) != 1:
            parser.error("--validate-only expects exactly one --input")
        frame = pd.read_parquet(args.input[0])
        summary = validate_frame(frame, label=str(args.input[0]))
        if args.against is not None:
            other = pd.read_parquet(args.against)
            other_summary = validate_frame(other, label=str(args.against))
            _compare_summaries(
                summary,
                other_summary,
                against=args.against,
                allow_question_overlap=args.allow_question_overlap,
            )
        if args.expect_baseline_split:
            validate_baseline_counts(summary, args.expect_baseline_split)
        if args.expect_eval_source:
            validate_eval_source_count(summary, args.expect_eval_source)
        summary = _public_summary(summary)
        summary["input"] = str(args.input[0])
    else:
        if args.output is None:
            parser.error("--output is required unless --validate-only is used")
        summary = convert_files(
            args.input,
            args.output,
            split=args.split,
            source_override=args.source,
            max_per_source=args.max_per_source,
            against=args.against,
            allow_question_overlap=args.allow_question_overlap,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
