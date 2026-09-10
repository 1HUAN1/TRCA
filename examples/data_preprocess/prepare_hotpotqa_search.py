"""Prepare HotpotQA rows for the existing Search-R1 environment format."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import pandas as pd

from agent_system.evaluation.hotpotqa_eval import coerce_ground_truths

DEFAULT_INPUT = "/root/data/mingjuchen/EMNLP2026/ADAS/eval_bench/qa/source_data/hotpotqa/dev.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/hotpotqa_smoke_data"
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."


def _json_loads_maybe(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in '[{"':
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def load_hotpotqa_rows(input_path: str) -> list[dict[str, Any]]:
    path = os.path.expanduser(input_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".jsonl"):
        rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("data", data.get("examples", []))
        return list(data)
    if path.endswith(".parquet"):
        return [_row_to_dict(row) for _, row in pd.read_parquet(path).iterrows()]
    raise ValueError(f"Unsupported HotpotQA input format: {path}")


def _extract_answers(row: dict[str, Any]) -> list[str]:
    for key in ("golden_answers", "answer", "answers"):
        answers = coerce_ground_truths(_json_loads_maybe(row.get(key)))
        if answers:
            return answers
    reward_model = _json_loads_maybe(row.get("reward_model"))
    if isinstance(reward_model, dict):
        answers = coerce_ground_truths(reward_model.get("ground_truth"))
        if answers:
            return answers
    extra_info = _json_loads_maybe(row.get("extra_info"))
    if isinstance(extra_info, dict):
        answers = coerce_ground_truths(_json_loads_maybe(extra_info.get("golden_answers")))
        if answers:
            return answers
    return []


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = _json_loads_maybe(row.get("metadata"))
    if isinstance(value, dict):
        return value
    extra = _json_loads_maybe(row.get("extra_info"))
    if isinstance(extra, dict):
        source = _json_loads_maybe(extra.get("source_metadata"))
        if isinstance(source, dict):
            return source
    return {}


def build_search_row(row: dict[str, Any], split: str, index: int, data_source: str = "hotpotqa") -> dict[str, Any]:
    extra_info = _json_loads_maybe(row.get("extra_info"))
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    question = str(row.get("question") or extra_info.get("question") or "").strip()
    answers = _extract_answers(row)
    if not question:
        raise ValueError("HotpotQA row is missing question")
    if not answers:
        raise ValueError(f"HotpotQA row is missing answers for question: {question[:80]}")

    metadata = _metadata(row)
    if not metadata:
        metadata = {"source_config": extra_info.get("source_config", "hotpotqa")}
    ground_truth = {"target": answers}
    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": question},
    ]
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source,
            }
        }
    }
    extra_info = {
        "index": index,
        "need_tools_kwargs": True,
        "question": question,
        "split": split,
        "hotpotqa_id": row.get("id", row.get("_id", extra_info.get("id", ""))),
        "golden_answers": answers,
        "question_type": metadata.get("type", "unknown"),
        "level": metadata.get("level", "unknown"),
        "supporting_facts": metadata.get("supporting_facts"),
        "tools_kwargs": tools_kwargs,
    }
    env_kwargs = {"ground_truth": ground_truth, "question": question, "data_source": data_source}
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "multi_hop_qa",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
        "metadata": metadata,
        "env_kwargs": env_kwargs,
    }


def write_hotpotqa_search_parquets(
    input_path: str,
    output_dir: str,
    train_size: int,
    val_size: int,
    seed: int = 0,
    data_source: str = "hotpotqa",
) -> tuple[str, str]:
    rows = load_hotpotqa_rows(input_path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    needed = train_size + val_size
    if len(rows) < needed:
        raise ValueError(f"Need {needed} rows but only found {len(rows)} in {input_path}")
    os.makedirs(output_dir, exist_ok=True)
    train_rows = [build_search_row(row, "train", idx, data_source=data_source) for idx, row in enumerate(rows[:train_size])]
    val_rows = [build_search_row(row, "test", idx, data_source=data_source) for idx, row in enumerate(rows[train_size:needed])]
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "test.parquet")
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HotpotQA Search-R1 style parquet files.")
    parser.add_argument("--input_path", default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_size", type=int, default=32)
    parser.add_argument("--val_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_source", default="hotpotqa")
    args = parser.parse_args()
    train_path, val_path = write_hotpotqa_search_parquets(
        input_path=args.input_path,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        data_source=args.data_source,
    )
    print(f"wrote train={train_path}")
    print(f"wrote test={val_path}")


if __name__ == "__main__":
    main()
