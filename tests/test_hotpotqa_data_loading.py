import os

import pytest

from examples.data_preprocess.prepare_hotpotqa_search import build_search_row, load_hotpotqa_rows

CANDIDATES = [
    "/root/data/mingjuchen/EMNLP2026/ADAS/eval_bench/qa/source_data/hotpotqa/dev.jsonl",
    "/root/data/mingjuchen/EMNLP2026/ADAS/eval_bench/qa/data/hotpotqa.parquet",
]


def test_build_search_row_has_trainer_fields():
    row = {
        "id": "dev_synthetic",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "golden_answers": ["yes"],
        "metadata": {"type": "comparison", "level": "hard", "supporting_facts": {"title": ["A", "B"]}},
    }
    converted = build_search_row(row, split="test", index=0)
    assert converted["data_source"] == "hotpotqa"
    assert converted["env_kwargs"]["ground_truth"] == {"target": ["yes"]}
    assert converted["env_kwargs"]["question"] == row["question"]
    assert converted["reward_model"]["style"] == "rule"
    assert converted["extra_info"]["question_type"] == "comparison"


def test_build_search_row_accepts_rewardflow_extra_info_question():
    row = {
        "data_source": "mixed_searchqa_hotpotqa",
        "reward_model": {"style": "rule", "ground_truth": '["Virgil Hill"]'},
        "extra_info": {
            "benchmark": "hotpotqa",
            "id": "train_5053",
            "question": "Guy Walters competed against which former professional boxer?",
            "golden_answers": '["Virgil Hill"]',
        },
    }
    converted = build_search_row(row, split="test", index=1)
    assert converted["env_kwargs"]["question"].startswith("Guy Walters")
    assert converted["env_kwargs"]["ground_truth"] == {"target": ["Virgil Hill"]}


def test_existing_hotpotqa_eval_data_can_be_loaded_if_present():
    path = next((candidate for candidate in CANDIDATES if os.path.exists(candidate)), None)
    if path is None:
        pytest.skip("No local HotpotQA data candidate is present on this machine")
    rows = load_hotpotqa_rows(path)
    assert rows
    sample = rows[0]
    assert "question" in sample
    assert any(key in sample for key in ("golden_answers", "answer", "reward_model", "extra_info"))
