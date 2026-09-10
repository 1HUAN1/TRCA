import json

import pandas as pd
import pytest

from examples.data_preprocess.prepare_searchqa_data import convert_files, validate_frame


def test_convert_searchqa_rows_adds_env_kwargs_and_tagged_prompt(tmp_path):
    source = tmp_path / "source.parquet"
    output = tmp_path / "train.parquet"
    pd.DataFrame(
        [
            {
                "id": "train_1",
                "data_source": "nq",
                "question": "Who wrote Hamlet?",
                "golden_answers": ["William Shakespeare"],
                "reward_model": {"style": "rule", "ground_truth": {"target": ["William Shakespeare"]}},
                "extra_info": {"source_config": "nq"},
            }
        ]
    ).to_parquet(source, index=False)

    summary = convert_files([source], output, split="train")
    converted = pd.read_parquet(output)
    row = converted.iloc[0]
    assert summary["rows"] == 1
    assert row["env_kwargs"]["question"] == "Who wrote Hamlet?"
    assert row["env_kwargs"]["ground_truth"]["target"].tolist() == ["William Shakespeare"]
    assert row["extra_info"]["tools_kwargs"]["search"]["create_kwargs"]["data_source"] == "nq"
    prompt = row["prompt"][0]["content"]
    assert "<search> query </search>" in prompt
    assert "William Shakespeare" not in prompt


def test_convert_searchqa_normalizes_2wiki_source(tmp_path):
    source = tmp_path / "source.jsonl"
    row = {
        "data_source": "mixed_searchqa",
        "question": "Where was Ada born?",
        "answer": ["London"],
        "extra_info": {"source_config": "2wikimultihopqa"},
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "test.parquet"
    convert_files([source], output, split="test")
    converted = pd.read_parquet(output)
    assert converted.iloc[0]["data_source"] == "2wiki"


def test_convert_searchqa_rejects_train_eval_question_overlap(tmp_path):
    train_source = tmp_path / "train_source.parquet"
    eval_source = tmp_path / "eval_source.parquet"
    train_output = tmp_path / "train.parquet"
    eval_output = tmp_path / "eval.parquet"
    row = {
        "data_source": "nq",
        "question": "Who wrote Hamlet?",
        "golden_answers": ["William Shakespeare"],
    }
    pd.DataFrame([row]).to_parquet(train_source, index=False)
    pd.DataFrame([row]).to_parquet(eval_source, index=False)
    convert_files([train_source], train_output, split="train")
    with pytest.raises(ValueError, match="overlaps"):
        convert_files([eval_source], eval_output, split="test", against=train_output)


def test_validate_searchqa_frame_requires_env_kwargs():
    with pytest.raises(ValueError, match="missing columns"):
        validate_frame(pd.DataFrame([{"data_source": "nq"}]), label="bad")


def test_convert_searchqa_rejects_record_id_overlap_even_for_different_questions(tmp_path):
    train_source = tmp_path / "train_source.parquet"
    eval_source = tmp_path / "eval_source.parquet"
    train_output = tmp_path / "train.parquet"
    eval_output = tmp_path / "eval.parquet"
    pd.DataFrame([{"id": "shared-id", "data_source": "nq", "question": "Who wrote Hamlet?", "answer": ["Shakespeare"]}]).to_parquet(train_source)
    pd.DataFrame([{"id": "shared-id", "data_source": "nq", "question": "Who wrote Macbeth?", "answer": ["Shakespeare"]}]).to_parquet(eval_source)
    convert_files([train_source], train_output, split="train")
    with pytest.raises(ValueError, match="record ID overlaps"):
        convert_files([eval_source], eval_output, split="test", against=train_output)


def test_direct_baseline_split_local_indexes_are_not_treated_as_stable_ids():
    common = {
        "data_source": "nq",
        "prompt": [{"role": "user", "content": "Question"}],
        "reward_model": {"ground_truth": {"target": ["answer"]}},
        "env_kwargs": {"question": "Question?", "ground_truth": {"target": ["answer"]}},
    }
    train = pd.DataFrame([{**common, "extra_info": {"index": 0, "split": "train"}}])
    test = pd.DataFrame([{**common, "extra_info": {"index": 0, "split": "test"}}])

    train_summary = validate_frame(train, label="train")
    test_summary = validate_frame(test, label="test")
    assert train_summary["stable_id_coverage"] == 0.0
    assert test_summary["stable_id_coverage"] == 0.0
    assert set(train_summary["record_ids"]).isdisjoint(test_summary["record_ids"])


def test_convert_searchqa_can_explicitly_allow_official_question_overlap(tmp_path):
    train_source = tmp_path / "train_source.parquet"
    eval_source = tmp_path / "eval_source.parquet"
    train_output = tmp_path / "train.parquet"
    eval_output = tmp_path / "eval.parquet"
    row = {"data_source": "nq", "question": "Who wrote Hamlet?", "answer": ["Shakespeare"]}
    pd.DataFrame([row]).to_parquet(train_source, index=False)
    pd.DataFrame([row]).to_parquet(eval_source, index=False)
    convert_files([train_source], train_output, split="train")
    summary = convert_files(
        [eval_source],
        eval_output,
        split="test",
        against=train_output,
        allow_question_overlap=True,
    )
    assert summary["question_overlap_with_against"] == 1
