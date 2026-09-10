from agent_system.evaluation.hotpotqa_eval import (
    coerce_ground_truths,
    evaluate_hotpotqa_predictions,
    exact_match_score,
    extract_answer_from_text,
    f1_score,
    normalize_answer,
    precision_recall_f1,
)


def test_exact_match_and_normalization():
    assert normalize_answer("The U.S.A.!") == "usa"
    assert exact_match_score("The Chief of Protocol", "chief of protocol") == 1.0
    assert f1_score("Chief of Protocol", "the Chief of Protocol") == 1.0


def test_partial_overlap_f1():
    precision, recall, f1 = precision_recall_f1("chief protocol", "chief of protocol")
    assert precision == 1.0
    assert recall == 2 / 3
    assert round(f1, 6) == 0.8
    assert f1_score("Shirley Temple", "Chief of Protocol") == 0.0


def test_multiple_gold_answers_and_empty_prediction():
    metrics = evaluate_hotpotqa_predictions(["W D Richter", ""], [["W. D. Richter", "Richter"], ["yes"]])
    assert metrics["eval/hotpotqa_num_examples"] == 2.0
    assert metrics["eval/hotpotqa_em"] == 0.5
    assert metrics["eval/hotpotqa_f1"] == 0.5


def test_yes_no_noanswer_special_cases():
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("yes", "no") == 0.0
    assert f1_score("some explanation yes", "yes") == 0.0


def test_extract_answer_span_and_ground_truth_coercion():
    assert extract_answer_from_text("<think>x</think><answer> Chief of Protocol </answer>") == "Chief of Protocol"
    assert extract_answer_from_text("Final Answer: Shirley Temple\nBecause...") == "Shirley Temple"
    assert coerce_ground_truths('{"target": ["yes", "no"]}') == ["yes", "no"]
    assert extract_answer_from_text("<think>Need more info</think><search>Scott Derrickson nationality</search>") == ""
