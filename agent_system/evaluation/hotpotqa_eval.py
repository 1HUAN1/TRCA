"""HotpotQA answer-level evaluation utilities.

This module intentionally evaluates final answer text, not environment reward,
Search-R1 EM reward, or supporting-fact F1.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable
import ast
import json
import re
import string

_SPECIAL_ANSWERS = {"yes", "no", "noanswer"}


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    text = "" if s is None else str(s)

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def _tokens(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def precision_recall_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Return HotpotQA-style answer precision, recall, and F1 for one gold."""
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(ground_truth)

    if normalized_prediction in _SPECIAL_ANSWERS or normalized_gold in _SPECIAL_ANSWERS:
        score = float(normalized_prediction == normalized_gold)
        return score, score, score

    prediction_tokens = _tokens(prediction)
    gold_tokens = _tokens(ground_truth)
    if not prediction_tokens or not gold_tokens:
        score = float(prediction_tokens == gold_tokens)
        return score, score, score

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def f1_score(prediction: str, ground_truth: str) -> float:
    return precision_recall_f1(prediction, ground_truth)[2]


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn: Callable[[str, str], float], prediction: str, ground_truths: Iterable[str]) -> float:
    scores = [metric_fn(prediction, ground_truth) for ground_truth in ground_truths]
    return max(scores) if scores else 0.0


def _as_plain(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def coerce_ground_truths(value: Any) -> list[str]:
    """Convert common HotpotQA/Search-R1 ground-truth encodings to strings."""
    value = _as_plain(value)
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("target", "golden_answers", "answers", "answer"):
            if key in value:
                return coerce_ground_truths(value[key])
        if "ground_truth" in value:
            return coerce_ground_truths(value["ground_truth"])
        return []
    if isinstance(value, (list, tuple, set)):
        answers: list[str] = []
        for item in value:
            answers.extend(coerce_ground_truths(item))
        return [answer for answer in answers if answer != ""]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            parsed_answers = coerce_ground_truths(parsed)
            if parsed_answers:
                return parsed_answers
        return [text]
    return [str(value)]


def extract_answer_from_text(text: Any) -> str:
    """Extract the answer span used for answer-level F1."""
    if text is None:
        return ""
    value = str(text)
    tag_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", value, flags=re.IGNORECASE | re.DOTALL)
    if tag_matches:
        return " ".join(tag_matches[-1].split())
    label_matches = re.findall(
        r"(?:final\s+answer|answer)\s*:\s*(.*?)(?:\n|$)",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if label_matches:
        return " ".join(label_matches[-1].strip().split())
    if re.search(r"</?(search|think)\b", value, flags=re.IGNORECASE):
        return ""
    lines = [line.strip() for line in value.strip().splitlines() if line.strip()]
    return " ".join((lines[-1] if lines else value.strip()).split())


def best_answer_scores(prediction: str, ground_truths: Iterable[str]) -> dict[str, float]:
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "em": 0.0}
    for ground_truth in ground_truths:
        precision, recall, f1 = precision_recall_f1(prediction, ground_truth)
        em = exact_match_score(prediction, ground_truth)
        if (f1, em, precision, recall) > (best["f1"], best["em"], best["precision"], best["recall"]):
            best = {"precision": precision, "recall": recall, "f1": f1, "em": em}
    return best


def evaluate_hotpotqa_predictions(predictions: Iterable[Any], references: Iterable[Any]) -> dict[str, float]:
    """Evaluate predictions against one or more acceptable gold answers."""
    totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "em": 0.0}
    count = 0
    for prediction, reference in zip(predictions, references):
        ground_truths = coerce_ground_truths(reference)
        if not ground_truths:
            continue
        scores = best_answer_scores(str(prediction), ground_truths)
        for key in totals:
            totals[key] += scores[key]
        count += 1
    denom = float(count) if count else 1.0
    return {
        "eval/hotpotqa_f1": totals["f1"] / denom,
        "eval/hotpotqa_em": totals["em"] / denom,
        "eval/hotpotqa_precision": totals["precision"] / denom,
        "eval/hotpotqa_recall": totals["recall"] / denom,
        "eval/hotpotqa_num_examples": float(count),
    }
