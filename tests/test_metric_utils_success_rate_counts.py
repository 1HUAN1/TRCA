#!/usr/bin/env python3
"""Focused regression test for validation count compatibility in metric_utils."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


METRIC_UTILS = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "trainer"
    / "ppo"
    / "metric_utils.py"
)


class SuccessRateCountFilterTest(unittest.TestCase):
    def test_episode_comprehension_excludes_count_metrics(self) -> None:
        tree = ast.parse(METRIC_UTILS.read_text(encoding="utf-8"))
        comprehensions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.DictComp)
            and isinstance(node.key, ast.JoinedStr)
            and any(
                isinstance(value, ast.Constant) and value.value == "episode/"
                for value in node.key.values
            )
        ]
        self.assertEqual(len(comprehensions), 1)

        expression = ast.Expression(body=comprehensions[0])
        ast.fix_missing_locations(expression)
        compiled = compile(expression, str(METRIC_UTILS), "eval")

        class Scalar:
            def __init__(self, value: float) -> None:
                self.value = value

            def item(self) -> float:
                return self.value

        class Batch:
            non_tensor_batch = {
                "pick_and_place_success_rate": [Scalar(0.5)],
                "pick_and_place_success_rate_count": 35,
                "unrelated_count": 128,
            }

        result = eval(compiled, {}, {"batch": Batch()})
        self.assertEqual(result, {"episode/pick_and_place_success_rate": 0.5})


if __name__ == "__main__":
    unittest.main()
