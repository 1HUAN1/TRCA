import ast
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_PATH = ROOT / "agent_system" / "multi_turn_rollout" / "rollout_loop.py"
TRAINER_PATH = ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"


def load_weight_helper():
    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validation_success_metric_weight"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TRAINER_PATH), "exec"), namespace)
    return namespace[function.name]


class ValidationSuccessCountTest(unittest.TestCase):
    def setUp(self):
        self.weight = load_weight_helper()

    def test_explicit_count_overrides_data_source_fallback(self):
        self.assertEqual(
            self.weight(
                "pick_success_rate",
                ["alfworld", "alfworld", "alfworld"],
                ["a", "b", "c"],
                [3, 3, 3],
            ),
            3,
        )

    def test_missing_explicit_count_keeps_old_fallback(self):
        self.assertEqual(
            self.weight("success_rate", ["alfworld"] * 3, ["a", "a", "b"]),
            2,
        )
        self.assertEqual(
            self.weight("pick_success_rate", ["alfworld"] * 3, ["a", "b", "c"]),
            0,
        )

    def test_explicit_count_must_be_consistent_and_non_negative(self):
        with self.assertRaises(AssertionError):
            self.weight("pick_success_rate", ["alfworld"] * 2, ["a", "b"], [1, 2])
        with self.assertRaises(AssertionError):
            self.weight("pick_success_rate", ["alfworld"] * 2, ["a", "b"], [-1, -1])

    def test_rollout_broadcasts_count_and_validator_skips_count_metric(self):
        rollout_source = ROLLOUT_PATH.read_text(encoding="utf-8")
        trainer_source = TRAINER_PATH.read_text(encoding="utf-8")
        self.assertIn('success_rate[f"{key}_count"] = len(value)', rollout_source)
        self.assertIn("not k.endswith('_count')", trainer_source)
        self.assertIn("test_batch.non_tensor_batch.get(count_key)", trainer_source)
        self.assertIn("success_rate_counts = {", trainer_source)
        self.assertIn("metric_dict[f'val/{k}_count']", trainer_source)


if __name__ == "__main__":
    unittest.main()
