import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import (
    compute_score,
)
from verl.trainer.ppo.validation_trajectory_dump import (
    build_validation_provenance,
    build_validation_trajectory_records,
    dump_validation_trajectory_records,
    snapshot_validation_gold_contexts,
    validate_validation_trajectory_metrics,
)


class ValidationTrajectoryDumpTests(unittest.TestCase):
    def setUp(self):
        self.provenance = build_validation_provenance(
            prompt_style="multihop_bridge_v1",
            global_step=np.int64(450),
            resume_path="/checkpoints/run/global_step_450",
            topk=3,
            max_steps=4,
            val_n=1,
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            top_k=-1,
        )
        self.gold_contexts = [
            {
                "question": "Question A",
                "ground_truth": {"target": ["A"]},
                "data_source": "bamboogle",
                "initial_input": "initial A",
            },
            {
                "question": "Question B",
                "ground_truth": {"target": ["B"]},
                "data_source": "bamboogle",
                "initial_input": "initial B",
            },
        ]

    def rows(self):
        # EpisodeRewardManager repeats episode_reward on every active row, hence [1, 1, 0].
        return {
            "traj_uid": np.array([b"traj-a", b"traj-a", "traj-b"], dtype=object),
            "data_source": np.array(["bamboogle"] * 3, dtype=object),
            "tool_output": np.array(
                [np.array(["doc-a"]), b"", torch.tensor([1, 2])], dtype=object
            ),
            "rewards": np.array([0.0, 1.0, 0.0], dtype=object),
            "is_action_valid": np.array([True, np.bool_(True), 1], dtype=object),
            "active_masks": np.array([True, True, True], dtype=object),
            "episode_rewards": np.array([1.0, 1.0, 0.0], dtype=object),
            "episode_lengths": np.array([2.0, 2.0, 1.0], dtype=object),
            "tool_callings": np.array([1.0, 1.0, 0.0], dtype=object),
        }

    def build(self, rows=None, scores=None, contexts=None):
        return build_validation_trajectory_records(
            prompts=[b"prompt-a1", "prompt-a2", "prompt-b1"],
            raw_actions=["<search>a</search>", "<answer>A</answer>", "<answer>B</answer>"],
            validation_scores=[1.0, 1.0, 0.0] if scores is None else scores,
            non_tensor_batch=self.rows() if rows is None else rows,
            gold_contexts=self.gold_contexts if contexts is None else contexts,
            provenance=self.provenance,
        )

    def test_groups_strict_active_steps_with_gold_and_provenance(self):
        records = self.build()
        self.assertEqual([row["trajectory_id"] for row in records], ["traj-a", "traj-b"])
        first, second = records
        self.assertEqual(first["question"], "Question A")
        self.assertEqual(first["ground_truth"], {"target": ["A"]})
        self.assertEqual(first["initial_input"], "initial A")
        self.assertTrue(first["strict_success"])
        self.assertEqual(first["terminal_score"], 1.0)
        self.assertEqual(first["episode_length"], 2)
        self.assertEqual(first["tool_call_count"], 1)
        self.assertEqual(first["evaluation_provenance"], self.provenance)
        self.assertEqual(first["steps"][0]["tool_output"], ["doc-a"])
        self.assertEqual(second["steps"][0]["tool_output"], [1, 2])
        json.dumps(records, allow_nan=False)

    def test_missing_required_flattened_field_is_rejected(self):
        for field in (
            "traj_uid",
            "data_source",
            "tool_output",
            "rewards",
            "active_masks",
            "episode_rewards",
            "episode_lengths",
            "tool_callings",
            "is_action_valid",
        ):
            with self.subTest(field=field):
                rows = self.rows()
                rows.pop(field)
                with self.assertRaisesRegex(KeyError, field):
                    self.build(rows=rows)

    def test_fractional_or_inconsistent_strict_em_is_rejected(self):
        rows = self.rows()
        rows["episode_rewards"][0] = 0.5
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            self.build(rows=rows)

        rows = self.rows()
        rows["episode_rewards"][1] = 0.0
        with self.assertRaisesRegex(ValueError, "inconsistent episode_reward"):
            self.build(rows=rows)

        rows = self.rows()
        rows["rewards"] = np.array([1.0, 0.0, 0.0], dtype=object)
        with self.assertRaisesRegex(ValueError, "terminal step"):
            self.build(rows=rows)

        with self.assertRaisesRegex(ValueError, "validation score rows"):
            self.build(scores=[1.0, 0.0, 0.0])

    def test_inactive_or_multielement_bool_is_rejected(self):
        rows = self.rows()
        rows["active_masks"][1] = False
        with self.assertRaisesRegex(ValueError, "inactive row"):
            self.build(rows=rows)

        rows = self.rows()
        rows["active_masks"][0] = np.array([True, False])
        with self.assertRaisesRegex(TypeError, "scalar"):
            self.build(rows=rows)

    def test_episode_length_must_match_steps_and_max_steps(self):
        rows = self.rows()
        rows["episode_lengths"][:2] = 3.0
        with self.assertRaisesRegex(ValueError, "step count"):
            self.build(rows=rows)

        rows = self.rows()
        rows["episode_lengths"][:2] = 5.0
        with self.assertRaisesRegex(ValueError, "exceeds max_steps"):
            self.build(rows=rows)

    def test_gold_snapshot_and_first_seen_mapping(self):
        contexts = snapshot_validation_gold_contexts(
            initial_inputs=["input A", "input B"],
            non_tensor_batch={
                "data_source": np.array(["bamboogle", "bamboogle"], dtype=object),
                "env_kwargs": np.array(
                    [
                        {
                            "question": "Question A",
                            "ground_truth": {"target": ["A"]},
                            "data_source": "bamboogle",
                        },
                        {
                            "question": "Question B",
                            "ground_truth": {"target": ["B"]},
                            "data_source": "bamboogle",
                        },
                    ],
                    dtype=object,
                ),
            },
        )
        records = self.build(contexts=contexts)
        self.assertEqual(records[0]["question"], "Question A")
        self.assertEqual(records[1]["question"], "Question B")
        with self.assertRaisesRegex(ValueError, "trajectory/gold count mismatch"):
            self.build(contexts=contexts[:1])

    def test_metric_crosscheck_per_data_source(self):
        records = self.build()
        validate_validation_trajectory_metrics(
            records,
            {"bamboogle_success_rate": 0.5, "success_rate": 0.5},
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_validation_trajectory_metrics(
                records, {"bamboogle_success_rate": 0.6, "success_rate": 0.5}
            )

    def test_atomic_write_duplicate_and_overwrite_refusal(self):
        records = self.build()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = dump_validation_trajectory_records(
                validation_data_dir=temp_dir, global_step=450, records=records
            )
            path = Path(output)
            original = path.read_bytes()
            self.assertEqual(path.name, "val_step_000450.jsonl")
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(list(Path(temp_dir).glob(".*.tmp")), [])
            with self.assertRaises(FileExistsError):
                dump_validation_trajectory_records(
                    validation_data_dir=temp_dir, global_step=450, records=records
                )
            self.assertEqual(path.read_bytes(), original)

        duplicate = [records[0], copy.deepcopy(records[0])]
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "unique"):
                dump_validation_trajectory_records(
                    validation_data_dir=temp_dir, global_step=1, records=duplicate
                )

    def test_disabled_dump_creates_no_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(
                dump_validation_trajectory_records(
                    validation_data_dir=None, global_step=0, records=self.build()
                )
            )
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_strict_normalized_em_is_unchanged(self):
        target = {"target": ["The Beatles"]}
        self.assertEqual(compute_score("<answer>Beatles</answer>", target), 1)
        self.assertEqual(
            compute_score("<answer>The Beatles were an English band</answer>", target), 0
        )
        self.assertEqual(compute_score("<answer>Beat</answer>", target), 0)


if __name__ == "__main__":
    unittest.main()
