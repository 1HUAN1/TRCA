import unittest

from omegaconf import OmegaConf

from agent_system.environments.env_manager import SearchEnvironmentManager
from agent_system.environments.prompts.search import (
    SEARCH_TEMPLATE,
    SEARCH_TEMPLATE_NO_HIS,
)
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import (
    compute_score,
)


class _Memory:
    def __init__(self, step_count=0, context=""):
        self.step_count = step_count
        self.context = context

    def __getitem__(self, index):
        return [None] * self.step_count

    def fetch(self, history_length, obs_key, action_key):
        return [self.context], [self.step_count]


def _manager(style=None, step_count=0, history_length=4):
    search = {} if style is None else {"prompt_style": style}
    manager = SearchEnvironmentManager.__new__(SearchEnvironmentManager)
    manager.config = OmegaConf.create(
        {
            "env": {
                "max_steps": 4,
                "history_length": history_length,
                "search": search,
            }
        }
    )
    manager.tasks = ["Which entity is connected through two relations?"]
    manager.memory = _Memory(step_count, "<search>first relation</search>")
    return manager


class SearchMultihopPromptTests(unittest.TestCase):
    def test_default_prompt_is_unchanged(self):
        manager = _manager()
        actual = manager.build_text_obs([manager.tasks[0]], init=True)[0]
        self.assertEqual(
            actual,
            SEARCH_TEMPLATE_NO_HIS.format(task_description=manager.tasks[0]),
        )

        manager = _manager(step_count=2)
        actual = manager.build_text_obs(["result"], init=False)[0]
        self.assertEqual(
            actual,
            SEARCH_TEMPLATE.format(
                task_description=manager.tasks[0],
                memory_context=manager.memory.context,
                step_count=2,
            ),
        )

    def test_multihop_prompt_reports_remaining_actions(self):
        for step_count, expected in [(0, 4), (1, 3), (2, 2), (3, 1)]:
            with self.subTest(step_count=step_count):
                manager = _manager("multihop_bridge_v1", step_count=step_count)
                prompt = manager.build_text_obs(
                    [manager.tasks[0] if step_count == 0 else "result"],
                    init=step_count == 0,
                )[0]
                self.assertIn(f"You have {expected} action(s) left", prompt)
                self.assertIn("bridge entity", prompt)
                self.assertIn("do not force two searches", prompt)
                self.assertIn("emit exactly one complete action block", prompt)
                self.assertNotIn("bamboogle", prompt.lower())

    def test_last_action_requires_answer_instead_of_unobservable_search(self):
        manager = _manager("multihop_bridge_v1", step_count=3)
        prompt = manager.build_text_obs(["result"], init=False)[0]
        self.assertIn("If exactly one action remains, do not search", prompt)
        self.assertIn("return the best supported final answer now", prompt)

    def test_multihop_policy_does_not_depend_on_dataset_source(self):
        manager = _manager("multihop_bridge_v1")
        prompt = manager.build_text_obs([manager.tasks[0]], init=True)[0]
        self.assertNotIn("data_source", prompt)
        self.assertNotIn("Bamboogle", prompt)

    def test_unknown_prompt_style_is_rejected(self):
        manager = _manager("typo")
        with self.assertRaisesRegex(
            ValueError, "Unsupported SearchQA prompt style"
        ):
            manager.build_text_obs([manager.tasks[0]], init=True)

    def test_main_metric_remains_normalized_strict_em(self):
        target = {"target": ["The Beatles"]}
        self.assertEqual(compute_score("<answer>Beatles</answer>", target), 1)
        self.assertEqual(
            compute_score(
                "<answer>The Beatles were an English band</answer>", target
            ),
            0,
        )
        self.assertEqual(compute_score("<answer>Beat</answer>", target), 0)


if __name__ == "__main__":
    unittest.main()
