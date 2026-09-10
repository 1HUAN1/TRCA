from reward_methods.schema import extract_action, extract_task, trajectory_from_existing_rollout


def test_extract_action_from_tags():
    assert extract_action("<think>x</think><action>open drawer 1</action>") == "open drawer 1"


def test_trajectory_adapter_fills_next_observation_and_success():
    raw = [
        {
            "input": "Your task is to: examine the box\ncurrent observation is: You are in a room.\nYour admissible actions are: ['go to drawer 1']",
            "output": "<action>go to drawer 1</action>",
            "uid": "u1",
            "traj_uid": "t1",
            "rewards": 0.0,
            "active_masks": True,
            "is_action_valid": True,
        },
        {
            "input": "Your task is to: examine the box\ncurrent observation is: You arrive at drawer 1.\nYour admissible actions are: ['open drawer 1']",
            "output": "<action>open drawer 1</action>",
            "uid": "u1",
            "traj_uid": "t1",
            "rewards": 1.0,
            "active_masks": True,
            "is_action_valid": True,
        },
    ]
    traj = trajectory_from_existing_rollout(raw, env_name="alfworld/AlfredTWEnv")
    assert traj.task_id == "u1"
    assert traj.final_success is True
    assert traj.terminal_reward == 1.0
    assert traj.steps[0].tool_output == "You arrive at drawer 1."
    assert traj.steps[0].diagnostic["admissible_actions"] == ["go to drawer 1"]

def test_extract_task_accepts_search_r1_question_prefix():
    prompt = "Use <search> when needed. Question: Who wrote Hamlet?\n"
    assert extract_task(prompt) == "Who wrote Hamlet?"


def test_search_last_step_keeps_explicit_post_action_tool_output():
    raw = [
        {
            "input": "Question: Who wrote Hamlet?",
            "output": "<search>Hamlet author</search>",
            "tool_output": "<information>Hamlet was written by William Shakespeare.</information>",
            "next_observation": "<information>Hamlet was written by William Shakespeare.</information>",
            "uid": "u-search",
            "traj_uid": "t-search",
            "rewards": 0.0,
            "active_masks": True,
            "is_action_valid": True,
        }
    ]
    traj = trajectory_from_existing_rollout(raw, env_name="search")
    assert "William Shakespeare" in traj.steps[0].tool_output
    assert "William Shakespeare" in str(traj.steps[0].next_state)
