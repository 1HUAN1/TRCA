from reward_methods.diagnostics import parse_alfworld_task_instruction, parse_diagnostics
from reward_methods.normalization import FCSRConfig
from reward_methods.schema import StepRecord, TrajectoryRecord


def _cfg():
    return FCSRConfig(alfworld_training_chain="optimized")


def test_alfworld_task_instruction_parser_extracts_goal_fields():
    parsed = parse_alfworld_task_instruction("examine the box with the desklamp.")
    assert parsed["task_type"] == "look"
    assert parsed["target_object"] == "box"
    assert parsed["required_tool"] == "desklamp"
    assert parsed["target_action"] == "examine"


def test_unrelated_take_has_low_goal_relevance():
    step = StepRecord(
        task_id="u",
        env_name="alfworld/AlfredTWEnv",
        step_id=0,
        observation="You are in a room with a lettuce and a box.",
        action="<action>take lettuce from table</action>",
        tool_output="You pick up the lettuce.",
    )
    step.diagnostic["valid_action"] = True
    step.diagnostic["task_instruction"] = "examine the box with the desklamp."
    diag = parse_diagnostics(step, None, TrajectoryRecord("u", "alfworld/AlfredTWEnv", [step]), _cfg())
    assert diag["goal_relevance"] < 0.35
    assert diag["completion_critical_action"] is False


def test_goal_actions_have_high_goal_relevance():
    for action in ("<action>use desklamp</action>", "<action>examine box</action>"):
        step = StepRecord(
            task_id="u",
            env_name="alfworld/AlfredTWEnv",
            step_id=0,
            observation="You are near a box and a desklamp.",
            action=action,
            tool_output="The box is now visible.",
        )
        step.diagnostic["valid_action"] = True
        step.diagnostic["task_instruction"] = "examine the box with the desklamp."
        diag = parse_diagnostics(step, None, TrajectoryRecord("u", "alfworld/AlfredTWEnv", [step]), _cfg())
        assert diag["goal_relevance"] >= 0.3
        assert diag["completion_critical_action"] is True


def test_repeat_noop_invalid_are_not_completion_critical():
    prev = StepRecord(
        task_id="u",
        env_name="alfworld/AlfredTWEnv",
        step_id=0,
        observation="You are near a desklamp.",
        action="<action>use desklamp</action>",
        tool_output="Nothing happens.",
    )
    step = StepRecord(
        task_id="u",
        env_name="alfworld/AlfredTWEnv",
        step_id=1,
        observation="You are near a desklamp.",
        action="<action>use desklamp</action>",
        tool_output="Nothing happens.",
    )
    step.diagnostic["valid_action"] = False
    step.diagnostic["task_instruction"] = "examine the box with the desklamp."
    traj = TrajectoryRecord("u", "alfworld/AlfredTWEnv", [prev, step])
    diag = parse_diagnostics(step, prev, traj, _cfg())
    assert diag["no_op"] is True
    assert diag["repeated_action"] is True
    assert diag["completion_critical_action"] is False
