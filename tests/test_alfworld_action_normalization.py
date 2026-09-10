from reward_methods.action_format import safe_normalize_action_text, safe_repair_action
from reward_methods.diagnostics import parse_diagnostics
from reward_methods.normalization import FCSRConfig
from reward_methods.schema import StepRecord, TrajectoryRecord


def test_safe_normalization_strips_format_noise_only():
    assert safe_normalize_action_text('Action: "Open Drawer 1."') == "open drawer 1"
    assert safe_normalize_action_text("<action>  TAKE Apple FROM Table! </action>") == "take apple from table"


def test_safe_repair_requires_exact_admissible_match():
    repaired, normalized, did_repair, reason = safe_repair_action(
        'Action: "Open Drawer 1."',
        ["open drawer 1", "close drawer 1"],
    )
    assert repaired == "open drawer 1"
    assert normalized == "open drawer 1"
    assert did_repair is True
    assert reason == "normalized_exact_admissible_match"

    repaired, normalized, did_repair, reason = safe_repair_action(
        "please open the thing",
        ["open drawer 1"],
    )
    assert repaired == "please open the thing"
    assert normalized == "please open the thing"
    assert did_repair is False
    assert reason == "normalized_not_admissible"


def test_diagnostics_records_raw_and_normalized_action_without_hiding_raw():
    step = StepRecord(
        task_id="u",
        env_name="alfworld/AlfredTWEnv",
        step_id=0,
        observation="You are at a drawer.",
        action='Action: "Open Drawer 1."',
        tool_output="The drawer is open.",
    )
    step.diagnostic["valid_action"] = True
    step.diagnostic["admissible_actions"] = ["open drawer 1"]
    step.diagnostic["task_instruction"] = "pick the apple from the drawer."
    diag = parse_diagnostics(step, None, TrajectoryRecord("u", "alfworld/AlfredTWEnv", [step]), FCSRConfig(alfworld_training_chain="optimized"))
    assert diag["raw_action"] == 'Action: "Open Drawer 1."'
    assert diag["normalized_action"] == "open drawer 1"
    assert diag["action_repaired"] is True
