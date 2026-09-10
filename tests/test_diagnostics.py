from reward_methods.diagnostics import parse_diagnostics
from reward_methods.normalization import FCSRConfig
from reward_methods.schema import StepRecord, TrajectoryRecord


def test_alfworld_invalid_noop_is_risky():
    cfg = FCSRConfig()
    step = StepRecord(
        task_id="u",
        env_name="alfworld/AlfredTWEnv",
        step_id=0,
        observation="You are at a drawer.",
        action="open drawer 2",
        tool_output="Nothing happens.",
    )
    step.diagnostic["valid_action"] = False
    diag = parse_diagnostics(step, None, TrajectoryRecord("u", "alfworld/AlfredTWEnv", [step]), cfg)
    assert diag["no_op"] is True
    assert diag["risk"] > 0.5
    assert diag["progress_delta"] <= 0.05


def test_webshop_premature_buy_detected():
    cfg = FCSRConfig()
    step = StepRecord(
        task_id="u",
        env_name="webshop/WebShopEnv",
        step_id=0,
        observation="Find a red shirt size medium.",
        action="buy now",
        tool_output="Please select options first.",
    )
    traj = TrajectoryRecord("u", "webshop/WebShopEnv", [step], final_success=False)
    diag = parse_diagnostics(step, None, traj, cfg)
    assert diag["premature_buy"] is True
    assert diag["risk"] >= 0.7


def test_sokoban_deadlock_detected():
    cfg = FCSRConfig()
    step = StepRecord(
        task_id="u",
        env_name="sokoban",
        step_id=0,
        observation="#####\n#$  #\n# . #\n#####",
        action="left",
        tool_output="#####\n#$  #\n# . #\n#####",
    )
    diag = parse_diagnostics(step, None, TrajectoryRecord("u", "sokoban", [step]), cfg)
    assert diag["deadlock_risk"] == 1.0

def test_search_question_text_is_not_mistaken_for_retrieval_evidence():
    cfg = FCSRConfig()
    step = StepRecord(
        task_id="u",
        env_name="search",
        step_id=0,
        observation="Question: Who wrote Hamlet?",
        action="<search>Hamlet author</search>",
        tool_output="",
    )
    step.diagnostic["task_instruction"] = "Who wrote Hamlet?"
    diag = parse_diagnostics(step, None, TrajectoryRecord("u", "search", [step]), cfg)
    assert diag["retrieval_success"] is False
    assert diag["tool_error"] is True
    assert diag["evidence_gain"] == 0.0


def test_search_bridge_query_accumulates_progress_without_gold_supporting_facts():
    cfg = FCSRConfig()
    first = StepRecord(
        task_id="u",
        env_name="search",
        step_id=0,
        observation="Question: What nationality was the director of Sinister?",
        action="<search>Sinister director</search>",
        tool_output="<information>Sinister was directed by Scott Derrickson.</information>",
    )
    first.diagnostic["task_instruction"] = "What nationality was the director of Sinister?"
    second = StepRecord(
        task_id="u",
        env_name="search",
        step_id=1,
        observation=first.tool_output,
        action="<search>Scott Derrickson nationality</search>",
        tool_output="<information>Scott Derrickson is an American film director.</information>",
    )
    second.diagnostic["task_instruction"] = first.diagnostic["task_instruction"]
    traj = TrajectoryRecord("u", "search", [first, second], final_success=False)
    first.diagnostic.update(parse_diagnostics(first, None, traj, cfg))
    second_diag = parse_diagnostics(second, first, traj, cfg)
    assert second_diag["query_contains_bridge_candidate"] is True
    assert second_diag["evidence_gain"] > 0.0
    assert second_diag["progress_delta"] > 0.0
    assert second_diag["evidence_coverage"] > first.diagnostic["evidence_coverage"]


def test_search_repeated_query_is_penalized_and_not_rewarded_as_new_evidence():
    cfg = FCSRConfig()
    first = StepRecord(
        task_id="u",
        env_name="search",
        step_id=0,
        action="<search>Hamlet author</search>",
        tool_output="<information>Hamlet is a tragedy written by William Shakespeare.</information>",
    )
    first.diagnostic["task_instruction"] = "Who wrote Hamlet?"
    repeated = StepRecord(
        task_id="u",
        env_name="search",
        step_id=1,
        action="<search>Hamlet author</search>",
        tool_output=first.tool_output,
    )
    repeated.diagnostic["task_instruction"] = first.diagnostic["task_instruction"]
    traj = TrajectoryRecord("u", "search", [first, repeated], final_success=False)
    first.diagnostic.update(parse_diagnostics(first, None, traj, cfg))
    repeated_diag = parse_diagnostics(repeated, first, traj, cfg)
    assert repeated_diag["repeated_query"] is True
    assert repeated_diag["evidence_gain"] == 0.0
    assert repeated_diag["risk"] > first.diagnostic["risk"]


def test_search_terminal_answer_is_not_a_noop_and_failed_early_answer_is_premature():
    cfg = FCSRConfig()
    correct = StepRecord(task_id="u", env_name="search", step_id=0, action="<answer>Shakespeare</answer>")
    correct.diagnostic["task_instruction"] = "Who wrote Hamlet?"
    correct_diag = parse_diagnostics(correct, None, TrajectoryRecord("u", "search", [correct], final_success=True), cfg)
    assert correct_diag["no_op"] is False
    assert correct_diag["premature_answer"] is False
    assert correct_diag["progress_delta"] == 1.0

    wrong = StepRecord(task_id="u", env_name="search", step_id=0, action="<answer>Marlowe</answer>")
    wrong.diagnostic["task_instruction"] = "Who wrote Hamlet?"
    wrong_diag = parse_diagnostics(wrong, None, TrajectoryRecord("u", "search", [wrong], final_success=False), cfg)
    assert wrong_diag["no_op"] is False
    assert wrong_diag["premature_answer"] is True
    assert wrong_diag["premature_finish"] is True


def test_search_error_detection_uses_post_action_output_not_question_text():
    cfg = FCSRConfig()
    step = StepRecord(
        task_id="u",
        env_name="search",
        step_id=0,
        observation="Question: Which theorem cannot be applied here?",
        action="<search>theorem applicability</search>",
        tool_output="<information>The theorem applies only to continuous functions.</information>",
    )
    step.diagnostic["task_instruction"] = "Which theorem cannot be applied here?"
    diag = parse_diagnostics(step, None, TrajectoryRecord("u", "search", [step]), cfg)
    assert diag["valid_action"] is True
    assert diag["tool_error"] is False
    assert diag["retrieval_success"] is True
    assert diag["evidence_gain"] > 0.0


def test_search_known_empty_result_sentinel_is_a_tool_error():
    cfg = FCSRConfig()
    step = StepRecord(
        task_id="u",
        env_name="search",
        step_id=0,
        action="<search>unknown entity</search>",
        tool_output="No search results found",
    )
    step.diagnostic["task_instruction"] = "Who is the unknown entity?"
    diag = parse_diagnostics(step, None, TrajectoryRecord("u", "search", [step]), cfg)
    assert diag["tool_error"] is True
    assert diag["retrieval_success"] is False
    assert diag["evidence_gain"] == 0.0


def test_search_and_answer_in_same_turn_is_invalid_format():
    cfg = FCSRConfig()
    step = StepRecord(
        task_id="u",
        env_name="search",
        step_id=0,
        action="<search>Hamlet author</search><answer>Shakespeare</answer>",
        tool_output="<information>Hamlet was written by William Shakespeare.</information>",
    )
    step.diagnostic["task_instruction"] = "Who wrote Hamlet?"
    diag = parse_diagnostics(step, None, TrajectoryRecord("u", "search", [step]), cfg)
    assert diag["valid_action"] is False
    assert diag["query_valid"] is False
    assert diag["answer_format_valid"] is False
