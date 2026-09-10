import json

from agent_system.evaluation.searchqa_eval import DATASETS, build_summary, rows_from_logs, write_latex, write_summary


def test_searchqa_eval_summary_uses_strict_em_and_sota_gap(tmp_path):
    log = tmp_path / "nq" / "train.log"
    log.parent.mkdir()
    log.write_text(
        "VALIDATION_METRIC_JSON: "
        + json.dumps(
            {
                "val/nq/test_score": 0.17,
                "val/nq_success_rate": 0.45,
                "val/nq/tool_call_count/mean": 1.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = rows_from_logs([log], "3b")
    summary = build_summary(rows, "3b")
    assert rows[0]["metric"] == "strict_em"
    assert rows[0]["em_percent"] == 45.0
    assert rows[0]["expanded_step_score_percent"] == 17.0
    assert round(rows[0]["gap_to_target_percent"], 6) == 0.6
    assert summary["all_seven_present"] is False
    assert summary["sota_candidate"] is False

    json_output = tmp_path / "summary.json"
    csv_output = tmp_path / "summary.csv"
    write_summary(summary, json_output, csv_output)
    assert json.loads(json_output.read_text())["primary_metric"] == "strict_em"
    assert "gap_to_target_percent" in csv_output.read_text()


def test_searchqa_eval_accepts_2wikimultihopqa_alias_and_success_only_metrics(tmp_path):
    metrics = {
        "val/nq_success_rate": 0.45,
        "val/triviaqa_success_rate": 0.61,
        "val/popqa_success_rate": 0.46,
        "val/hotpotqa_success_rate": 0.39,
        "val/2wikimultihopqa_success_rate": 0.38,
        "val/musique_success_rate": 0.18,
        "val/bamboogle_success_rate": 0.65,
    }
    log = tmp_path / "train.log"
    log.write_text("VALIDATION_METRIC_JSON: " + json.dumps(metrics) + "\n", encoding="utf-8")

    rows = rows_from_logs([log], "3b")
    summary = build_summary(rows, "3b")
    assert [row["dataset"] for row in rows] == list(DATASETS)
    assert summary["all_seven_present"] is True
    assert summary["external_average_target_differs_from_cell_macro"] is True
    assert next(row for row in rows if row["dataset"] == "2wiki")["em_percent"] == 38.0

    latex = tmp_path / "summary.tex"
    write_latex(summary, latex)
    text = latex.read_text(encoding="utf-8")
    assert "\\textbf{FGPO (Ours)}" in text
    assert text.rstrip().endswith("\\\\")
