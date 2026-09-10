"""Collect SearchQA validation EM metrics from trainer logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("nq", "triviaqa", "popqa", "hotpotqa", "2wiki", "musique", "bamboogle")
DATASET_SOURCE_KEYS = {
    "nq": ("nq",),
    "triviaqa": ("triviaqa",),
    "popqa": ("popqa",),
    "hotpotqa": ("hotpotqa",),
    "2wiki": ("2wiki", "2wikimultihopqa"),
    "musique": ("musique",),
    "bamboogle": ("bamboogle",),
}
SOTA_TARGETS = {
    "3b": {
        "nq": 44.4,
        "triviaqa": 60.5,
        "popqa": 45.5,
        "hotpotqa": 38.6,
        "2wiki": 37.0,
        "musique": 17.4,
        "bamboogle": 64.5,
        "average": 44.3,
    },
    "7b": {
        "nq": 46.4,
        "triviaqa": 65.5,
        "popqa": 51.5,
        "hotpotqa": 42.1,
        "2wiki": 43.6,
        "musique": 22.6,
        "bamboogle": 69.0,
        "average": 48.3,
    },
}
PREFIX = "VALIDATION_METRIC_JSON: "


def read_last_validation_metrics(path: Path) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if PREFIX not in line:
                continue
            payload = line.split(PREFIX, 1)[1].strip()
            last = json.loads(payload)
    if last is None:
        raise ValueError(f"No {PREFIX.strip()} record found in {path}")
    return last


def rows_from_logs(log_paths: Iterable[Path], model_size: str) -> list[dict[str, Any]]:
    targets = SOTA_TARGETS[model_size]
    rows: dict[str, dict[str, Any]] = {}
    for path in log_paths:
        metrics = read_last_validation_metrics(path)
        for dataset in DATASETS:
            strict_em = None
            expanded_step_score = None
            matched_source = dataset
            for source_key in DATASET_SOURCE_KEYS[dataset]:
                score_key = f"val/{source_key}/test_score"
                success_key = f"val/{source_key}_success_rate"
                if success_key in metrics:
                    strict_em = float(metrics[success_key])
                if score_key in metrics:
                    expanded_step_score = float(metrics[score_key])
                if strict_em is not None or expanded_step_score is not None:
                    matched_source = source_key
                    break
            if strict_em is None and expanded_step_score is None:
                continue
            if dataset in rows:
                raise ValueError(f"Multiple validation logs supplied for {dataset}")
            if strict_em is None:
                strict_em = expanded_step_score
            if expanded_step_score is None:
                expanded_step_score = strict_em
            em_percent = 100.0 * strict_em
            target = targets[dataset]
            rows[dataset] = {
                "dataset": dataset,
                "metric": "strict_em",
                "em_percent": em_percent,
                "success_rate_percent": em_percent,
                "expanded_step_score_percent": 100.0 * expanded_step_score,
                "tool_call_count_mean": float(metrics.get(f"val/{matched_source}/tool_call_count/mean", 0.0)),
                "sota_target_percent": target,
                "gap_to_target_percent": em_percent - target,
                "strictly_exceeds_target": em_percent > target,
                "log_path": str(path),
            }
    return [rows[dataset] for dataset in DATASETS if dataset in rows]


def build_summary(rows: list[dict[str, Any]], model_size: str) -> dict[str, Any]:
    all_seven = len(rows) == len(DATASETS)
    average = sum(row["em_percent"] for row in rows) / len(rows) if rows else 0.0
    average_target = SOTA_TARGETS[model_size]["average"]
    cell_target_average = sum(SOTA_TARGETS[model_size][dataset] for dataset in DATASETS) / len(DATASETS)
    return {
        "model_size": model_size,
        "primary_metric": "strict_em",
        "datasets": rows,
        "datasets_present": [row["dataset"] for row in rows],
        "all_seven_present": all_seven,
        "average_em_percent": average,
        "average_sota_target_percent": average_target,
        "cell_target_macro_average_percent": cell_target_average,
        "external_average_target_differs_from_cell_macro": abs(average_target - cell_target_average) > 1e-9,
        "average_gap_to_target_percent": average - average_target,
        "sota_candidate": all_seven
        and average > average_target
        and all(bool(row["strictly_exceeds_target"]) for row in rows),
    }


def write_summary(summary: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = (
        "dataset",
        "metric",
        "em_percent",
        "success_rate_percent",
        "expanded_step_score_percent",
        "tool_call_count_mean",
        "sota_target_percent",
        "gap_to_target_percent",
        "strictly_exceeds_target",
        "log_path",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["datasets"])


def write_latex(summary: dict[str, Any], latex_path: Path) -> None:
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    by_dataset = {row["dataset"]: row for row in summary["datasets"]}
    values = [f'{by_dataset[dataset]["em_percent"]:.1f}' if dataset in by_dataset else "--" for dataset in DATASETS]
    average = f'{summary["average_em_percent"]:.1f}' if summary["datasets"] else "--"
    row = "RL Training & \\textbf{FGPO (Ours)} & " + " & ".join(values) + f" & {average} \\\\\n"
    latex_path.write_text(row, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=Path, default=[])
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--model-size", choices=sorted(SOTA_TARGETS), required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path)
    parser.add_argument("--require-all-seven", action="store_true")
    args = parser.parse_args()

    log_paths = list(args.log)
    if args.log_root is not None:
        log_paths.extend(sorted(args.log_root.glob("**/train.log")))
    if not log_paths:
        parser.error("provide at least one --log or --log-root")
    missing = [str(path) for path in log_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validation logs: {missing}")

    rows = rows_from_logs(log_paths, args.model_size)
    summary = build_summary(rows, args.model_size)
    write_summary(summary, args.json_output, args.csv_output)
    if args.latex_output is not None:
        write_latex(summary, args.latex_output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_all_seven and not summary["all_seven_present"]:
        raise SystemExit(f"missing validation datasets: {sorted(set(DATASETS) - set(summary['datasets_present']))}")


if __name__ == "__main__":
    main()
