"""Validation-only trajectory grouping and atomic JSONL output."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def json_safe(value: Any) -> Any:
    """Convert common rollout values to strict-JSON-compatible objects."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return json_safe(value.item() if value.numel() == 1 else value.tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.item() if value.ndim == 0 else value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(json_safe(key)): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _batch_values(
    batch: Mapping[str, Any], key: str, length: int, *, required: bool
) -> list[Any]:
    if key not in batch:
        if required:
            raise KeyError(f"validation trajectory field is missing: {key}")
        return [None] * length
    values = batch[key]
    if isinstance(values, torch.Tensor):
        values = [values] * length if values.ndim == 0 else list(values)
    elif isinstance(values, np.ndarray):
        values = [values.item()] * length if values.ndim == 0 else list(values)
    elif isinstance(values, (list, tuple)):
        values = list(values)
    elif length == 1:
        values = [values]
    else:
        raise ValueError(f"validation trajectory field {key!r} is not batched")
    if len(values) != length:
        raise ValueError(
            f"validation trajectory field {key!r} has {len(values)} rows; expected {length}"
        )
    return values


def _scalar(value: Any, label: str) -> Any:
    value = json_safe(value)
    if isinstance(value, (list, dict)):
        raise TypeError(f"{label} must be a scalar, got {value!r}")
    return value


def _finite_scalar(value: Any, label: str) -> float:
    value = _scalar(value, label)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite numeric scalar, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return value


def _strict_binary(value: Any, label: str) -> float:
    value = _scalar(value, label)
    if isinstance(value, bool):
        value = float(value)
    elif isinstance(value, (int, float)):
        value = float(value)
    else:
        raise TypeError(f"{label} must be scalar 0 or 1, got {value!r}")
    if not math.isfinite(value) or value not in (0.0, 1.0):
        raise ValueError(f"{label} must be finite scalar 0 or 1, got {value!r}")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    value = _scalar(value, label)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(_strict_binary(value, label))
    raise TypeError(f"{label} must be a boolean scalar, got {value!r}")


def _nonnegative_integer(value: Any, label: str, *, positive: bool = False) -> int:
    value = _finite_scalar(value, label)
    if not value.is_integer() or value < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer, got {value!r}")
    return int(value)


def build_validation_provenance(
    *,
    prompt_style: Any,
    global_step: Any,
    resume_path: Any,
    topk: Any,
    max_steps: Any,
    val_n: Any,
    temperature: Any,
    do_sample: Any,
    top_p: Any = None,
    top_k: Any = None,
) -> dict[str, Any]:
    return {
        "prompt_style": str(_scalar(prompt_style, "prompt_style")),
        "global_step": _nonnegative_integer(global_step, "global_step"),
        "resume_path": json_safe(resume_path),
        "topk": _nonnegative_integer(topk, "topk", positive=True),
        "max_steps": _nonnegative_integer(max_steps, "max_steps", positive=True),
        "val_n": _nonnegative_integer(val_n, "val_n", positive=True),
        "temperature": _finite_scalar(temperature, "temperature"),
        "do_sample": _strict_bool(do_sample, "do_sample"),
        "top_p": json_safe(top_p),
        "top_k": json_safe(top_k),
    }


def snapshot_validation_gold_contexts(
    *, initial_inputs: Iterable[Any], non_tensor_batch: Mapping[str, Any]
) -> list[dict[str, Any]]:
    initial_inputs = list(initial_inputs)
    length = len(initial_inputs)
    data_sources = _batch_values(
        non_tensor_batch, "data_source", length, required=True
    )
    env_kwargs = _batch_values(non_tensor_batch, "env_kwargs", length, required=True)
    contexts = []
    for index, (initial_input, data_source, kwargs) in enumerate(
        zip(initial_inputs, data_sources, env_kwargs)
    ):
        kwargs = json_safe(kwargs)
        if not isinstance(kwargs, Mapping):
            raise TypeError(f"env_kwargs[{index}] must be a mapping")
        if "question" not in kwargs or "ground_truth" not in kwargs:
            raise KeyError(
                f"env_kwargs[{index}] must contain question and ground_truth"
            )
        source = _scalar(data_source, f"data_source[{index}]")
        env_source = kwargs.get("data_source", source)
        if _scalar(env_source, f"env_kwargs[{index}].data_source") != source:
            raise ValueError(f"data_source mismatch at validation item {index}")
        contexts.append(
            {
                "question": json_safe(kwargs["question"]),
                "ground_truth": json_safe(kwargs["ground_truth"]),
                "data_source": json_safe(source),
                "initial_input": json_safe(initial_input),
            }
        )
    return contexts


def build_validation_trajectory_records(
    *,
    prompts: Iterable[Any],
    raw_actions: Iterable[Any],
    validation_scores: Iterable[Any],
    non_tensor_batch: Mapping[str, Any],
    gold_contexts: Iterable[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Group flattened active rollout rows into one audited record per trajectory."""
    prompts = list(prompts)
    raw_actions = list(raw_actions)
    validation_scores = list(validation_scores)
    gold_contexts = [json_safe(context) for context in gold_contexts]
    provenance = json_safe(provenance)
    length = len(raw_actions)
    if len(prompts) != length or len(validation_scores) != length:
        raise ValueError("prompts/actions/validation_scores must have identical row counts")
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    max_steps = _nonnegative_integer(
        provenance.get("max_steps"), "provenance.max_steps", positive=True
    )

    required_fields = (
        "traj_uid",
        "data_source",
        "tool_output",
        "rewards",
        "active_masks",
        "episode_rewards",
        "episode_lengths",
        "tool_callings",
        "is_action_valid",
    )
    values = {
        key: _batch_values(non_tensor_batch, key, length, required=True)
        for key in required_fields
    }

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row_index in range(length):
        if not _strict_bool(values["active_masks"][row_index], f"active_masks[{row_index}]"):
            raise ValueError("flattened validation output contains an inactive row")
        trajectory_id_value = _scalar(
            values["traj_uid"][row_index], f"traj_uid[{row_index}]"
        )
        trajectory_id = str(trajectory_id_value)
        if not trajectory_id:
            raise ValueError(f"empty traj_uid at validation row {row_index}")
        source = _scalar(values["data_source"][row_index], f"data_source[{row_index}]")
        episode_reward = _strict_binary(
            values["episode_rewards"][row_index], f"episode_rewards[{row_index}]"
        )
        raw_reward = _strict_binary(values["rewards"][row_index], f"rewards[{row_index}]")
        validation_score = _strict_binary(
            validation_scores[row_index], f"validation_scores[{row_index}]"
        )
        episode_length = _nonnegative_integer(
            values["episode_lengths"][row_index],
            f"episode_lengths[{row_index}]",
            positive=True,
        )
        tool_count = _nonnegative_integer(
            values["tool_callings"][row_index], f"tool_callings[{row_index}]"
        )
        if episode_length > max_steps:
            raise ValueError(
                f"episode_length {episode_length} exceeds max_steps {max_steps}"
            )
        if tool_count > episode_length:
            raise ValueError("tool_call_count cannot exceed episode_length")

        record = grouped.get(trajectory_id)
        if record is None:
            record = {
                "trajectory_id": trajectory_id,
                "data_source": json_safe(source),
                "strict_success": bool(episode_reward),
                "terminal_score": episode_reward,
                "episode_length": episode_length,
                "tool_call_count": tool_count,
                "validation_score": validation_score,
                "evaluation_provenance": dict(provenance),
                "steps": [],
                "_raw_rewards": [],
                "_validation_scores": [],
            }
            grouped[trajectory_id] = record
        else:
            if record["data_source"] != json_safe(source):
                raise ValueError(f"inconsistent data_source for trajectory {trajectory_id}")
            for field, actual, expected in (
                ("episode_reward", episode_reward, record["terminal_score"]),
                ("episode_length", episode_length, record["episode_length"]),
                ("tool_call_count", tool_count, record["tool_call_count"]),
            ):
                if actual != expected:
                    raise ValueError(f"inconsistent {field} for trajectory {trajectory_id}")

        record["_raw_rewards"].append(raw_reward)
        record["_validation_scores"].append(validation_score)
        record["steps"].append(
            {
                "step": len(record["steps"]) + 1,
                "prompt": json_safe(prompts[row_index]),
                "raw_action": json_safe(raw_actions[row_index]),
                "tool_output": json_safe(values["tool_output"][row_index]),
                "reward": raw_reward,
                "is_action_valid": _strict_bool(
                    values["is_action_valid"][row_index],
                    f"is_action_valid[{row_index}]",
                ),
                "active_mask": True,
            }
        )

    if len(grouped) != len(gold_contexts):
        raise ValueError(
            f"trajectory/gold count mismatch: {len(grouped)} trajectories vs "
            f"{len(gold_contexts)} gold contexts"
        )

    records = []
    for (trajectory_id, record), context in zip(grouped.items(), gold_contexts):
        if not isinstance(context, Mapping):
            raise TypeError("gold context must be a mapping")
        if context.get("data_source") != record["data_source"]:
            raise ValueError(f"gold data_source mismatch for trajectory {trajectory_id}")
        if len(record["steps"]) != record["episode_length"]:
            raise ValueError(f"step count does not equal episode_length for {trajectory_id}")
        raw_rewards = record.pop("_raw_rewards")
        validation_row_scores = record.pop("_validation_scores")
        episode_reward = record["terminal_score"]
        if sum(raw_rewards) != episode_reward:
            raise ValueError(f"raw reward sum does not equal episode reward for {trajectory_id}")
        if raw_rewards[-1] != episode_reward or any(raw_rewards[:-1]):
            raise ValueError(f"strict EM must occur only on the terminal step for {trajectory_id}")
        # EpisodeRewardManager copies episode_rewards to every flattened active row.
        if any(score != episode_reward for score in validation_row_scores):
            raise ValueError(
                f"validation score rows do not match episode reward for {trajectory_id}"
            )
        record["question"] = json_safe(context.get("question"))
        record["ground_truth"] = json_safe(context.get("ground_truth"))
        record["initial_input"] = json_safe(context.get("initial_input"))
        records.append(record)
    return records


def validate_validation_trajectory_metrics(
    records: Iterable[Mapping[str, Any]], success_rates: Mapping[str, Any]
) -> None:
    records = list(records)
    by_source: dict[str, list[float]] = defaultdict(list)
    all_scores = []
    for record in records:
        source = str(_scalar(record.get("data_source"), "record.data_source"))
        score = _strict_binary(record.get("terminal_score"), "record.terminal_score")
        if _strict_bool(record.get("strict_success"), "record.strict_success") != bool(score):
            raise ValueError("strict_success does not match terminal_score")
        by_source[source].append(score)
        all_scores.append(score)
    for source, scores in by_source.items():
        key = f"{source}_success_rate"
        if key not in success_rates:
            raise KeyError(f"missing validation success metric: {key}")
        expected = _finite_scalar(success_rates[key], key)
        actual = sum(scores) / len(scores)
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"strict EM mean {actual} does not match {key}={expected}")
    if all_scores and "success_rate" in success_rates:
        actual = sum(all_scores) / len(all_scores)
        expected = _finite_scalar(success_rates["success_rate"], "success_rate")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"global strict EM mean {actual} does not match success_rate={expected}"
            )


def dump_validation_trajectory_records(
    *,
    validation_data_dir: str | os.PathLike[str] | None,
    global_step: int,
    records: Iterable[Mapping[str, Any]],
) -> str | None:
    if not validation_data_dir:
        return None
    records = list(records)
    trajectory_ids = [str(record.get("trajectory_id", "")) for record in records]
    if not all(trajectory_ids) or len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("validation trajectory IDs must be non-empty and unique")

    output_dir = Path(validation_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"val_step_{int(global_step):06d}.jsonl"
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite validation dump: {final_path}")
    fd, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=f".{final_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, final_path)
        os.unlink(temporary_name)
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return str(final_path)
