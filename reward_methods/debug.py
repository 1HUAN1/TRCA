"""Debug JSONL dumping for reward methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from reward_methods.schema import TrajectoryRecord, trajectory_to_json


def _safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(key): _safe_json(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe_json(value) for value in obj]
        return str(obj)


def dump_reward_trajectories(trajectories: list[TrajectoryRecord], dump_dir: str, prefix: str = "fcsr_rewards") -> str:
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    path = Path(dump_dir) / f"{prefix}_{int(time.time() * 1000)}.jsonl"
    with path.open("w", encoding="utf-8") as fout:
        for traj in trajectories:
            fout.write(json.dumps(_safe_json(trajectory_to_json(traj)), ensure_ascii=False) + "\n")
    return str(path)

