# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (RewardFlow) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Core functions to implement RewardFlow algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to implement RewardFlow
"""

import uuid
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from difflib import SequenceMatcher

import numpy as np
import torch
from verl import DataProto


def _get_config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            value = config.get(key, default)
            if value is not None:
                return value
        except Exception:
            pass
    try:
        value = getattr(config, key)
        if value is not None:
            return value
    except Exception:
        pass
    return default


def _get_reward_method(config) -> str:
    method = _get_config_value(config, "reward_method", None)
    if method is None:
        algorithm = _get_config_value(config, "algorithm", None)
        method = _get_config_value(algorithm, "reward_method", "rewardflow")
    return str(method or "rewardflow").strip().lower()


def _decode_fcsr_step_outputs(total_batch_list, tokenizer) -> int:
    """Attach text inputs/outputs required by diagnostic reward methods."""
    if tokenizer is None:
        return 0
    decode_errors = 0
    for trajectory in total_batch_list:
        for step in trajectory:
            if not step.get("input"):
                raw_prompt = step.get("raw_prompt")
                if isinstance(raw_prompt, (list, tuple)):
                    contents = [item.get("content", "") for item in raw_prompt if isinstance(item, dict)]
                    step["input"] = "\n".join(str(content) for content in contents if content)
                elif raw_prompt:
                    step["input"] = str(raw_prompt)
                elif step.get("prompts") is not None:
                    prompt = step.get("prompts")
                    try:
                        if hasattr(prompt, "detach"):
                            prompt = prompt.detach().cpu()
                        if hasattr(prompt, "tolist"):
                            prompt = prompt.tolist()
                        if prompt and isinstance(prompt[0], (list, tuple)):
                            prompt = prompt[0]
                        step["input"] = tokenizer.decode(prompt, skip_special_tokens=True)
                    except Exception as exc:
                        decode_errors += 1
                        step["fcsr_prompt_decode_error"] = str(exc)
            if step.get("output"):
                continue
            response = step.get("responses")
            if response is None:
                continue
            try:
                if hasattr(response, "detach"):
                    response = response.detach().cpu()
                if hasattr(response, "tolist"):
                    response = response.tolist()
                if response and isinstance(response[0], (list, tuple)):
                    response = response[0]
                step["output"] = tokenizer.decode(response, skip_special_tokens=True)
            except Exception as exc:
                decode_errors += 1
                step["fcsr_decode_error"] = str(exc)
    return decode_errors

def to_hashable(x):
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list[int]):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size: list[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            


def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_returns(batch: DataProto):
    rewards = batch.non_tensor_batch['step_rewards'].astype(np.float32)
    all_returns = torch.tensor(rewards, dtype=torch.float32, device=batch.batch['input_ids'].device)
    
    return all_returns

def compute_rewardflow_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   ):
    
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode-level group reward
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
    
    # Compute step_group_uids
    step_group_uids = build_step_group(anchor_obs, index)

    # Compute step-level group reward
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_all_data: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for RewardFlow.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_all_data: bool
            If True (more stable), the mean and std are computed across all data in the batch. 
            If False (i.e., standard episode-level adv), the mean and std are computed across N trajectories.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_all_data:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: False)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    unique_indices = np.unique(index)

    group_size = []
    for idx in unique_indices:
        indices = np.where(index == idx)[0]
        obs_group = anchor_obs[indices]
        
        clusters = defaultdict(list)
        for i, obs in enumerate(obs_group):
            clusters[to_hashable(obs)].append(indices[i])
        
        for obs, original_indices in clusters.items():
            uid = str(uuid.uuid4())
            
            group_size.append(len(original_indices))
            for original_idx in original_indices:
                step_group_uids[original_idx] = uid

    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for RewardFlow.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages

def general_clean_trajectory(trajectory, idx_to_state):
    return trajectory


def _config_get(config, dotted_path, default=None):
    """Read nested dict/OmegaConf config values without requiring a schema."""
    current = config
    for part in dotted_path.split('.'):
        if current is None:
            return default
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        try:
            current = getattr(current, part)
            continue
        except (AttributeError, KeyError):
            pass
        getter = getattr(current, 'get', None)
        if getter is None:
            return default
        try:
            current = getter(part, default)
        except TypeError:
            return default
    return current


def _config_mapping(config_section):
    """Convert a config subsection with items() into a plain dict."""
    if config_section is None:
        return {}
    if isinstance(config_section, dict):
        return dict(config_section)
    items = getattr(config_section, 'items', None)
    if items is None:
        return {}
    return {key: value for key, value in items()}


def _as_bool(value):
    """Interpret Hydra-style bool/string config values."""
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _as_float(value, default):
    """Convert a scalar config value to float with a safe default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default):
    """Convert a scalar config value to int with a safe default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sasfr_enabled(config):
    """Return True only when both top-level and SASFR feature flags are enabled."""
    return (
        _as_bool(_config_get(config, 'failure_reuse.enabled', False))
        and _as_bool(_config_get(config, 'failure_reuse.sasfr.enabled', False))
    )


def _build_sasfr_config(config):
    """Build the pure-miner config from trainer config."""
    from rewardflow.failure_reuse import SASFRConfig

    weights = _config_mapping(_config_get(config, 'failure_reuse.sasfr.weights', None))
    if not weights:
        weights = None
    return SASFRConfig(
        selection_mode=str(_config_get(config, 'failure_reuse.sasfr.selection_mode', 'prefix_only')),
        random_seed=_as_int(_config_get(config, 'failure_reuse.sasfr.random_seed', 42), 42),
        min_score=_as_float(_config_get(config, 'failure_reuse.sasfr.min_score', 0.75), 0.75),
        success_reward_threshold=_as_float(
            _config_get(config, 'failure_reuse.sasfr.success_reward_threshold', 0.99), 0.99
        ),
        max_failure_ratio=_as_float(_config_get(config, 'failure_reuse.sasfr.max_failure_ratio', 0.10), 0.10),
        max_steps_per_failed_traj=_as_int(
            _config_get(config, 'failure_reuse.sasfr.max_steps_per_failed_traj', 1), 1
        ),
        max_steps_per_group=_as_int(_config_get(config, 'failure_reuse.sasfr.max_steps_per_group', 8), 8),
        partial_reward_scale=_as_float(
            _config_get(config, 'failure_reuse.sasfr.partial_reward_scale', 0.10), 0.10
        ),
        max_partial_reward=_as_float(_config_get(config, 'failure_reuse.sasfr.max_partial_reward', 0.10), 0.10),
        use_prefix_alignment=_as_bool(_config_get(config, 'failure_reuse.sasfr.use_prefix_alignment', True)),
        use_success_similarity=_as_bool(_config_get(config, 'failure_reuse.sasfr.use_success_similarity', False)),
        use_topology_score=_as_bool(_config_get(config, 'failure_reuse.sasfr.use_topology_score', False)),
        use_partial_progress=_as_bool(_config_get(config, 'failure_reuse.sasfr.use_partial_progress', False)),
        use_webshop_rules=_as_bool(_config_get(config, 'failure_reuse.sasfr.use_webshop_rules', False)),
        use_alfworld_rules=_as_bool(_config_get(config, 'failure_reuse.sasfr.use_alfworld_rules', False)),
        allow_anchorless_candidates=_as_bool(_config_get(config, 'failure_reuse.sasfr.allow_anchorless_candidates', False)),
        search_ratio_cap=_as_float(_config_get(config, 'failure_reuse.sasfr.search_ratio_cap', 0.25), 0.25),
        disable_search_reward=_as_bool(_config_get(config, 'failure_reuse.sasfr.disable_search_reward', False)),
        search_reward=_as_float(_config_get(config, 'failure_reuse.sasfr.search_reward', 0.03), 0.03),
        recovery_reward=_as_float(_config_get(config, 'failure_reuse.sasfr.recovery_reward', 0.06), 0.06),
        require_useful_search_after_recovery=_as_bool(
            _config_get(config, 'failure_reuse.sasfr.require_useful_search_after_recovery', True)
        ),
        product_click_reward=_as_float(_config_get(config, 'failure_reuse.sasfr.product_click_reward', 0.08), 0.08),
        option_click_reward=_as_float(_config_get(config, 'failure_reuse.sasfr.option_click_reward', 0.10), 0.10),
        buy_ready_reward=_as_float(_config_get(config, 'failure_reuse.sasfr.buy_ready_reward', 0.12), 0.12),
        weights=weights,
    )


def _sasfr_source_traj_uid(trajectory, uid, traj_idx):
    """Return the same stable source trajectory uid convention used by the miner."""
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        for key in ('source_traj_uid', 'traj_uid', 'trajectory_uid', 'rollout_uid', 'trajectory_id', 'id'):
            value = step.get(key)
            if value is not None:
                return str(value)
    return f'{uid}:traj{traj_idx}'


def _scalar_reward(value, default=0.0):
    """Convert reward values that may be Python, numpy, or scalar tensor objects."""
    if value is None:
        return default
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


FAILURE_REUSE_METADATA_KEYS = (
    'failure_reuse_selected',
    'failure_reuse_score',
    'failure_reuse_partial_reward',
    'failure_reuse_reference_traj_uid',
    'failure_reuse_reason',
    'failure_reuse_score_components',
    'failure_reuse_base_step_reward',
    'failure_reuse_augmented_step_reward',
    'failure_reuse_reward_delta',
    'failure_reuse_reward_applied',
)


def _failure_reuse_default_components():
    return {
        'prefix_alignment': 0.0,
        'success_similarity': 0.0,
        'topology_score': 0.0,
        'partial_progress': 0.0,
        'validity': 0.0,
        'bad_step_penalty': 0.0,
    }


def _initialize_failure_reuse_step_defaults(step):
    """Ensure selected-only SASFR metadata keys collate to the full step batch."""
    if not isinstance(step, dict):
        return
    base = _scalar_reward(step.get('step_rewards', step.get('rewards', 0.0)))
    defaults = {
        'failure_reuse_selected': False,
        'failure_reuse_score': 0.0,
        'failure_reuse_partial_reward': 0.0,
        'failure_reuse_reference_traj_uid': '',
        'failure_reuse_reason': '',
        'failure_reuse_score_components': _failure_reuse_default_components(),
        'failure_reuse_base_step_reward': base,
        'failure_reuse_augmented_step_reward': base,
        'failure_reuse_reward_delta': 0.0,
        'failure_reuse_reward_applied': False,
    }
    for key, value in defaults.items():
        if key not in step:
            step[key] = dict(value) if isinstance(value, dict) else value


def _initialize_failure_reuse_defaults(total_batch_list):
    for trajectory in total_batch_list:
        for step in trajectory:
            _initialize_failure_reuse_step_defaults(step)


def _apply_selected_failure_step(step, selected, config):
    """Attach SASFR metadata and optionally apply bounded partial step reward."""
    components = asdict(selected.components)
    metadata_only = _as_bool(_config_get(config, 'failure_reuse.sasfr.metadata_only', True))
    base = _scalar_reward(step.get('step_rewards', step.get('rewards', 0.0)))
    partial = _scalar_reward(selected.partial_reward)
    application = str(_config_get(config, 'failure_reuse.sasfr.reward_application', 'max')).lower()
    max_reward = _as_float(_config_get(config, 'failure_reuse.sasfr.max_augmented_step_reward', 1.0), 1.0)

    if metadata_only:
        new_value = base
    elif application == 'replace':
        new_value = partial
    elif application == 'add':
        new_value = base + partial
    else:
        new_value = max(base, partial)
    new_value = min(max_reward, new_value)
    delta = new_value - base
    reward_applied = (not metadata_only) and abs(delta) > 1e-12

    step['failure_reuse_selected'] = True
    step['failure_reuse_score'] = selected.reuse_score
    step['failure_reuse_partial_reward'] = selected.partial_reward
    step['failure_reuse_reference_traj_uid'] = selected.reference_success_traj_uid
    step['failure_reuse_reason'] = selected.reason
    step['failure_reuse_score_components'] = components
    step['failure_reuse_base_step_reward'] = base
    step['failure_reuse_augmented_step_reward'] = new_value
    step['failure_reuse_reward_delta'] = delta
    step['failure_reuse_reward_applied'] = reward_applied
    if not metadata_only:
        step['step_rewards'] = new_value

    return {
        'components': components,
        'base_step_reward': base,
        'augmented_step_reward': new_value,
        'reward_delta': delta,
        'reward_applied': reward_applied,
    }


def _selected_step_dump_row(selected, apply_info, config):
    """Build one JSONL row for a selected failed step."""
    row = asdict(selected)
    row['original_final_success'] = False
    row['selection_mode'] = str(_config_get(config, 'failure_reuse.sasfr.selection_mode', 'prefix_only'))
    row['metadata_only'] = _as_bool(_config_get(config, 'failure_reuse.sasfr.metadata_only', True))
    row['reward_applied'] = bool(apply_info.get('reward_applied', False))
    row['base_step_reward'] = _scalar_reward(apply_info.get('base_step_reward'))
    row['augmented_step_reward'] = _scalar_reward(apply_info.get('augmented_step_reward'))
    row['reward_delta'] = _scalar_reward(apply_info.get('reward_delta'))
    row['min_score'] = _as_float(_config_get(config, 'failure_reuse.sasfr.min_score', 0.75), 0.75)
    row['max_failure_ratio'] = _as_float(_config_get(config, 'failure_reuse.sasfr.max_failure_ratio', 0.10), 0.10)
    row['partial_reward_scale'] = _as_float(_config_get(config, 'failure_reuse.sasfr.partial_reward_scale', 0.10), 0.10)
    row['reward_application'] = str(_config_get(config, 'failure_reuse.sasfr.reward_application', 'max'))
    row['use_prefix_alignment'] = _as_bool(_config_get(config, 'failure_reuse.sasfr.use_prefix_alignment', True))
    row['use_success_similarity'] = _as_bool(_config_get(config, 'failure_reuse.sasfr.use_success_similarity', False))
    row['use_topology_score'] = _as_bool(_config_get(config, 'failure_reuse.sasfr.use_topology_score', False))
    row['search_ratio_cap'] = _as_float(_config_get(config, 'failure_reuse.sasfr.search_ratio_cap', 0.25), 0.25)
    row['disable_search_reward'] = _as_bool(_config_get(config, 'failure_reuse.sasfr.disable_search_reward', False))
    row['search_reward'] = _as_float(_config_get(config, 'failure_reuse.sasfr.search_reward', 0.03), 0.03)
    row['recovery_reward'] = _as_float(_config_get(config, 'failure_reuse.sasfr.recovery_reward', 0.06), 0.06)
    row['require_useful_search_after_recovery'] = _as_bool(
        _config_get(config, 'failure_reuse.sasfr.require_useful_search_after_recovery', True)
    )
    row['product_click_reward'] = _as_float(_config_get(config, 'failure_reuse.sasfr.product_click_reward', 0.08), 0.08)
    row['option_click_reward'] = _as_float(_config_get(config, 'failure_reuse.sasfr.option_click_reward', 0.10), 0.10)
    row['buy_ready_reward'] = _as_float(_config_get(config, 'failure_reuse.sasfr.buy_ready_reward', 0.12), 0.12)
    row['timestamp'] = time.time()
    row['pid'] = os.getpid()
    return row


def _dump_sasfr_selected_steps(rows, dump_dir):
    """Append selected failed-step JSONL rows without interrupting training on dump errors."""
    if not rows:
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        rank = os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or os.environ.get('SLURM_PROCID')
        pid = os.getpid()
        if rank is None:
            filename = f'sasfr_selected_steps_pid{pid}.jsonl'
        else:
            filename = f'sasfr_selected_steps_rank{rank}_pid{pid}.jsonl'
        path = os.path.join(dump_dir, filename)
        with open(path, 'a', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    except Exception as exc:
        print(f'Warning: failed to dump SASFR selected steps to {dump_dir}: {exc}')


def _metric_mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _metric_std(values):
    values = list(values)
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)


def _emit_ab1_metrics(metrics):
    """Print one structured AB1 metrics line for train.log parsers."""
    try:
        print('AB1_METRIC_JSON: ' + json.dumps(metrics, sort_keys=True))
    except Exception as exc:
        print(f'Warning: failed to emit AB1_METRIC_JSON: {exc}')


def _apply_sasfr_failure_reuse(
    total_batch_list,
    raw_state_list,
    raw_action_list,
    first_last_indices,
    value_dict_list,
    state_to_idx_list,
    config,
):
    """Run SASFR-Mini per UID group and attach selected-step metadata/rewards."""
    from rewardflow.failure_reuse import mine_sasfr_steps

    if _as_bool(_config_get(config, 'failure_reuse.allow_pseudo_success', False)):
        print('Warning: SASFR-Mini ignores allow_pseudo_success=true and still requires true success anchors.')

    _initialize_failure_reuse_defaults(total_batch_list)

    sasfr_config = _build_sasfr_config(config)
    dump_enabled = _as_bool(_config_get(config, 'failure_reuse.dump_selected_steps', False))
    dump_dir = str(_config_get(config, 'failure_reuse.dump_dir', 'outputs/failure_reuse'))
    metadata_only = _as_bool(_config_get(config, 'failure_reuse.sasfr.metadata_only', True))
    selection_mode = str(_config_get(config, 'failure_reuse.sasfr.selection_mode', 'prefix_only'))
    reward_application = str(_config_get(config, 'failure_reuse.sasfr.reward_application', 'max'))

    batch_metrics = defaultdict(float)
    selected_rows = []
    selected_scores = []
    selected_partials = []
    base_rewards = []
    reward_deltas = []
    positive_base_count = 0
    reward_applied_steps = 0
    reward_applied_sum = 0.0

    for group_idx, (start_idx, end_idx) in enumerate(first_last_indices):
        if start_idx >= end_idx or not total_batch_list[start_idx]:
            continue
        uid = str(total_batch_list[start_idx][0].get('uid', group_idx))
        selected_steps, group_metrics = mine_sasfr_steps(
            uid=uid,
            trajectory_group=total_batch_list[start_idx:end_idx],
            raw_state_group=raw_state_list[start_idx:end_idx],
            raw_action_group=raw_action_list[start_idx:end_idx],
            value_dict=value_dict_list[group_idx] if group_idx < len(value_dict_list) else None,
            state_to_idx=state_to_idx_list[group_idx] if group_idx < len(state_to_idx_list) else None,
            config=sasfr_config,
        )
        for key, value in group_metrics.items():
            if key.endswith('_mean') or key.endswith('_std') or key.endswith('_max') or key == 'sasfr/selected_ratio':
                continue
            batch_metrics[key] += float(value)
        if not selected_steps:
            continue

        selected_by_key = {(step.source_traj_uid, step.step_index): step for step in selected_steps}
        for local_traj_idx, trajectory in enumerate(total_batch_list[start_idx:end_idx]):
            source_uid = _sasfr_source_traj_uid(trajectory, uid, local_traj_idx)
            for step_index, step in enumerate(trajectory):
                selected = selected_by_key.get((source_uid, step_index))
                if selected is None:
                    continue
                apply_info = _apply_selected_failure_step(step, selected, config)
                row = _selected_step_dump_row(selected, apply_info, config)
                selected_rows.append(row)
                selected_scores.append(float(selected.reuse_score))
                selected_partials.append(float(selected.partial_reward))
                base_reward = float(apply_info['base_step_reward'])
                delta = float(apply_info['reward_delta'])
                base_rewards.append(base_reward)
                reward_deltas.append(delta)
                positive_base_count += int(base_reward > 0.0)
                reward_applied_steps += int(apply_info['reward_applied'])
                reward_applied_sum += delta if apply_info['reward_applied'] else 0.0

    candidate_steps = batch_metrics['sasfr/failed_candidate_steps']
    selected_count = len(selected_rows)
    metrics = {
        'event': 'sasfr_batch_metrics',
        'selection_mode': selection_mode,
        'sasfr/groups_total': batch_metrics['sasfr/groups_total'],
        'sasfr/groups_with_success': batch_metrics['sasfr/groups_with_success'],
        'sasfr/groups_with_anchorless_candidates': batch_metrics['sasfr/groups_with_anchorless_candidates'],
        'sasfr/groups_skipped_no_success': batch_metrics['sasfr/groups_skipped_no_success'],
        'sasfr/success_trajs': batch_metrics['sasfr/success_trajs'],
        'sasfr/failed_trajs': batch_metrics['sasfr/failed_trajs'],
        'sasfr/failed_candidate_steps': candidate_steps,
        'sasfr/selected_steps': float(selected_count),
        'sasfr/selected_ratio': float(selected_count / candidate_steps) if candidate_steps else 0.0,
        'sasfr/reuse_score_mean': _metric_mean(selected_scores),
        'sasfr/reuse_score_std': _metric_std(selected_scores),
        'sasfr/partial_reward_mean': _metric_mean(selected_partials),
        'sasfr/partial_reward_max': max(selected_partials) if selected_partials else 0.0,
        'sasfr/reward_applied_steps': float(reward_applied_steps),
        'sasfr/reward_applied_sum': float(reward_applied_sum),
        'sasfr/base_step_reward_mean_selected': _metric_mean(base_rewards),
        'sasfr/base_step_reward_positive_rate_selected': float(positive_base_count / selected_count) if selected_count else 0.0,
        'sasfr/reward_delta_mean_selected': _metric_mean(reward_deltas),
        'sasfr/bad_step_rejected': batch_metrics['sasfr/bad_step_rejected'],
        'sasfr/metadata_only': metadata_only,
        'sasfr/min_score': sasfr_config.min_score,
        'sasfr/max_failure_ratio': sasfr_config.max_failure_ratio,
        'sasfr/partial_reward_scale': sasfr_config.partial_reward_scale,
        'sasfr/reward_application': reward_application,
    }
    _emit_ab1_metrics(metrics)
    if (not metadata_only) and selected_count > 0 and reward_applied_steps == 0:
        print('Warning: SASFR selected steps but reward_applied_steps=0; reward_application=max may be saturated or hook is ineffective.')
    if dump_enabled:
        _dump_sasfr_selected_steps(selected_rows, dump_dir)

def apply_rewardflow_propagation(total_batch_list, 
                                 config,
                                 tokenizer,
                                 envs,
                                 return_metrics=False):
    """
    Main function to compute RewardFlow propagation for multi-turn trajectories.
    
    Args:
        total_batch_list: List of trajectory batches
        config: Configuration object with algorithm settings
        tokenizer: Tokenizer for decoding responses
        envs: Environment object with projection function
        
    Returns:
        total_batch_list: Updated batch list with propagated rewards
    """
    reward_method = _get_reward_method(config)
    if reward_method != "rewardflow":
        from reward_methods.mixer import SUPPORTED_REWARD_METHODS, compute_rewards_for_batch

        if reward_method not in SUPPORTED_REWARD_METHODS:
            raise ValueError(
                f"Unsupported reward_method={reward_method}. "
                "Use rewardflow, outcome_only, dcsr, fcsr_lite, or webshop_iprr_lite."
            )
        decode_errors = _decode_fcsr_step_outputs(total_batch_list, tokenizer)
        total_batch_list, _metrics = compute_rewards_for_batch(total_batch_list, config)
        _metrics["fcsr/decode_error_count"] = float(decode_errors)
        print("length of total_batch_list: ", len(total_batch_list))
        if return_metrics:
            return total_batch_list, _metrics
        return total_batch_list

    from agent_system.multi_turn_rollout.utils import (
        extract_unique_states, 
        build_trajectory, 
        get_personalization,
        unique_trajectory
    )
    from rewardflow.propagation import (
        propagate_reward_decay
    )


    raw_state_list, raw_action_list = envs.state_preprocess(total_batch_list, tokenizer)
    
    # Group trajectories by UID
    uid_list = []
    for i in range(len(total_batch_list)):
        for j in range(len(total_batch_list[0])):
            uid_list.append(total_batch_list[i][j]['uid'])
            break
    
    # Find first and last indices for each group of identical elements
    first_last_indices = []
    if uid_list:
        prev = uid_list[0]
        start = 0
        for idx in range(1, len(uid_list)):
            if uid_list[idx] != prev:
                first_last_indices.append((start, idx))
                start = idx
                prev = uid_list[idx]
        first_last_indices.append((start, len(uid_list)))

    # Process each group for reward propagation
    value_dict_list = []
    idx_to_state_list = []
    state_to_idx_list = []
    flat_unique_trajectory_list = []
    for (start_idx, end_idx) in first_last_indices:
        selected_state_list = raw_state_list[start_idx:end_idx]
        selected_action_list = raw_action_list[start_idx:end_idx]
        _, state_to_idx, idx_to_state = extract_unique_states(selected_state_list)
        trajectory = build_trajectory(selected_state_list, selected_action_list, state_to_idx)
        cleaned_trajectory = envs.clean_trajectory(trajectory, idx_to_state)
        flat_unique_trajectory = unique_trajectory(cleaned_trajectory)
        flat_unique_trajectory_list.append(flat_unique_trajectory)

        G_propagate, value_dict = propagate_reward_decay(
            flat_unique_trajectory, 
            gamma=config.algorithm.gamma, 
            max_iter=1000
        )
        
        for idx in range(len(idx_to_state)):
            if idx not in value_dict:
                value_dict[idx] = 0.0

        value_dict_list.append(value_dict)
        idx_to_state_list.append(idx_to_state)
        state_to_idx_list.append(state_to_idx)
    new_reward_list = []
    new_reward_abs_list = []
    for i in range(len(total_batch_list)):
        batch_idx = i * len(state_to_idx_list) // len(total_batch_list)
        initial_state = to_hashable(total_batch_list[i][0]['anchor_obs'])
        new_reward = []
        new_reward_abs = []
        
        try:
            state_idx = state_to_idx_list[batch_idx][initial_state]
        except KeyError as e:
            print(f"KeyError: initial_state {initial_state} not found in state_to_idx_list[{batch_idx}]")
            print(f"Available keys: {list(state_to_idx_list[batch_idx].keys())}")
            raise e
        try:
            prev_reward = value_dict_list[batch_idx][state_idx]
        except KeyError as e:
            print(f"KeyError: state_idx {state_idx} not found in value_dict_list[{batch_idx}]")
            print(f"Available keys: {list(value_dict_list[batch_idx].keys())}")
            raise e
        
        for j in range(len(total_batch_list[i])-1):
            if not total_batch_list[i][j]['active_masks']:
                break
            cur_state = to_hashable(total_batch_list[i][j+1]['anchor_obs'])
            state_idx = state_to_idx_list[batch_idx][cur_state]
            cur_reward = value_dict_list[batch_idx][state_idx]
            
            new_reward.append(cur_reward - prev_reward)
            new_reward_abs.append(cur_reward)
            prev_reward = cur_reward
        new_reward_abs_list.append(new_reward_abs)
        new_reward_list.append(new_reward)

    for i in range(len(total_batch_list)):
        for j in range(len(total_batch_list[i])):
            if j < len(new_reward_list[i]):
                total_batch_list[i][j]['step_rewards'] = new_reward_list[i][j]
                total_batch_list[i][j]['step_rewards_abs'] = new_reward_abs_list[i][j]
            else:
                total_batch_list[i][j]['step_rewards'] = total_batch_list[i][j]['rewards'] 
                total_batch_list[i][j]['step_rewards_abs'] = total_batch_list[i][j]['rewards'] 
    if _sasfr_enabled(config):
        _apply_sasfr_failure_reuse(
            total_batch_list,
            raw_state_list,
            raw_action_list,
            first_last_indices,
            value_dict_list,
            state_to_idx_list,
            config,
        )

    print("length of total_batch_list: ", len(total_batch_list))
    if return_metrics:
        return total_batch_list, {}
    return total_batch_list
