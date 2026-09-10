from types import SimpleNamespace

import numpy as np
import torch

from agent_system.multi_turn_rollout.utils import adjust_batch
from verl import DataProto


def _config():
    return SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=2, nnodes=1),
        algorithm=SimpleNamespace(use_kl_in_reward=False),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(log_prob_micro_batch_size_per_gpu=8),
            ref=SimpleNamespace(log_prob_micro_batch_size_per_gpu=8),
            actor=SimpleNamespace(
                ppo_mini_batch_size=4,
                ppo_micro_batch_size_per_gpu=1,
                use_kl_loss=True,
            ),
        ),
    )


def test_adjust_batch_can_pad_by_more_than_the_source_batch():
    data = DataProto.from_dict(
        tensors={"input_ids": torch.arange(7).unsqueeze(-1)},
        non_tensors={"uid": np.arange(7, dtype=object)},
    )

    adjusted = adjust_batch(_config(), data)

    assert len(adjusted) == 16
    assert adjusted.batch["input_ids"][:7].squeeze(-1).tolist() == list(range(7))
    assert set(adjusted.non_tensor_batch["uid"].tolist()) == set(range(7))
