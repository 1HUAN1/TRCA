import math
from types import SimpleNamespace

import pytest
import torch

from reward_methods.kl_monitor import (
    compute_kl_estimators_from_delta,
    compute_kl_monitor_metrics,
    compute_old_policy_kl_from_logratio,
    masked_mean,
)


def test_ref_kl_estimators_zero_delta():
    delta = torch.zeros(2, 3)
    mask = torch.ones(2, 3)
    metrics = compute_kl_estimators_from_delta(delta, mask)
    assert metrics["kl_ref/k1_mean"] == pytest.approx(0.0)
    assert metrics["kl_ref/k2_mean"] == pytest.approx(0.0)
    assert metrics["kl_ref/k3_mean"] == pytest.approx(0.0)


def test_ref_kl_estimators_positive_and_negative_delta():
    delta = torch.tensor([[0.1, -0.1]])
    mask = torch.ones_like(delta)
    metrics = compute_kl_estimators_from_delta(delta, mask)
    expected_k3 = ((math.exp(-0.1) - 1.0 + 0.1) + (math.exp(0.1) - 1.0 - 0.1)) / 2.0
    assert metrics["kl_ref/k1_mean"] == pytest.approx(0.0)
    assert metrics["kl_ref/k2_mean"] == pytest.approx(0.005)
    assert metrics["kl_ref/k3_mean"] == pytest.approx(expected_k3, rel=1e-5, abs=1e-7)
    assert metrics["kl_ref/k3_negative_rate"] == pytest.approx(0.0)


def test_masked_mean_ignores_prompt_or_padding_tokens():
    x = torch.tensor([[100.0, 1.0, 3.0, 100.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    assert masked_mean(x, mask).item() == pytest.approx(2.0)


def test_compute_kl_monitor_metrics_uses_response_mask_and_ref_log_prob():
    current = torch.tensor([[5.0, 0.2, -0.2, 9.0]])
    ref = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    batch = SimpleNamespace(
        batch={
            "old_log_probs": current,
            "ref_log_prob": ref,
            "response_mask": mask,
        },
        non_tensor_batch={},
    )
    metrics = compute_kl_monitor_metrics(batch)
    expected = ((math.exp(-0.2) - 1.0 + 0.2) + (math.exp(0.2) - 1.0 - 0.2)) / 2.0
    assert metrics["kl_monitor/ref_logprob_available"] == 1.0
    assert metrics["kl_ref/k1_mean"] == pytest.approx(0.0)
    assert metrics["kl_ref/k3_mean"] == pytest.approx(expected)


def test_old_policy_kl_formula_orientation():
    logratio = torch.tensor([[0.1, -0.1]])
    mask = torch.ones_like(logratio)
    metrics = compute_old_policy_kl_from_logratio(logratio, mask)
    expected_k3 = ((math.exp(0.1) - 1.0 - 0.1) + (math.exp(-0.1) - 1.0 + 0.1)) / 2.0
    assert metrics["kl_old/k1_mean"] == pytest.approx(0.0)
    assert metrics["kl_old/k2_mean"] == pytest.approx(0.005)
    assert metrics["kl_old/k3_mean"] == pytest.approx(expected_k3, rel=1e-5, abs=1e-7)
