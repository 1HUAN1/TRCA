"""Detached KL estimator monitoring utilities.

These helpers only compute logging metrics. They must not be used to alter
training loss, rewards, or advantages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.detach().float()
    mask = mask.detach().float()
    return (x * mask).sum() / (mask.sum() + eps)


def masked_std(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = masked_mean(x, mask, eps=eps)
    var = masked_mean((x.detach().float() - mean).pow(2), mask, eps=eps)
    return torch.sqrt(torch.clamp(var, min=0.0))


def masked_min(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = x.detach().float()[mask.detach().bool()]
    if valid.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    return valid.min()


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = x.detach().float()[mask.detach().bool()]
    if valid.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    return valid.max()


def _sequence_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.detach().float()
    mask = mask.detach().float()
    return (x * mask).sum(dim=-1) / (mask.sum(dim=-1) + eps)


def _masked_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    values = values.detach().float()
    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    return torch.quantile(values, q)


def _basic_stats(k1: torch.Tensor, k2: torch.Tensor, k3: torch.Tensor, mask: torch.Tensor, prefix: str, ratio: torch.Tensor) -> Dict[str, float]:
    return {
        f"{prefix}/k1_mean": _to_float(masked_mean(k1, mask)),
        f"{prefix}/k2_mean": _to_float(masked_mean(k2, mask)),
        f"{prefix}/k3_mean": _to_float(masked_mean(k3, mask)),
        f"{prefix}/k1_std": _to_float(masked_std(k1, mask)),
        f"{prefix}/k2_std": _to_float(masked_std(k2, mask)),
        f"{prefix}/k3_std": _to_float(masked_std(k3, mask)),
        f"{prefix}/k1_min": _to_float(masked_min(k1, mask)),
        f"{prefix}/k1_max": _to_float(masked_max(k1, mask)),
        f"{prefix}/k3_min": _to_float(masked_min(k3, mask)),
        f"{prefix}/k3_max": _to_float(masked_max(k3, mask)),
        f"{prefix}/k1_negative_rate": _to_float(masked_mean((k1 < 0).float(), mask)),
        f"{prefix}/k3_negative_rate": _to_float(masked_mean((k3 < -1e-6).float(), mask)),
        f"{prefix}/delta_abs_mean": _to_float(masked_mean(k1.abs(), mask)),
        f"{prefix}/ratio_ref_current_mean": _to_float(masked_mean(ratio, mask)),
    }


def _sequence_stats(k1: torch.Tensor, k2: torch.Tensor, k3: torch.Tensor, mask: torch.Tensor, prefix: str) -> Dict[str, float]:
    token_count = mask.detach().float().sum(dim=-1)
    valid_seq = token_count > 0
    seq_k1 = _sequence_mean(k1, mask)[valid_seq]
    seq_k2 = _sequence_mean(k2, mask)[valid_seq]
    seq_k3 = _sequence_mean(k3, mask)[valid_seq]
    if seq_k3.numel() == 0:
        zero = 0.0
        return {
            f"{prefix}/seq_k1_mean": zero,
            f"{prefix}/seq_k2_mean": zero,
            f"{prefix}/seq_k3_mean": zero,
            f"{prefix}/seq_k3_p50": zero,
            f"{prefix}/seq_k3_p90": zero,
            f"{prefix}/seq_k3_p95": zero,
            f"{prefix}/seq_k3_p99": zero,
        }
    return {
        f"{prefix}/seq_k1_mean": _to_float(seq_k1.mean()),
        f"{prefix}/seq_k2_mean": _to_float(seq_k2.mean()),
        f"{prefix}/seq_k3_mean": _to_float(seq_k3.mean()),
        f"{prefix}/seq_k3_p50": _to_float(_masked_quantile(seq_k3, 0.50)),
        f"{prefix}/seq_k3_p90": _to_float(_masked_quantile(seq_k3, 0.90)),
        f"{prefix}/seq_k3_p95": _to_float(_masked_quantile(seq_k3, 0.95)),
        f"{prefix}/seq_k3_p99": _to_float(_masked_quantile(seq_k3, 0.99)),
    }


def compute_kl_estimators_from_delta(
    delta: torch.Tensor,
    mask: torch.Tensor,
    prefix: str = "kl_ref",
    clip_exp_max: float = 1e6,
    log_sequence_stats: bool = True,
) -> Dict[str, float]:
    """Compute Schulman k1/k2/k3 for KL(current || reference).

    delta is logp_current - logp_ref, shaped [batch, response_len].
    """
    with torch.no_grad():
        delta = delta.detach().float()
        mask = mask.detach().float()
        ratio = torch.exp(torch.clamp(-delta, max=float(np.log(clip_exp_max)))).clamp(max=clip_exp_max)
        k1 = delta
        k2 = 0.5 * delta.pow(2)
        k3 = ratio - 1.0 + delta
        metrics = _basic_stats(k1, k2, k3, mask, prefix, ratio)
        if log_sequence_stats:
            metrics.update(_sequence_stats(k1, k2, k3, mask, prefix))
        unclipped_ratio = torch.exp(torch.clamp(-delta, max=80.0))
        metrics[f"{prefix}/ratio_ref_current_clip_rate"] = _to_float(masked_mean((unclipped_ratio > clip_exp_max).float(), mask))
        return metrics


def compute_old_policy_kl_from_logratio(
    logratio: torch.Tensor,
    mask: torch.Tensor,
    prefix: str = "kl_old",
    clip_exp_max: float = 1e6,
    log_sequence_stats: bool = True,
) -> Dict[str, float]:
    """Compute estimators for KL(old || current) on old-policy samples.

    logratio is logp_current - logp_old, shaped [batch, response_len].
    """
    with torch.no_grad():
        logratio = logratio.detach().float()
        mask = mask.detach().float()
        ratio = torch.exp(torch.clamp(logratio, max=float(np.log(clip_exp_max)))).clamp(max=clip_exp_max)
        k1 = -logratio
        k2 = 0.5 * logratio.pow(2)
        k3 = ratio - 1.0 - logratio
        metrics = _basic_stats(k1, k2, k3, mask, prefix, ratio)
        if log_sequence_stats:
            metrics.update(_sequence_stats(k1, k2, k3, mask, prefix))
        unclipped_ratio = torch.exp(torch.clamp(logratio, max=80.0))
        metrics[f"{prefix}/ratio_current_old_clip_rate"] = _to_float(masked_mean((unclipped_ratio > clip_exp_max).float(), mask))
        return metrics


def _success_labels_from_batch(batch: Any, device: torch.device) -> Optional[torch.Tensor]:
    non_tensor = getattr(batch, "non_tensor_batch", {}) or {}
    rewards = non_tensor.get("episode_rewards")
    if rewards is None:
        return None
    try:
        labels = torch.tensor(np.asarray(rewards, dtype=np.float32), device=device)
        return labels >= 0.999
    except Exception:
        return None


def _add_success_failure_split(metrics: Dict[str, float], k3: torch.Tensor, mask: torch.Tensor, batch: Any, prefix: str) -> None:
    labels = _success_labels_from_batch(batch, k3.device)
    if labels is None or labels.numel() != k3.shape[0]:
        return
    token_count = mask.detach().float().sum(dim=-1)
    valid_seq = token_count > 0
    seq_k3 = _sequence_mean(k3, mask)
    success = labels & valid_seq
    failure = (~labels) & valid_seq
    if success.any():
        metrics[f"{prefix}/success_seq_k3_mean"] = _to_float(seq_k3[success].mean())
    if failure.any():
        metrics[f"{prefix}/failure_seq_k3_mean"] = _to_float(seq_k3[failure].mean())


def compute_kl_monitor_metrics(
    batch: Any,
    clip_exp_max: float = 1e6,
    log_sequence_stats: bool = True,
    log_success_failure_split: bool = True,
) -> Dict[str, float]:
    """Compute detached KL monitor metrics from a VERL DataProto batch."""
    tensors = batch.batch
    mask = tensors.get("response_mask")
    if mask is None:
        responses = tensors.get("responses")
        attention_mask = tensors.get("attention_mask")
        if responses is not None and attention_mask is not None:
            mask = attention_mask[:, -responses.shape[-1]:]
    metrics: Dict[str, float] = {
        "kl_monitor/ref_logprob_available": float("ref_log_prob" in tensors),
        "kl_monitor/old_logprob_available": float("rollout_log_probs" in tensors),
    }
    if mask is None:
        metrics["kl_monitor/response_token_count_mean"] = 0.0
        metrics["kl_monitor/response_mask_nonzero_rate"] = 0.0
        metrics["kl_monitor/empty_response_mask_warning"] = 1.0
        return metrics

    mask = mask.detach().float()
    token_count = mask.sum(dim=-1)
    metrics["kl_monitor/response_token_count_mean"] = _to_float(token_count.float().mean())
    metrics["kl_monitor/response_mask_nonzero_rate"] = _to_float((token_count > 0).float().mean())
    metrics["kl_monitor/empty_response_mask_warning"] = float(metrics["kl_monitor/response_mask_nonzero_rate"] == 0.0)

    current_logp = tensors.get("old_log_probs")
    ref_logp = tensors.get("ref_log_prob")
    if current_logp is not None and ref_logp is not None:
        delta = current_logp.detach().float() - ref_logp.detach().float()
        ref_metrics = compute_kl_estimators_from_delta(
            delta=delta,
            mask=mask,
            prefix="kl_ref",
            clip_exp_max=clip_exp_max,
            log_sequence_stats=log_sequence_stats,
        )
        metrics.update(ref_metrics)
        if log_success_failure_split:
            ratio = torch.exp(torch.clamp(-delta, max=float(np.log(clip_exp_max)))).clamp(max=clip_exp_max)
            k3 = ratio - 1.0 + delta
            _add_success_failure_split(metrics, k3, mask, batch, "kl_ref")
        metrics["kl_monitor/k3_negative_warning"] = float(metrics.get("kl_ref/k3_negative_rate", 0.0) > 0.001)

    rollout_logp = tensors.get("rollout_log_probs")
    if current_logp is not None and rollout_logp is not None:
        old_metrics = compute_old_policy_kl_from_logratio(
            logratio=current_logp.detach().float() - rollout_logp.detach().float(),
            mask=mask,
            prefix="kl_old",
            clip_exp_max=clip_exp_max,
            log_sequence_stats=log_sequence_stats,
        )
        metrics.update(old_metrics)
    return metrics
