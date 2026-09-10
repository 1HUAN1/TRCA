"""Apply the configured transition reward to collected rollout steps."""

from __future__ import annotations

from typing import Any

from reward_methods.mixer import compute_rewards_for_batch


def decode_rollout_text(total_batch_list: list[list[dict[str, Any]]], tokenizer: Any) -> int:
    """Populate text fields needed by reward evaluators from rollout tensors."""
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
                    prompt = step["prompts"]
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
            if step.get("output") or step.get("responses") is None:
                continue
            response = step["responses"]
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


def apply_configured_rewards(
    total_batch_list: list[list[dict[str, Any]]], config: Any, tokenizer: Any
) -> tuple[list[list[dict[str, Any]]], dict[str, float | str]]:
    """Decode rollout text, apply the selected reward, and return its metrics."""
    decode_errors = decode_rollout_text(total_batch_list, tokenizer)
    total_batch_list, metrics = compute_rewards_for_batch(total_batch_list, config)
    metrics["fcsr/decode_error_count"] = float(decode_errors)
    return total_batch_list, metrics
