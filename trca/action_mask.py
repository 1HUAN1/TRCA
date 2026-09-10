"""Token masks that credit only executable action/search/answer content."""

from __future__ import annotations

import re
from typing import Any, Iterable

import torch


_ACTION_BLOCK = re.compile(
    r"<(?P<tag>action|search|answer)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return str(tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))


def _offsets(tokenizer: Any, text: str, token_ids: list[int]) -> list[tuple[int, int]]:
    """Retokenize decoded text for offsets; fall back to per-token decoding."""
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        candidate = [int(item) for item in encoded["input_ids"]]
        offsets = [tuple(map(int, item)) for item in encoded["offset_mapping"]]
        if candidate == token_ids and len(offsets) == len(token_ids):
            return offsets
    except (KeyError, TypeError, ValueError, NotImplementedError):
        pass

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        piece = _decode(tokenizer, [token_id])
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return offsets


def action_token_mask(token_ids: Iterable[int], tokenizer: Any) -> list[int]:
    """Mask complete executable spans; malformed or reasoning spans receive no credit."""
    ids = [int(item) for item in token_ids]
    mask = [0] * len(ids)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    active_length = len(ids)
    if pad_token_id is not None:
        while active_length and ids[active_length - 1] == int(pad_token_id):
            active_length -= 1
    active_ids = ids[:active_length]
    if not active_ids:
        return mask

    text = _decode(tokenizer, active_ids)
    think_ranges = [match.span() for match in _THINK_BLOCK.finditer(text)]
    content_ranges = []
    for match in _ACTION_BLOCK.finditer(text):
        block_start, block_end = match.span()
        if any(block_start < think_end and block_end > think_start for think_start, think_end in think_ranges):
            continue
        content_ranges.append(match.span("body"))

    for index, (start, end) in enumerate(_offsets(tokenizer, text, active_ids)):
        if end > start and any(start < right and end > left for left, right in content_ranges):
            mask[index] = 1
    return mask


def build_batch_action_token_mask(responses: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    rows = [action_token_mask(row.detach().cpu().tolist(), tokenizer) for row in responses]
    return torch.tensor(rows, dtype=torch.long, device=responses.device)
