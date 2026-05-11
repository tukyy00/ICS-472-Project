"""CTC decoding helpers."""

from __future__ import annotations

import math
from collections import defaultdict

import torch

DIGITS = "0123456789"
BLANK_INDEX = 0
INDEX_TO_CHAR = {index + 1: char for index, char in enumerate(DIGITS)}
NEG_INF = -1.0e30


def decode_indices(indices: list[int], blank_index: int = BLANK_INDEX) -> str:
    """Greedy CTC collapse: remove repeats and blanks."""
    output: list[str] = []
    previous: int | None = None
    for index in indices:
        if index != previous and index != blank_index:
            char = INDEX_TO_CHAR.get(index)
            if char is not None:
                output.append(char)
        previous = index
    return "".join(output)


def encode_label(label: str) -> list[int]:
    return [DIGITS.index(char) + 1 for char in label if char in DIGITS]


def _logadd(*values: float) -> float:
    valid = [value for value in values if value > NEG_INF / 2]
    if not valid:
        return NEG_INF
    maximum = max(valid)
    return maximum + math.log(sum(math.exp(value - maximum) for value in valid))


def ctc_prefix_beam_decode(
    log_probs: torch.Tensor,
    beam_width: int = 10,
    topk: int = 11,
    blank_index: int = BLANK_INDEX,
) -> str:
    """Decode one CTC log-probability matrix shaped ``[time, classes]``."""
    if log_probs.ndim != 2:
        raise ValueError(f"Expected [time, classes] log_probs, got {tuple(log_probs.shape)}")

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}
    topk = max(1, min(int(topk), log_probs.size(1)))
    beam_width = max(1, int(beam_width))

    for frame in log_probs:
        values, indices = torch.topk(frame, k=topk)
        next_beams: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])

        for prefix, (p_blank, p_nonblank) in beams.items():
            for value, index_tensor in zip(values.tolist(), indices.tolist()):
                index = int(index_tensor)
                logp = float(value)
                if index == blank_index:
                    current = next_beams[prefix]
                    current[0] = _logadd(current[0], p_blank + logp, p_nonblank + logp)
                    continue

                last = prefix[-1] if prefix else None
                if index == last:
                    current = next_beams[prefix]
                    current[1] = _logadd(current[1], p_nonblank + logp)

                    extended = prefix + (index,)
                    current_extended = next_beams[extended]
                    current_extended[1] = _logadd(current_extended[1], p_blank + logp)
                else:
                    extended = prefix + (index,)
                    current_extended = next_beams[extended]
                    current_extended[1] = _logadd(current_extended[1], p_blank + logp, p_nonblank + logp)

        ranked = sorted(
            ((prefix, scores[0], scores[1]) for prefix, scores in next_beams.items()),
            key=lambda item: _logadd(item[1], item[2]),
            reverse=True,
        )
        beams = {prefix: (p_blank, p_nonblank) for prefix, p_blank, p_nonblank in ranked[:beam_width]}

    best_prefix = max(beams.items(), key=lambda item: _logadd(item[1][0], item[1][1]))[0]
    return "".join(INDEX_TO_CHAR[index] for index in best_prefix if index in INDEX_TO_CHAR)

