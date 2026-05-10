"""Dependency-free CTC prefix beam search for Paper15."""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from .charset import BLANK_INDEX, Charset

NEG_INF = -1.0e30


def _logadd(*values: float) -> float:
    valid = [value for value in values if value > NEG_INF / 2]
    if not valid:
        return NEG_INF
    maximum = max(valid)
    return maximum + math.log(sum(math.exp(value - maximum) for value in valid))


def ctc_prefix_beam_decode(log_probs: torch.Tensor, charset: Charset, beam_width: int = 10, topk: int = 16) -> str:
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}
    topk = max(1, min(topk, log_probs.size(1)))
    for frame in log_probs:
        values, indices = torch.topk(frame, k=topk)
        next_beams: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])
        for prefix, (p_blank, p_nonblank) in beams.items():
            for logp, index_value in zip(values.tolist(), indices.tolist()):
                index = int(index_value)
                if index == BLANK_INDEX:
                    next_beams[prefix][0] = _logadd(next_beams[prefix][0], p_blank + logp, p_nonblank + logp)
                    continue
                last = prefix[-1] if prefix else None
                if index == last:
                    next_beams[prefix][1] = _logadd(next_beams[prefix][1], p_nonblank + logp)
                    extended = prefix + (index,)
                    next_beams[extended][1] = _logadd(next_beams[extended][1], p_blank + logp)
                else:
                    extended = prefix + (index,)
                    next_beams[extended][1] = _logadd(next_beams[extended][1], p_blank + logp, p_nonblank + logp)
        ranked = sorted(next_beams.items(), key=lambda item: _logadd(item[1][0], item[1][1]), reverse=True)
        beams = {prefix: (scores[0], scores[1]) for prefix, scores in ranked[:beam_width]}
    best = max(beams.items(), key=lambda item: _logadd(item[1][0], item[1][1]))[0]
    return " ".join("".join(charset.index_to_char[index] for index in best if index in charset.index_to_char).split())

