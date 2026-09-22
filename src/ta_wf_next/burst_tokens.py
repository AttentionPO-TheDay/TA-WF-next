"""Stored-order direction runs, not TLS records or inferred web resources.

No learned vocabulary, timestamps, labels, domain routing or model dependencies.
The exact and coarse views are deliberately separate: coarse tokens do not
silently retain exact run lengths through a feature bypass.
"""
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Iterable


@dataclass(frozen=True)
class Run:
    direction: int
    count: int
    touches_left_boundary: bool
    touches_right_boundary: bool


@dataclass(frozen=True)
class CoarseToken:
    direction: int
    log2_count_bin: int
    previous_bin_delta: int | None
    next_bin_delta: int | None
    touches_left_boundary: bool
    touches_right_boundary: bool


@dataclass(frozen=True)
class Encoding:
    runs: tuple[Run, ...]
    observed_count: int
    budget: int
    end_reason: str

    def decode(self) -> tuple[int, ...]:
        return tuple(d for r in self.runs for d in [r.direction] * r.count)

    def coarse(self) -> tuple[CoarseToken, ...]:
        """Bin k means [2**k, 2**(k+1)-1]; neighbours use bins only.

        Export only this return value for a coarse-only model. Encoding metadata
        and exact runs are audit information, not additional model channels.
        """
        bins = [r.count.bit_length()-1 for r in self.runs]
        return tuple(CoarseToken(r.direction, bins[i],
            bins[i]-bins[i-1] if i else None,
            bins[i+1]-bins[i] if i+1 < len(bins) else None,
            r.touches_left_boundary, r.touches_right_boundary)
            for i, r in enumerate(self.runs))


def encode_directions(values: Iterable[float], *, budget: int = 5000,
                      input_kind: str = 'direction') -> Encoding:
    """Consume at most budget entries, preserving order; zero is padding.

    input_kind='signed_timestamp' uses ONLY sign, including for nonmonotone
    timestamps. 'direction' accepts only -1,0,+1. Nonfinite values and internal
    zeros inside the observed prefix are rejected. Unobserved suffix is never
    inspected. Boundary flags indicate possible censoring, not a known true
    start/end of a network burst. No post-budget lookahead is used to decide it.
    """
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget <= 0:
        raise ValueError('budget must be a positive integer')
    if input_kind not in ('direction', 'signed_timestamp'):
        raise ValueError('unsupported input_kind')
    iterator = iter(values)
    directions = []
    padding = False
    exhausted = False
    for _ in range(budget):
        try:
            value = float(next(iterator))
        except StopIteration:
            exhausted = True
            break
        if not math.isfinite(value):
            raise ValueError('nonfinite observation')
        if input_kind == 'direction' and value not in (-1, 0, 1):
            raise ValueError('direction input must contain only -1,0,+1')
        if value == 0:
            padding = True
            continue
        if padding:
            raise ValueError('nonzero observation after padding inside budget')
        directions.append(1 if value > 0 else -1)
    pairs = []
    for d in directions:
        if pairs and pairs[-1][0] == d:
            pairs[-1][1] += 1
        else:
            pairs.append([d, 1])
    runs = tuple(Run(d, n, i == 0, i == len(pairs)-1)
                 for i, (d, n) in enumerate(pairs))
    reason = 'padding' if padding else 'input_end' if exhausted else 'budget'
    return Encoding(runs, len(directions), int(budget), reason)
