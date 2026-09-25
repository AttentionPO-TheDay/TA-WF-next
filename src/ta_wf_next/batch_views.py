"""CPU-safe batch adapters for the explicit traffic views.

The adapters keep observation and token budgets separate. Padding is always
paired with a boolean mask; no mask is silently converted into a learned zero
value. Coarse runs expose only binned run lengths and boundary metadata.
"""
from dataclasses import dataclass
from typing import Sequence

import torch

from .traffic_views import TrafficViews, generate_views


@dataclass(frozen=True)
class BatchView:
    values: torch.Tensor
    mask: torch.Tensor
    token_count: torch.Tensor
    source_observed_count: torch.Tensor
    truncated: torch.Tensor


def _stack(rows: list[list[list[float]]], width: int, channels: int,
           source_counts: Sequence[int]) -> BatchView:
    if not rows:
        raise ValueError('empty batch')
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError('width must be a positive integer')
    b = len(rows)
    out = torch.zeros((b, width, channels), dtype=torch.float32)
    mask = torch.zeros((b, width), dtype=torch.bool)
    counts = torch.zeros((b,), dtype=torch.int64)
    truncated = torch.zeros((b,), dtype=torch.bool)
    for i, row in enumerate(rows):
        n = min(len(row), width)
        if n:
            out[i, :n] = torch.tensor(row[:n], dtype=torch.float32)
        mask[i, :n] = True
        counts[i] = len(row)
        truncated[i] = len(row) > width
    return BatchView(out, mask, counts, torch.tensor(source_counts, dtype=torch.int64), truncated)


def packet_batch(views: Sequence[TrafficViews], *, width: int = 5000) -> BatchView:
    rows = [[[float(d)] for d in v.packet_direction] for v in views]
    return _stack(rows, width, 1, [v.runs.observed_count for v in views])


def exact_run_batch(views: Sequence[TrafficViews], *, width: int = 256) -> BatchView:
    rows = [[[float(r.direction), float(r.count), float(r.touches_left_boundary),
              float(r.touches_right_boundary)] for r in v.runs.runs] for v in views]
    return _stack(rows, width, 4, [v.runs.observed_count for v in views])


def coarse_run_batch(views: Sequence[TrafficViews], *, width: int = 256) -> BatchView:
    rows = []
    for v in views:
        tokens = v.runs.coarse()
        rows.append([[float(t.direction), float(t.log2_count_bin),
                      float(t.previous_bin_delta or 0), float(t.previous_bin_delta is not None),
                      float(t.next_bin_delta or 0), float(t.next_bin_delta is not None),
                      float(t.touches_left_boundary), float(t.touches_right_boundary)]
                     for t in tokens])
    return _stack(rows, width, 8, [v.runs.observed_count for v in views])


def window_batch(views: Sequence[TrafficViews], *, size: int = 250, width: int = 32) -> BatchView:
    selected = [dict(v.direction_windows)[size] for v in views]
    rows = [[[float(w.positive_fraction), float(w.transition_fraction),
              float(w.observed_count), float(w.partial)] for w in ws] for ws in selected]
    return _stack(rows, width, 4, [v.runs.observed_count for v in views])


def views_from_rows(rows: Sequence[Sequence[float]], *, input_kind: str = 'direction',
                    observation_budget: int = 5000,
                    window_sizes: tuple[int, ...] = (50, 250)) -> tuple[TrafficViews, ...]:
    """Build views with one explicit observation budget and no label input."""
    return tuple(generate_views(row, input_kind=input_kind,
                                budget=observation_budget,
                                window_sizes=window_sizes) for row in rows)
