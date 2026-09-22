"""Explicit, unlearned views of a single stored-order observation prefix.

Missing modalities are None, not fabricated zero channels. These are candidate
representations, not a claim of invariance, physical timing, or new information.
"""
from dataclasses import dataclass
from itertools import islice
from numbers import Integral
from typing import Iterable

from .burst_tokens import Encoding, encode_directions


@dataclass(frozen=True)
class DirectionWindow:
    start: int
    observed_count: int
    positive_fraction: float
    transition_fraction: float
    partial: bool


@dataclass(frozen=True)
class TimingView:
    # Position-aligned; first observation of each sign has no predecessor.
    same_direction_interval: tuple[float | None, ...]
    run_span: tuple[float | None, ...]
    run_zero_span: tuple[bool, ...]
    invalid_same_direction_intervals: int
    invalid_run_spans: int


@dataclass(frozen=True)
class SizeView:
    absolute_sizes: tuple[float, ...]
    run_totals: tuple[float, ...]


@dataclass(frozen=True)
class TrafficViews:
    packet_direction: tuple[int, ...]
    runs: Encoding
    direction_windows: tuple[tuple[int, tuple[DirectionWindow, ...]], ...]
    timing: TimingView | None
    size: SizeView | None

    def select(self, *names: str) -> dict:
        """Explicit export whitelist. Never implicitly exports audit metadata.

        Separate calls/views permit ablations. Exporting several views is an
        explicit information choice; e.g. packet_direction defeats coarse-only.
        """
        available = dict(packet_direction=self.packet_direction,
                         exact_runs=self.runs.runs, coarse_runs=self.runs.coarse(),
                         direction_windows=self.direction_windows,
                         timing=self.timing, size=self.size)
        if not names or len(names) != len(set(names)):
            raise ValueError('select distinct explicit view names')
        if any(name not in available for name in names):
            raise ValueError('unknown view')
        if any(available[name] is None for name in names):
            raise ValueError('requested modality missing or disabled')
        return {name: available[name] for name in names}


def generate_views(values: Iterable[float], *, input_kind: str,
                   budget: int = 5000, window_sizes: tuple[int, ...] = (50, 250),
                   enable_timing: bool = False) -> TrafficViews:
    """input_kind explicitly declares direction, signed_timestamp or signed_size.

    Zero denotes padding, not an observed packet. Timestamp units must be seconds
    by caller contract. Size units are caller-declared dataset units (not assumed
    Tor cells). At most budget raw positions are consumed. No labels, URLs,
    condition identities, sorting, imputation, rate or cross-sign IAT are used.
    """
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget <= 0:
        raise ValueError('budget must be positive integer')
    if input_kind not in ('direction', 'signed_timestamp', 'signed_size'):
        raise ValueError('explicit supported input_kind required')
    if enable_timing and input_kind != 'signed_timestamp':
        raise ValueError('timing requires real timestamp input')
    if any(isinstance(w, bool) or not isinstance(w, Integral) or w <= 0 for w in window_sizes):
        raise ValueError('window sizes must be positive integers')
    if len(set(window_sizes)) != len(window_sizes):
        raise ValueError('duplicate window size')
    prefix = tuple(float(v) for v in islice(values, int(budget)))
    # The signed mode here is ONLY a sign extractor; size is never interpreted as time.
    runs = encode_directions(prefix, budget=budget,
        input_kind='direction' if input_kind == 'direction' else 'signed_timestamp')
    direction = runs.decode()
    windows = []
    for width in window_sizes:
        entries = []
        for start in range(0, len(direction), width):
            part = direction[start:start+width]
            switches = sum(a != b for a,b in zip(part,part[1:]))
            entries.append(DirectionWindow(start,len(part),sum(d>0 for d in part)/len(part),
                switches/(len(part)-1) if len(part)>1 else 0.,len(part)<width))
        windows.append((int(width),tuple(entries)))
    magnitudes = tuple(abs(v) for v in prefix[:runs.observed_count])
    size = None
    if input_kind == 'signed_size':
        totals=[]; offset=0
        for run in runs.runs:
            totals.append(sum(magnitudes[offset:offset+run.count])); offset+=run.count
        size=SizeView(magnitudes,tuple(totals))
    timing = None
    if enable_timing:
        previous={}; intervals=[]; bad_intervals=0
        for d,t in zip(direction,magnitudes):
            delta=t-previous[d] if d in previous else None
            if delta is not None and delta<0:
                delta=None; bad_intervals+=1
            intervals.append(delta); previous[d]=t
        spans=[]; zero=[]; bad_spans=0; offset=0
        for run in runs.runs:
            times=magnitudes[offset:offset+run.count]; offset+=run.count
            # Check the entire run, not only endpoints.
            valid=all(a<=b for a,b in zip(times,times[1:]))
            span=times[-1]-times[0] if valid else None
            spans.append(span); zero.append(span==0)
            bad_spans+=int(not valid)
        timing=TimingView(tuple(intervals),tuple(spans),tuple(zero),bad_intervals,bad_spans)
    return TrafficViews(direction,runs,tuple(windows),timing,size)
