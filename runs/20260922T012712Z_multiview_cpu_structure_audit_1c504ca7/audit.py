from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/mnt/data2/ren/datasets')
OUT = Path(__file__).resolve().parent / 'artifacts' / 'structure.json'
PACKET_BUDGETS = (256, 512, 1000, 2500, 5000)
RUN_TOKEN_BUDGETS = (64, 128, 256, 512, 1024)
RUN_POSITION_PROBES = (32, 64, 128, 256, 512)
WINDOWS = (50, 250)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def summarize(x: np.ndarray) -> dict:
    if x.ndim != 2 or x.shape[1] < 5000:
        raise ValueError(f'expected [N,>=5000], got {x.shape}')
    a = np.asarray(x[:, :5000])
    if not np.isfinite(a).all():
        raise ValueError('nonfinite input')
    nonzero = a != 0
    lengths = nonzero.sum(1)
    # The protocol treats zero as suffix padding. Internal zero is retained as
    # a structural warning, not silently repaired.
    positions = np.arange(5000)[None, :]
    last = np.where(nonzero, positions + 1, 0).max(1)
    # With zero-only suffix padding, last nonzero position equals nonzero count.
    internal = last != lengths
    run_counts=[]; max_runs=[]; mean_runs=[]
    run_packet_coverage = {b: [] for b in RUN_TOKEN_BUDGETS}
    run_end_positions = {b: [] for b in RUN_POSITION_PROBES}
    for row, n in zip(a, lengths):
        s = np.sign(row[:int(n)]).astype(np.int8)
        if len(s) == 0:
            lens = np.array([], dtype=np.int64)
        else:
            cuts = np.flatnonzero(s[1:] != s[:-1]) + 1
            lens = np.diff(np.r_[0, cuts, len(s)])
        run_counts.append(len(lens)); max_runs.append(int(lens.max()) if len(lens) else 0)
        mean_runs.append(float(lens.mean()) if len(lens) else 0.)
        cumulative = np.cumsum(lens)
        for b in RUN_TOKEN_BUDGETS:
            retained = int(cumulative[min(b, len(lens))-1]) if len(lens) else 0
            run_packet_coverage[b].append(retained / int(n) if n else 1.)
        for b in RUN_POSITION_PROBES:
            if len(cumulative) >= b:
                run_end_positions[b].append(int(cumulative[b-1]))
    result = {
        'rows': int(len(a)),
        'sha256': None,
        'observed_length': {k: float(v) for k,v in zip(('min','p10','p50','p90','max'), np.quantile(lengths,[0,.1,.5,.9,1]))},
        'shorter_than_5000': int((lengths < 5000).sum()),
        'empty': int((lengths == 0).sum()),
        'rows_with_internal_zero': int(internal.sum()),
        'run_count': {k: float(v) for k,v in zip(('min','p10','p50','p90','max'), np.quantile(run_counts,[0,.1,.5,.9,1]))},
        'run_length_mean': {k: float(v) for k,v in zip(('min','p10','p50','p90','max'), np.quantile(mean_runs,[0,.1,.5,.9,1]))},
        'max_run_length': {k: float(v) for k,v in zip(('min','p50','p90','max'), np.quantile(max_runs,[0,.5,.9,1]))},
        'packet_budgets': {}, 'run_token_budgets': {}, 'run_position_probes': {}, 'windows': {}
    }
    for b in PACKET_BUDGETS:
        observed = np.minimum(lengths, b)
        result['packet_budgets'][str(b)] = {
            'rows_truncated_by_budget': int((lengths > b).sum()),
            'fraction_truncated': float((lengths > b).mean()),
            'median_observed': float(np.median(observed)),
        }
    run_counts_array = np.asarray(run_counts)
    for b in RUN_TOKEN_BUDGETS:
        coverage = np.asarray(run_packet_coverage[b])
        result['run_token_budgets'][str(b)] = {
            'rows_truncated': int((run_counts_array > b).sum()),
            'fraction_truncated': float((run_counts_array > b).mean()),
            'packet_coverage': {k: float(v) for k,v in zip(('min','p10','p50','p90','max'), np.quantile(coverage,[0,.1,.5,.9,1]))},
        }
    for b in RUN_POSITION_PROBES:
        ends = np.asarray(run_end_positions[b])
        result['run_position_probes'][str(b)] = {
            'eligible_rows': int(len(ends)),
            'eligible_fraction': float(len(ends) / len(a)),
            'raw_packet_end_position': ({k: float(v) for k,v in zip(('min','p10','p50','p90','max'), np.quantile(ends,[0,.1,.5,.9,1]))} if len(ends) else None),
        }
    for w in WINDOWS:
        nwin = np.ceil(lengths / w).astype(int)
        partial = (lengths > 0) & (lengths % w != 0)
        result['windows'][str(w)] = {
            'rows_with_partial_window': int(partial.sum()),
            'fraction_with_partial_window': float(partial.mean()),
            'total_windows': int(nwin.sum()),
        }
    return result

def load(name: str, path: Path) -> tuple[str, dict]:
    with np.load(path, allow_pickle=False) as z:
        if 'X' not in z:
            raise ValueError(f'{path} has no X')
        x = z['X']
        # Deliberately never access y.
        return name, summarize(x)

def main() -> None:
    files = [('NetworkDrift/train', DATA_ROOT/'NetworkDrift'/'train.npz'), ('NetworkDrift/valid', DATA_ROOT/'NetworkDrift'/'valid.npz'),
             ('NetworkDrift/JP', DATA_ROOT/'NetworkDrift'/'JP.npz'), ('BehaviorDrift/subpage', DATA_ROOT/'BehaviorDrift'/'subpage.npz')]
    out = {'schema_version': 2, 'x_only': True, 'observation_budget': 5000,
           'packet_budgets': PACKET_BUDGETS, 'run_token_budgets': RUN_TOKEN_BUDGETS,
           'run_position_probes': RUN_POSITION_PROBES, 'window_sizes': WINDOWS,
           'files': {}, 'source_file_sha256': {}}
    for name, path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        key, info = load(name, path)
        info['sha256'] = digest(path)
        out['files'][key] = info
        out['source_file_sha256'][key] = info['sha256']
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'output': str(OUT), 'files': {k:v['rows'] for k,v in out['files'].items()}}, indent=2))

if __name__ == '__main__':
    main()
