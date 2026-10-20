"""Explicit direction-only data and identity boundaries; shared files read-only."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
DS = json.loads((ROOT / 'configs/datasets.json').read_text())
DATA = Path(DS['data_root'])
CFG = json.loads((RUN / 'config.json').read_text())


def digest(row):
    # WideGenerator consumes the positional TAM. Isolate this effective
    # input as well as exact direction sequences, including pooling collisions.
    bins = np.arange(5000) * 1800 // 5000
    active = row != 0
    keys = bins[active] + (row[active] < 0) * 1800
    counts = np.bincount(keys, minlength=3600).astype('<i2')
    return hashlib.sha256(counts.tobytes()).hexdigest()


def direction(x):
    if not np.isfinite(x).all():
        raise ValueError('nonfinite input')
    output = np.zeros((len(x), 5000), dtype=np.int8)
    width = min(x.shape[1], 5000)
    output[:, :width] = np.sign(x[:, :width]).astype(np.int8)
    return output


def dedup(x, y, uids, excluded=()):
    hashes = [digest(row) for row in x]
    labels = defaultdict(set)
    for key, label in zip(hashes, y):
        labels[key].add(int(label))
    conflicts = {key for key, values in labels.items() if len(values) != 1}
    seen = set(excluded) | conflicts
    keep = []
    for i, key in enumerate(hashes):
        if key not in seen and np.any(x[i]):
            keep.append(i)
            seen.add(key)
    keep = np.asarray(keep, dtype=np.int64)
    return {'x': x[keep], 'y': y[keep], 'uids': uids[keep], 'hashes': [hashes[i] for i in keep]}, {
        'raw_rows': len(x), 'retained_rows': len(keep), 'conflicting_groups': len(conflicts)}


def select_source(raw, per_class):
    indices = []
    for label in sorted(set(raw['y'])):
        members = np.flatnonzero(raw['y'] == label)
        members = sorted(members, key=lambda i: raw['hashes'][i])
        if len(members) < per_class:
            raise ValueError(f'insufficient source class {label}')
        indices.extend(members[:per_class])
    return subset(raw, indices)


def subset(raw, indices):
    indices = np.asarray(indices, dtype=np.int64)
    return {key: ([value[i] for i in indices] if isinstance(value, list) else value[indices])
            for key, value in raw.items()}


def read_records(records):
    output = np.zeros((len(records), 5000), dtype=np.int8)
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[record['path']].append((i, record['row']))
    for path, rows in groups.items():
        with h5py.File(path, 'r') as handle:
            for start in range(0, len(rows), 512):
                batch = rows[start:start + 512]
                positions = sorted({r for _, r in batch})
                values = direction(np.asarray(handle['vector'][positions]))
                lookup = {position: j for j, position in enumerate(positions)}
                for i, position in batch:
                    output[i] = values[lookup[position]]
    return output, np.array([r['label'] for r in records], np.int64), np.array([r['uid'] for r in records], np.int64)


def wtt():
    root = DATA / DS['datasets']['wtt_time_processed']['path']
    meta = json.loads((root / 'label_map.json').read_text())
    mapping = {r['site_name']: r['global_class_id'] for r in meta['sites']}
    assert set(mapping.values()) == set(range(100))
    records = []
    with (root / 'sample_index.jsonl').open() as stream:
        for uid, line in enumerate(stream):
            row = json.loads(line)
            if row['condition_name'] not in ('dynamic_browser_version_network', 'static_browser_version_network'):
                continue
            relative = row['source_file'].split('shusterman_concept_drift_v2/', 1)[1]
            records.append({'path': str(DATA / DS['datasets']['shusterman']['path'] / relative),
                            'row': int(row['row_index']), 'label': int(mapping[row['site_name']]),
                            'date': row['collection_date'], 'role': row['protocol_role'], 'uid': uid})
    # Source candidate budget is fixed before loading features: 120/class,
    # hash order spreads the deterministic pool across historical dates.
    historical = defaultdict(list)
    for row in records:
        if row['role'] == 'historical_train':
            historical[row['label']].append(row)
    candidates = []
    for label in sorted(historical):
        candidates.extend(sorted(historical[label], key=lambda r: hashlib.sha256(str(r['uid']).encode()).hexdigest())[:120])
    source_raw, source_audit = dedup(*read_records(candidates))
    source = select_source(source_raw, CFG['model']['source_rows_per_class'])
    source_hashes = set(source['hashes'])
    targets, audit = {}, {'source': source_audit, 'dates': {}}
    for date in CFG['wtt_future_dates']:
        query_rows = [r for r in records if r['date'] == date and r['role'] == 'future_query']
        ref_rows = sorted([r for r in records if r['date'] == date and r['role'] == 'future_reference_candidate'],
                          key=lambda r: hashlib.sha256(str(r['uid']).encode()).hexdigest())[:2048]
        query, qa = dedup(*read_records(query_rows), excluded=source_hashes)
        reference, ra = dedup(*read_records(ref_rows), excluded=source_hashes | set(query['hashes']))
        if len(reference['y']) < CFG['adaptation']['support_rows']:
            raise ValueError(f'insufficient independent reference for {date}')
        reference = subset(reference, range(CFG['adaptation']['support_rows']))
        targets[date] = {'reference': reference, 'query': query, 'phase': 'frozen_external_benchmark'}
        audit['dates'][date] = {'query': qa, 'reference': ra}
    return {'source': source, 'classes': 100, 'targets': targets, 'mapping': mapping, 'audit': audit}


def read_awf(name, mapping=None):
    root = DATA / DS['datasets']['awf']['path']
    # Released labels are NumPy object strings in this trusted local archive.
    with np.load(root / name, allow_pickle=True) as data:
        labels = data['labels'].astype(str)
        if mapping is None:
            mapping = {label: i for i, label in enumerate(sorted(set(labels)))}
        y = np.array([mapping.get(label, -1) for label in labels], np.int64)
        known = np.flatnonzero(y >= 0)
        # The large archive is read once; only selected directions survive.
        if name == CFG['awf_source_file']:
            known = np.concatenate([np.flatnonzero(y == c) for c in range(len(mapping))])
        x = direction(data['data'][known])
        return x, y[known], known, mapping, int((y < 0).sum())


def awf():
    x, y, uids, mapping, _ = read_awf(CFG['awf_source_file'])
    source_raw, sa = dedup(x, y, uids)
    source = select_source(source_raw, CFG['model']['source_rows_per_class'])
    source_hashes = set(source['hashes'])
    targets, audit = {}, {'source': sa, 'files': {}}
    for name in CFG['awf_query_files']:
        x, y, uids, _, unknown = read_awf(name, mapping)
        clean, qa = dedup(x, y, uids, excluded=source_hashes)
        order = sorted(range(len(clean['y'])), key=lambda i: hashlib.sha256((name + '|' + clean['hashes'][i]).encode()).hexdigest())
        count = CFG['adaptation']['support_rows']
        assert len(order) > count
        targets[name] = {'reference': subset(clean, order[:count]), 'query': subset(clean, order[count:]),
                         'phase': 'heldout_reference_diagnostic'}
        audit['files'][name] = {**qa, 'unknown_label_rows_excluded': unknown, 'source_defined_class_count': len(mapping)}
    return {'source': source, 'classes': len(mapping), 'targets': targets, 'mapping': mapping, 'audit': audit}


def check(bundle):
    source = bundle['source']
    n = bundle['classes']
    assert set(source['y']) == set(range(n))
    for target in bundle['targets'].values():
        reference, query = target['reference'], target['query']
        assert set(source['hashes']).isdisjoint(reference['hashes'])
        assert set(source['hashes']).isdisjoint(query['hashes'])
        assert set(reference['hashes']).isdisjoint(query['hashes'])
        assert len(query['y']) > 0
        for split in (source, reference, query):
            assert (split['y'] >= 0).all() and (split['y'] < n).all()
            assert len(split['hashes']) == len(set(split['hashes']))
    return bundle


def load_cached(name):
    import torch
    preflight = json.loads((RUN / 'artifacts/data_preflight.json').read_text())
    path = RUN / 'artifacts' / f'{name}_prepared.pt'
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    assert h.hexdigest() == preflight['cache_sha256'][name]
    return check(torch.load(path, map_location='cpu', weights_only=False))
