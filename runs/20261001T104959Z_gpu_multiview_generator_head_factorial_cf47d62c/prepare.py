"""CPU-only fixed-row timestamp/TAM preparation; no model, scoring, or future access."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
PRIOR_RUN = ROOT / 'runs/20261001T091714Z_gpu_native_varcnn_rf_source_ce09ee6b'
PRIOR = PRIOR_RUN / 'artifacts/native_prepared.pt'
CAPACITY = ROOT / 'runs/20261001T052912Z_source_error_capacity_audit_b32edd4f/artifacts/manifest.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def independent_tam(row):
    # Independent bincount reconstruction of the RF author's nonnegative bin rule.
    # Input is already float32; promote its observed values for author float arithmetic.
    observed = row[row != 0]
    times = np.abs(observed).astype(np.float64)
    bins = np.floor(times * 1799 / 80).astype(np.int64)
    bins[times >= 80] = 1799
    encoded = bins + (observed < 0).astype(np.int64) * 1800
    return np.bincount(encoded, minlength=3600).reshape(2, 1800).astype(np.float32)


def audit(x):
    obs = x != 0
    assert np.isfinite(x).all(), 'nonfinite input'
    assert obs.any(1).all(), 'empty trace'
    assert not (((~obs).cumsum(1) > 0) & obs).any(), 'zero inside observed prefix'
    stats = {'rows': len(x), 'width': 5000, 'nonfinite': 0,
             'internal_zero_rows': 0, 'packet_count': int(obs.sum()),
             'padding_count': int((~obs).sum()),
             'ge80_packets': int(((np.abs(x) >= 80) & obs).sum()),
             'positive_packets': int((x > 0).sum()),
             'negative_packets': int((x < 0).sum())}
    for name, selector in [('mixed', lambda v: v[v != 0]),
                           ('positive', lambda v: v[v > 0]),
                           ('negative', lambda v: v[v < 0])]:
        violations = rows_bad = 0
        for row in x:
            diff = np.diff(np.abs(selector(row)).astype(np.float64))
            violations += int((diff < 0).sum())
            rows_bad += int((diff < 0).any())
        stats[name] = {'negative_deltas': violations, 'rows_with_negative_delta': rows_bad}
    return stats


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    dataset_config = ROOT / 'configs/datasets.json'
    datasets = json.loads(dataset_config.read_text())
    base = Path(datasets['data_root']) / datasets['datasets']['proteus_temporal']['path']
    prior_manifest_path = PRIOR_RUN / 'artifacts/prep_manifest.json'
    prior_manifest = json.loads(prior_manifest_path.read_text())
    capacity = json.loads(CAPACITY.read_text())
    prior_sha = sha(PRIOR)
    assert prior_sha == prior_manifest['hashes'][str(PRIOR.relative_to(ROOT))]
    prior = torch.load(PRIOR, map_location='cpu', weights_only=False)
    prepared, audits, hashes, row_hashes = {}, {}, {}, {}
    for role, filename, selection_key, expected_count in [
            ('source', 'train.npz', 'source150_rows', 15300),
            ('valid', 'valid.npz', 'valid_rows', 510)]:
        old = prior[role]
        rows = old['rows'].numpy()
        assert rows.dtype == np.int64 and len(rows) == expected_count
        assert len(np.unique(rows)) == len(rows)
        assert np.array_equal(rows, capacity[selection_key]), 'selection changed'
        raw = base / filename
        raw_sha = sha(raw)
        assert raw_sha == prior_manifest['raw_hashes'][str(raw)]
        assert raw_sha == capacity['raw_hashes'][filename]
        with np.load(raw, allow_pickle=False) as f:
            # NPZ decompresses whole X/y members; only selected rows are retained or analyzed.
            x = np.ascontiguousarray(f['X'][rows, :5000].astype(np.float32))
            labels = np.ascontiguousarray(f['y'][rows].astype(np.int64))
        directions = np.sign(x).astype(np.float32)
        assert np.array_equal(directions, old['direction'].numpy()[:, 0])
        assert np.array_equal(labels, old['labels'].numpy())
        assert set(np.unique(labels)) == set(range(102))
        assert np.array_equal(np.bincount(labels, minlength=102),
                              np.full(102, 150 if role == 'source' else 5))
        t = old['tam'].numpy()[:, 0].copy()
        assert t.shape == (len(x), 2, 1800) and t.dtype == np.float32
        assert np.isfinite(t).all() and (t >= 0).all()
        assert all(np.array_equal(independent_tam(row), t[i]) for i, row in enumerate(x))
        assert np.array_equal(t.sum((1, 2)), (x != 0).sum(1))
        assert np.array_equal(t[:, 0].sum(1), (x > 0).sum(1))
        assert np.array_equal(t[:, 1].sum(1), (x < 0).sum(1))
        audits[role] = audit(x)
        audits[role].update({'tam_independent_parity_rows': len(x),
                            'prior_direction_labels_rows_exact': True,
                            'per_row_channel_count_conservation': True})
        row_hashes[role] = {hashlib.sha256(row.tobytes()).hexdigest() for row in directions}
        prepared[role] = {'timestamps': torch.from_numpy(x), 'tam': torch.from_numpy(t),
                          'labels': torch.from_numpy(labels), 'rows': old['rows'].clone()}
        hashes[role] = {'raw_file': str(raw), 'raw_sha256': raw_sha,
                        'selected_arrays': {name: array_sha(value.numpy())
                                            for name, value in prepared[role].items()},
                        'direction_array_sha256': array_sha(directions)}
    overlap = len(row_hashes['source'] & row_hashes['valid'])
    assert overlap == 0, 'selected source/valid direction overlap'
    synthetic = np.array([1e-8, -.1, 80 / 1799, -79.99999, 80, -80, 81, -81, 0], np.float32)
    check = independent_tam(synthetic)
    assert check.sum() == 8 and check[0, -1] == 2 and check[1, -1] == 2
    target = R / 'artifacts/prepared.pt'
    torch.save(prepared, target)
    manifest = {
        'schema_version': 1,
        'input_rule': 'First 5000 signed stored-order timestamps cast float32. No sorting, no per-packet interarrival inference. TAM is exact prior cache Bx1x2x1800 squeezed to Bx2x1800; independent author-bin reconstruction abs(float32) promoted to float64, int(time*1799/80), time>=80 in last bin.',
        'mapping': {'timestamp_sign_positive': 'positive_direction/channel0',
                    'timestamp_sign_negative': 'negative_direction/channel1', 'timestamp_zero': 'suffix_padding'},
        'selected_input_hashes': hashes,
        'hashes': {str(path.relative_to(ROOT)): sha(path) for path in
                   [dataset_config, PRIOR, prior_manifest_path, CAPACITY, R / 'prepare.py', target]},
        'audits': audits,
        'selected_source_valid_direction_overlap': overlap,
        'prior_overlap_audit': {'manifest': str(CAPACITY.relative_to(ROOT)),
                                'source_excluded_for_full_valid_overlap': capacity['excluded']['overlap'],
                                'full_valid_not_reanalyzed': True},
        'exposure': {'raw_files': ['train.npz', 'valid.npz'],
                     'selected_source_rows': 15300, 'selected_valid_rows': 510,
                     'raw_member_access': 'NPZ whole X/y member decompression; only fixed selected first5000 X and selected y retained/analyzed; file-byte hashes include full archives without additional semantic analysis.',
                     'future': False, 'wtt_time': False, 'awf': False,
                     'model_evaluation': False, 'training': False, 'cpu_only': True},
        'elapsed_seconds': time.monotonic() - started}
    (R / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (R / 'DATA_AUDIT.md').write_text(
        '# Fixed source/valid data preparation\n\n'
        'Completed on CPU. Exactly the prior native experiment’s 15,300 source rows (150/class) and 510 valid rows (5/class), in unchanged row order and label mapping. No model forward, optimization, or scoring.\n\n'
        'Finite and suffix-padding checks passed. Original signs, labels, row lists, raw archive hashes and prior cache hash match exactly. Cached TAM is independently reconstructed for every selected row, with per-row/per-direction packet conservation. Selected source/valid direction overlap is zero. The earlier full-valid overlap audit and frozen source exclusions are reused; full-valid directions are not reanalyzed here.\n\n'
        'First 5000 stored-order signed timestamps are float32; no sorting or manufactured interarrival times. Mixed-direction absolute-time backtracking remains a recorded semantic limitation; per-direction order is checked. Times at or beyond 80 seconds accumulate in the final TAM bin, following the prior audited author rule.\n\n'
        'NPZ access decompresses whole X/y members; only fixed selected rows are retained or analyzed. Full archive bytes are read for provenance hashes. This repeats authorized source/valid structure and label checks; it does not open future dates, WTT-Time, or AWF and does not add model-selection exposure. Detailed hashes, counts and mapping are in manifest.json.\n\n'
        '```json\n' + json.dumps(audits, indent=2) + '\n```\n')
    print(json.dumps({'prepared': str(target), 'sha256': sha(target),
                      'overlap': overlap, 'audits': audits, 'elapsed_seconds': manifest['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
