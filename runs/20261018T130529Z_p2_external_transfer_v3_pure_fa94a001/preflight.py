"""Inspect/freeze all roles and effective inputs before any external scoring."""
import json
from pathlib import Path
import numpy as np
import torch
from data import wtt, awf, check, DS, DATA
from runner import RUN, C, save, sha, inputs, metrics
from model import MultiViewModel
from token_alignment import aligned_generator, ResidualTokenProjector


def main():
    assert C['status'] == 'frozen'
    summary = {}
    paths = {RUN.parents[1] / 'configs/datasets.json'}
    wroot = DATA / DS['datasets']['wtt_time_processed']['path']
    paths.update([wroot / 'sample_index.jsonl', wroot / 'label_map.json'])
    # File contents are hashed; metadata-only future inspection remains
    # distinct from model training, adaptation, and performance evaluation.
    with (wroot / 'sample_index.jsonl').open() as stream:
        for line in stream:
            r = json.loads(line)
            if r['condition_name'] in ('dynamic_browser_version_network', 'static_browser_version_network') and (
                    r['protocol_role'] == 'historical_train' or r['collection_date'] in C['wtt_future_dates']):
                relative = r['source_file'].split('shusterman_concept_drift_v2/', 1)[1]
                paths.add(DATA / DS['datasets']['shusterman']['path'] / relative)
    for name in [C['awf_source_file']] + C['awf_query_files']:
        paths.add(DATA / DS['datasets']['awf']['path'] / name)
    print('hashing raw files', len(paths), flush=True)
    data_hashes = {str(path): sha(path) for path in sorted(paths)}
    prior_path = RUN / 'artifacts/data_preflight.json'
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        assert prior['passed'] and prior['config_sha256'] == sha(RUN / 'config.json')
        assert prior['data_sha256'] == data_hashes, 'raw data changed since preflight'
        for name, expected in prior['cache_sha256'].items():
            assert sha(RUN / 'artifacts' / f'{name}_prepared.pt') == expected
        print('reusing verified data preflight and immutable cached roles', flush=True)
        return
    caches = {}
    for name, build in [('wtt_time', wtt), ('awf', awf)]:
        print('structural preflight', name, flush=True)
        bundle = check(build())
        summary[name] = {'classes': bundle['classes'], 'source_rows': len(bundle['source']['y']),
                         'audit': bundle['audit'],
                         'targets': {date: {'reference_rows': len(target['reference']['y']), 'query_rows': len(target['query']['y'])}
                                     for date, target in bundle['targets'].items()}}
        cache_path = RUN / 'artifacts' / f'{name}_prepared.pt'
        torch.save(bundle, cache_path)
        caches[name] = sha(cache_path)
        del bundle
    # Contract, hook equivalence, class-count metric, batch tail and
    # effective-input isolation are exercised without real target inference.
    torch.set_num_threads(2)
    model = MultiViewModel('wide_bn', 100).cuda().eval()
    x = torch.zeros(3, 5000)
    x[:, :100] = 1
    with torch.inference_mode():
        plain = model(*inputs(x))
        with aligned_generator(model, ResidualTokenProjector().cuda()):
            aligned = model(*inputs(x))
    assert torch.equal(plain, aligned) and plain.shape == (3, 100)
    m = metrics(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), 200)
    assert m['accuracy'] == 1 and m['macro_f1'] == .01 and m['negative_adaptation_rate'] == 0
    save('data_preflight.json', {'passed': True, 'config_sha256': sha(RUN / 'config.json'),
         'data_sha256': data_hashes, 'cache_sha256': caches, 'summary': summary, 'real_target_inference': False,
         'training': False, 'adaptation': False, 'performance_scoring': False})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
