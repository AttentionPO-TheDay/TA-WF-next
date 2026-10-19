"""Verify fixed splits and parameter scopes before target adaptation/scoring."""
import json
import numpy as np
import torch
from runtime import (RUN, ROOT, C, setup, verify_freeze, sha, save, load_source, load_target, load_model)
from runner import adapt
from methods import forward_features


def fingerprints(values):
    import hashlib
    return {hashlib.sha256(row.numpy().astype('<f4', copy=False).tobytes()).hexdigest() for row in values}


def main():
    setup()
    verify_freeze()
    source_root = ROOT / C['source_run']
    source_audit = json.loads((source_root / 'artifacts/weight_audit.json').read_text())
    assert source_audit['passed']
    baseline_root = ROOT / 'runs' / C['frozen_baseline_run']
    baseline_audit = json.loads((baseline_root / 'artifacts/audit.json').read_text())
    assert baseline_audit['passed'] and baseline_audit['cells'] == 9
    # The baseline audit pins all source/target tensors and split manifests.
    for relative, expected in json.loads((baseline_root / 'artifacts/freeze.json').read_text()).items():
        assert sha(ROOT / relative) == expected, relative
    source = load_source()
    all_source = torch.load(source_root / 'artifacts/prepared.pt', weights_only=True, map_location='cpu')
    protected = fingerprints(all_source['source']['timestamps']) | fingerprints(all_source['valid']['timestamps'])
    splits = {}
    for role in C['roles']:
        reference, query = load_target(role)
        sf, qf = fingerprints(reference['timestamps']), fingerprints(query['timestamps'])
        assert len(sf) == 1024 and len(qf) == len(query['rows'])
        assert protected.isdisjoint(sf | qf) and sf.isdisjoint(qf)
        splits[role] = {'source_valid_overlap': 0, 'reference_query_overlap': 0,
                        'reference_rows': len(reference['rows']), 'query_rows': len(query['rows'])}
    # Five two-step scope/gradient checks use ONLY source inputs and labels;
    # no real target forward occurs in this preflight.
    support = {key: source[key][:1024].clone() for key in ('timestamps', 'tam', 'rows')}
    scopes = {}
    model = load_model(42)
    with torch.inference_mode():
        x, t = source['timestamps'][:16].cuda(), source['tam'][:16].cuda()
        a = model(x, t)
        b, _ = forward_features(model, x, t)
        assert torch.equal(a, b), 'feature wrapper changed inference'
    del model
    for condition in C['adaptation_conditions'] + ['tent_coadapt']:
        model = load_model(42)
        result = adapt(model, source, support, condition, 42, 'source_only_smoke', steps=2)
        scopes[condition] = result['cost']
        assert result['cost']['candidate_updates'] == 2
        del model
        torch.cuda.empty_cache()
    save('preflight.json', {'passed': True, 'config_sha256': sha(RUN / 'config.json'),
         'splits': splits, 'scope_smoke': scopes, 'source_only_smoke': True,
         'real_target_inference': False, 'query_labels_used_for_selection': False,
         'reuse_frozen_baseline': C['frozen_baseline_run']})
    print('Preflight passed: all scopes, gradients, feature parity, fixed split and checkpoint hashes', flush=True)


if __name__ == '__main__':
    main()
