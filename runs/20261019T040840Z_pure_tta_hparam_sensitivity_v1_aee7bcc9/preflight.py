"""Source-only scope smoke and inherited split audit; no target scoring."""
import hashlib
import json

import numpy as np
import torch

from adapt import adapt
from methods import forward_features
from runtime import RUN, ROOT, C, setup, verify_freeze, sha, save, load_source, load_target, load_model


def fingerprints(values):
    return {hashlib.sha256(row.numpy().astype('<f4', copy=False).tobytes()).hexdigest() for row in values}


def main():
    setup()
    verify_freeze()
    base = ROOT / 'runs' / C['base_matrix_run']
    base_audit = json.loads((base / 'artifacts/postrun_audit.json').read_text())
    assert base_audit['passed'] and base_audit['total_metrics_recomputed'] == 45
    source = load_source()
    prepared = torch.load(ROOT / C['source_run'] / 'artifacts/prepared.pt', weights_only=True, map_location='cpu')
    protected = fingerprints(prepared['source']['timestamps']) | fingerprints(prepared['valid']['timestamps'])
    splits = {}
    for role in C['roles']:
        reference, query = load_target(role)
        sf, qf = fingerprints(reference['timestamps']), fingerprints(query['timestamps'])
        assert len(sf) == 1024 and len(qf) == len(query['rows'])
        assert protected.isdisjoint(sf | qf) and sf.isdisjoint(qf)
        splits[role] = {'source_valid_overlap': 0, 'reference_query_overlap': 0,
                        'reference_rows': len(reference['rows']), 'query_rows': len(query['rows'])}
    fake_support = {key: source[key][:1024].clone() for key in ('timestamps', 'tam', 'rows')}
    model = load_model(42)
    with torch.inference_mode():
        x, t = source['timestamps'][:16].cuda(), source['tam'][:16].cuda()
        a = model(x, t)
        b, _ = forward_features(model, x, t)
        assert torch.equal(a, b)
    del model
    reports = {}
    for variant, settings in C['variants'].items():
        model = load_model(42)
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        condition = settings['condition']
        result = adapt(model, source, fake_support, condition, 42,
                       'source_only_smoke', steps=2, lr_override=settings['learning_rate'],
                       gate_override=settings.get('gate_nats'))
        assert result['cost']['candidate_updates'] == 2
        assert result['cost']['learning_rate'] == settings['learning_rate']
        assert all(torch.equal(model.state_dict()[k].cpu(), initial[k]) for k in initial if k.startswith('classifier.'))
        if condition == 'no_pseudo_fixed':
            assert any(not torch.equal(model.state_dict()[k].cpu(), initial[k]) for k in initial if k.startswith('final_norm.'))
        else:
            trainable = set(result['cost']['trainable_names'])
            assert all(torch.equal(model.state_dict()[k].cpu(), initial[k]) for k in initial if k not in trainable)
        reports[variant] = result['cost']
        del model
        torch.cuda.empty_cache()
    target_streams = [v['streams']['target'] for v in reports.values()]
    assert all(s == target_streams[0] for s in target_streams), 'target batch ordering varies by hyperparameter'
    source_streams = [v['streams']['source'] for k,v in reports.items() if C['variants'][k]['condition']=='no_pseudo_fixed']
    assert all(s == source_streams[0] for s in source_streams), 'source batch ordering varies by hyperparameter'
    save('preflight.json', {'passed': True, 'config_sha256': sha(RUN / 'config.json'),
        'base_audit_sha256': sha(base / 'artifacts/postrun_audit.json'),
        'splits': splits, 'source_only_smoke': reports,
        'target_query_labels_used': False, 'real_target_smoke': False})
    print('Source-only preflight passed', flush=True)


if __name__ == '__main__':
    main()
