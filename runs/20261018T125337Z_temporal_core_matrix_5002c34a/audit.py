"""Reload all adapted checkpoints and independently recompute paired metrics."""
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from runtime import RUN, ROOT, C, setup, verify_freeze, load_model, load_target, predict, configure, sha, save


def main():
    setup()
    verify_freeze()
    scores = json.loads((RUN / 'artifacts/scores.json').read_text())
    assert len(scores) == 54
    expected = {(s, r, c) for s in C['seeds'] for r in C['roles'] for c in C['conditions'] + ['tent_coadapt']}
    assert expected == {(r['seed'], r['role'], r['condition']) for r in scores}
    reloaded = 0
    for role in C['roles']:
        _, query = load_target(role)
        y = query['labels'].numpy()
        for seed in C['seeds']:
            pure = load_model(seed)
            with np.load(ROOT / 'runs' / C['frozen_baseline_run'] / 'artifacts' / f'pred_{seed}_{role}_frozen.npz') as z:
                base = z['predictions'].copy()
            base_f1 = f1_score(y, base, labels=list(range(102)), average='macro', zero_division=0)
            for condition in C['conditions'] + ['tent_coadapt']:
                row = next(r for r in scores if (r['seed'], r['role'], r['condition']) == (seed, role, condition))
                if condition == 'frozen':
                    path = ROOT / row['prediction_file']
                    with np.load(path) as z:
                        assert np.array_equal(z['rows'], query['rows'].numpy()) and np.array_equal(z['labels'], y)
                        p = z['predictions'].copy()
                else:
                    key = f'{seed}_{role}_{condition}'
                    path = RUN / 'artifacts' / f'pred_{key}.npz'
                    ck_path = RUN / 'checkpoints' / f'{key}.pt'
                    assert sha(path) == row['prediction_sha256'] and sha(ck_path) == row['checkpoint_sha256']
                    ck = torch.load(ck_path, map_location='cpu', weights_only=True)
                    assert ck['config_sha256'] == sha(RUN / 'config.json')
                    model = load_model(seed)
                    names = configure(model, condition)
                    model.load_state_dict(ck['state_dict'], strict=True)
                    original = pure.state_dict()
                    for name, value in model.state_dict().items():
                        if condition == 'tent_entropy_gate' and name not in names:
                            assert torch.equal(value, original[name]), name
                        if condition == 'no_pseudo_fixed' and name.startswith(('classifier.', 'final_norm.')):
                            assert torch.equal(value, original[name]), name
                        if condition == 'tent_bn' and name in dict(model.named_parameters()) and name not in names:
                            assert torch.equal(value, original[name]), name
                        if condition == 'tent_coadapt' and name in dict(model.named_buffers()):
                            assert torch.equal(value, original[name]), name
                    logits = predict(model, query)
                    with np.load(path) as z:
                        assert np.array_equal(z['rows'], query['rows'].numpy()) and np.array_equal(z['labels'], y)
                        assert np.allclose(logits, z['logits'], atol=1e-6, rtol=1e-5)
                        p = z['predictions'].copy()
                        assert np.array_equal(p, logits.argmax(1))
                    updates = json.loads((RUN / 'artifacts' / f'updates_{key}.json').read_text())
                    dynamics = json.loads((RUN / 'artifacts' / f'dynamics_{key}.json').read_text())
                    assert len(updates) == 800 and [r['step'] for r in dynamics] == list(range(0, 801, 10))
                    assert all(d['scope'] == 'fixed_1024_unlabeled_reference_eval_mode' for d in dynamics)
                    assert row['query_batch_statistics'] == (condition == 'tent_bn')
                    reloaded += 1
                    del model
                acc = accuracy_score(y, p)
                f1 = f1_score(y, p, labels=list(range(102)), average='macro', zero_division=0)
                assert abs(acc - row['accuracy']) < 1e-12 and abs(f1 - row['macro_f1']) < 1e-12
                assert abs(100 * (f1 - base_f1) - row['delta_f1_pp']) < 1e-10
                correct = base == y
                nar = float((correct & (p != y)).sum() / correct.sum())
                recovery = float((~correct & (p == y)).sum() / (~correct).sum())
                assert nar == row['negative_adaptation_rate'] and recovery == row['recovery_rate']
            del pure
    assert reloaded == 45
    save('postrun_audit.json', {'passed': True, 'total_metrics_recomputed': 54, 'adapted_checkpoints_reloaded': reloaded,
         'historical_pure_frozen_references': 9, 'query_labels_used_for_updates_or_selection': False,
         'tent_bn_query_forward_is_transductive': True})


if __name__ == '__main__':
    main()
