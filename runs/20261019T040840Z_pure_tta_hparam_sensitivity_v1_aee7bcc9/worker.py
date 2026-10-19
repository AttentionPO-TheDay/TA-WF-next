"""Run all frozen sensitivity arms for one seed on an isolated GPU."""
import json
import os
import time
import traceback

import numpy as np
import torch

from adapt import adapt
from runtime import RUN, ROOT, C, save, sha, setup, verify_freeze, load_source, load_target, load_model, predict, metrics


def main():
    setup()
    verify_freeze()
    seed = int(os.environ['TEMPORAL_SEED'])
    assert seed in C['seeds'] and os.environ['CUDA_VISIBLE_DEVICES'] == str(C['budget']['physical_gpu'])
    preflight = json.loads((RUN / 'artifacts/preflight.json').read_text())
    assert preflight['passed'] and preflight['config_sha256'] == sha(RUN / 'config.json')
    base = ROOT / 'runs' / C['base_matrix_run']
    base_rows = json.loads((base / 'artifacts/scores.json').read_text())
    source = load_source()
    score_path = RUN / 'artifacts' / f'scores_seed_{seed}.json'
    scores = json.loads(score_path.read_text()) if score_path.exists() else []
    started = time.monotonic()
    for role in C['roles']:
        support, query = load_target(role)
        with np.load(ROOT / 'runs' / C['frozen_baseline_run'] / 'artifacts' /
                     f'pred_{seed}_{role}_frozen.npz') as z:
            assert np.array_equal(z['rows'], query['rows'].numpy())
            assert np.array_equal(z['labels'], query['labels'].numpy())
            frozen = z['predictions'].copy()
        for variant, settings in C['variants'].items():
            if time.monotonic() - started > C['budget']['max_wall_seconds']:
                raise TimeoutError('worker wall budget exceeded')
            key = f'{seed}_{role}_{variant}'
            condition = settings['condition']
            control = next(r for r in base_rows if (r['seed'],r['role'],r['condition']) ==
                           (seed,role,condition))
            assert sha(base / 'artifacts' / f'pred_{seed}_{role}_{condition}.npz') == control['prediction_sha256']
            found = [r for r in scores if r['seed']==seed and r['role']==role and r['variant']==variant]
            if found:
                assert len(found)==1
                assert sha(RUN / 'artifacts' / f'pred_{key}.npz') == found[0]['prediction_sha256']
                assert sha(RUN / 'checkpoints' / f'{key}.pt') == found[0]['checkpoint_sha256']
                continue
            torch.cuda.reset_peak_memory_stats()
            model = load_model(seed)
            report = adapt(model, source, support, condition, seed, role,
                           steps=settings['candidate_updates'], lr_override=settings['learning_rate'],
                           gate_override=settings.get('gate_nats'))
            logits = predict(model, query)
            predicted = logits.argmax(1)
            y = query['labels'].numpy()
            row = {'seed':seed,'role':role,'variant':variant,'condition':condition,
                   **metrics(y,predicted,frozen), **report['cost'],
                   'rows':len(predicted), 'query_batch_statistics':False,
                   'gate_nats': settings.get('gate_nats'),
                   'delta_f1_vs_control_pp':0.0,
                   'peak_cuda_bytes':int(torch.cuda.max_memory_allocated()),
                   'origin':'new_pure_temporal_sensitivity'}
            row['delta_f1_vs_control_pp'] = 100*(row['macro_f1']-control['macro_f1'])
            prediction_path = RUN / 'artifacts' / f'pred_{key}.npz'
            with prediction_path.with_suffix('.npz.tmp').open('wb') as stream:
                np.savez_compressed(stream, logits=logits, predictions=predicted, labels=y,
                                    rows=query['rows'].numpy())
            prediction_path.with_suffix('.npz.tmp').replace(prediction_path)
            ck_path = RUN / 'checkpoints' / f'{key}.pt'
            temp = ck_path.with_suffix('.partial')
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'seed':seed,'role':role,'variant':variant,
                        'config_sha256':sha(RUN/'config.json')},temp)
            temp.replace(ck_path)
            row['prediction_sha256']=sha(prediction_path)
            row['checkpoint_sha256']=sha(ck_path)
            save(f'updates_{key}.json',report['history'])
            save(f'dynamics_{key}.json',report['dynamics'])
            scores.append(row)
            save(f'scores_seed_{seed}.json',scores)
            save(f'progress_seed_{seed}.json',{'completed_new_cells':len(scores),
                'total_new_cells':21,'last_cell':key,'elapsed_seconds':time.monotonic()-started})
            print(json.dumps({'completed':key,'macro_f1':row['macro_f1'],
                              'delta_f1_vs_control_pp':row['delta_f1_vs_control_pp']}),flush=True)
            del model
            torch.cuda.empty_cache()
    assert len(scores)==21


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        save(f'failure_seed_{os.environ.get("TEMPORAL_SEED","unknown")}.json',
             {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
