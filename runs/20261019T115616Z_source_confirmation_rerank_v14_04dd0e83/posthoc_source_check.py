"""Posthoc cached source-validation diagnosis; never refits or selects a model."""
import json
import numpy as np
from runtime import C, RUN, save, csvout, read, verify
from rerank import fitverify
from metrics import curve

verify()
fitverify()
rows = []
for seed in C['seeds']:
    for sl in C['slices']:
        d = read(RUN/'artifacts'/f'source_features_valid_{seed}_{sl}.npz')
        ss = read(RUN/'artifacts'/f'source_scores_valid_{seed}_{sl}.npz')
        for score in C['scores']:
            m, _, _ = curve(d['labels'], d['base'], d['candidates'], ss[score])
            rows.append(dict(seed=seed, slice=sl, score=score, **m))
save('posthoc_source_valid_cells.json', rows)
csvout('posthoc_source_valid_cells.csv', rows)
summary = []
for sl in C['slices']:
    for score in C['scores']:
        rr = [r for r in rows if r['slice'] == sl and r['score'] == score]
        summary.append(dict(slice=sl, score=score, pauc_norm=float(np.mean([r['pauc_norm'] for r in rr])), rec1=float(np.mean([r['rec_at_nar1'] for r in rr])), W=float(np.mean([r['other_wrong_accepted'] for r in rr]))))
save('posthoc_source_valid_summary.json', summary)
print(json.dumps(summary, ensure_ascii=False))
save('posthoc_diagnostic_scope.json', dict(reason='assess source synthetic versus real target performance after frozen main gate failed', posthoc=True, cached_source_valid_only=True, new_models=0, new_thresholds_for_deployment=0, original_decisions_unchanged=True, source_validation_is_observed_and_calibration_used=True))
verify()
fitverify()
