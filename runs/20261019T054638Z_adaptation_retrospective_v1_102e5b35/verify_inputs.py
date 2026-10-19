"""Unify historical input/output manifests and independently check uncovered summary."""
import hashlib,json
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
def main():
    c=json.loads((RUN/'config.json').read_text());known={}
    for run in c['runs'].values():
        p=ROOT/run
        for n,h in json.loads((p/'artifacts/freeze.json').read_text()).items():known.setdefault(n,set()).add(h)
        for n,h in json.loads((p/'artifacts/delivery_audit.json').read_text())['outputs'].items():known.setdefault(str((p/n).relative_to(ROOT)),set()).add(h)
    inputs=json.loads((RUN/'artifacts/inputs_sha256.json').read_text());inputs.update(json.loads((RUN/'artifacts/wider_inputs_sha256.json').read_text()))
    uncovered=[]
    for n,h in inputs.items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
        if n in known:assert h in known[n],n
        elif not n.endswith('artifacts/delivery_audit.json'):uncovered.append(n)
    r=ROOT/c['runs']['v3']/'artifacts'
    assert uncovered==[str((r/'summary.json').relative_to(ROOT))]
    summary=json.loads((r/'summary.json').read_text());scores=json.loads((r/'scores.json').read_text());comparisons=0
    for x in summary:
        rows=[v for v in scores if v['condition']==x['condition'] and v['policy']==x['policy']]
        for metric in ['macro_f1','accuracy','delta_f1_pp','nar','recovery']:
            vals=[]
            for seed in c['seeds']:
                byfactor=[]
                for factor in ['version','network','behavior']:
                    byfactor.append(np.mean([v[metric] for v in rows if v['seed']==seed and v['slice'].startswith(factor)]))
                vals.append(np.mean(byfactor))
            scale=1 if metric=='delta_f1_pp' else 100
            assert abs(float(np.mean(vals))*scale-x[metric+'_mean'])<1e-10
            assert abs(float(np.std(vals,ddof=1))*scale-x[metric+'_sd'])<1e-10
            comparisons+=2
    out=dict(passed=True,all_read_inputs_unchanged=len(inputs),historical_summary_without_delivery_hash=uncovered,
        uncovered_summary_independently_recomputed=True,summary_scalar_checks=comparisons,
        original_predictions_and_checkpoints_unchanged=True,new_model_forward=0,new_training=0)
    (RUN/'artifacts/combined_input_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
