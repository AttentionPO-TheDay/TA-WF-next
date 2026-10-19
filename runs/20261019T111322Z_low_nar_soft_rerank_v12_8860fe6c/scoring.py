import hashlib,json
import numpy as np
from scipy.special import logsumexp
from runtime import RUN,ROOT,C,save,views,state,sha

def evidence(z,b,c,temps,weights):
    n=len(b);i=np.arange(n);odds=(z[:,:,i,c]-z[:,:,i,b])/temps[:,:,None];mean=np.sum(weights[:,:,None]*odds,axis=(0,1));sd=np.sqrt(np.maximum(np.sum(weights[:,:,None]*(odds-mean)**2,axis=(0,1)),0))
    return dict(equal_logodds=odds.mean((0,1)),weighted_logodds=mean,weighted_soft=mean-C['fusion']['penalty']*sd)
def scale(x):return dict(mean=float(x.mean()),sd=max(float(x.std()),.001))
def norm(x,p):return (x-p['mean'])/p['sd']
def fit():
    z,rows,y=views('valid');temps=[]
    for seed in C['seeds']:
        s=json.loads((ROOT/C['v10_run']/'artifacts'/f'train_summary_{seed}.json').read_text());t=next(r['temperature'] for r in s if r['arm']=='direction_iat');temps.append([t]*5)
    temps=np.array(temps);q=z/temps[:,:,None,None];nll=(logsumexp(q,axis=-1)-q[:,:,np.arange(len(y)),y]).mean(-1);weights=np.exp(-nll);weights/=weights.sum();cal=dict(temperatures=temps.tolist(),view_nll=nll.tolist(),weights=weights.tolist(),sourcevalid_rows=rows.tolist(),states={})
    for seed in C['seeds']:
        for name in C['slices']:
            s=state(seed,name,'valid');assert np.array_equal(rows,s['rows']) and np.array_equal(y,s['labels']);e=evidence(z,s['base'],s['candidate'],temps,weights);sc={k:scale(v) for k,v in e.items()};sc['margin']=scale(s['margin']);cal['states'][f'{seed}_{name}']=dict(scales=sc,sourcevalid_changed=int((s['base']!=s['candidate']).sum()))
    save('source_calibration.json',cal);save('calibration_audit.json',dict(passed=True,sourcevalid_only=True,query_read=False,weight_sum=float(weights.sum()),all_weights_positive=bool((weights>0).all()),no_new_temperature_selection=True,sd_floor=.001,calibration_sha256=sha(RUN/'artifacts/source_calibration.json')))
    return cal
def scores(z,s,cal,seed,name,partition):
    e=evidence(z,s['base'],s['candidate'],np.array(cal['temperatures']),np.array(cal['weights']));sc=cal['states'][f'{seed}_{name}']['scales'];prior=norm(s['margin'],sc['margin']);aux=norm(e['weighted_soft'],sc['weighted_soft']);other=norm(e['weighted_logodds'],sc['weighted_logodds']);changed=s['base']!=s['candidate'];ix=np.flatnonzero(changed);h=[hashlib.sha256(f'v12|{seed}|{name}|{partition}|{int(s["rows"][i])}'.encode()).digest() for i in ix];perm=np.array(sorted(range(len(ix)),key=lambda j:h[j]),dtype=int);sh=aux.copy();sh[ix]=aux[ix[perm]];assert np.array_equal(np.sort(sh[ix]),np.sort(aux[ix]));w=C['fusion']['blend_weight']
    out=dict(v9_margin=s['margin'],v11_min=s['old_min'],**e,margin_plus_logodds=prior+w*other,margin_plus_soft=prior+w*aux,shuffled_margin_soft=prior+w*sh);assert list(out)==C['scores'] and all(np.isfinite(v).all() for v in out.values())
    return out
