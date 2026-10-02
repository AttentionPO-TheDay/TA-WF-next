"""Fixed source/valid RF TAM structural preparation; no optimization or future access."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from rf_native import RFNative

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
PRIOR = ROOT/'runs/20261001T052912Z_source_error_capacity_audit_b32edd4f/artifacts/prepared.pt'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''): h.update(b)
    return h.hexdigest()

def save(p, value):
    p.write_text(json.dumps(value, indent=2)+'\n')

def tam(x):
    out = np.zeros((len(x), 1, 2, 1800), np.float32)
    for i, row in enumerate(x):
        obs = row[row != 0]
        t = np.abs(obs.astype(np.float64))
        bins = np.minimum((t * 1799 / 80).astype(np.int64), 1799)
        np.add.at(out[i, 0], ((obs < 0).astype(np.int64), bins), 1)
    return out

def author_scalar(row):
    # Literal official fun algorithm with signed-timestamp -> times/sizes adaptation.
    times = [abs(float(v)) for v in row if v != 0]
    sizes = [np.sign(float(v)) for v in row if v != 0]
    feature = [[0 for _ in range(1800)], [0 for _ in range(1800)]]
    for i in range(len(sizes)):
        if sizes[i] > 0:
            if times[i] >= 80: feature[0][-1] += 1
            else: feature[0][int(times[i] * 1799 / 80)] += 1
        if sizes[i] < 0:
            if times[i] >= 80: feature[1][-1] += 1
            else: feature[1][int(times[i] * 1799 / 80)] += 1
    return np.array(feature, np.float32)

def audit(x):
    finite = np.isfinite(x)
    obs = x != 0
    internal = (((~obs).cumsum(1)>0)&obs).any(1)
    assert finite.all(), 'nonfinite timestamps'
    assert not internal.any(), 'internal padding'
    assert obs.any(1).all(), 'empty rows'
    stats = {'rows':len(x), 'width':x.shape[1], 'nonfinite':int((~finite).sum()), 'internal_zero_rows':int(internal.sum()), 'padding_count':int((~obs).sum()), 'packet_count':int(obs.sum()), 'ge80_packets':int(((np.abs(x)>=80)&obs).sum()), 'ge80_rows':int(((np.abs(x)>=80)&obs).any(1).sum())}
    for label, select in [('mixed',lambda row:row[row!=0]), ('positive',lambda row:row[row>0]), ('negative',lambda row:row[row<0])]:
        pairs = bad = rows_bad = 0
        for row in x:
            dt = np.diff(np.abs(select(row).astype(np.float64)))
            pairs += len(dt); bad += int((dt<0).sum()); rows_bad += int((dt<0).any())
        stats[label] = {'adjacent_pairs':pairs,'negative_deltas':bad,'negative_delta_fraction':bad/pairs if pairs else 0,'rows_with_negative_delta':rows_bad,'row_fraction':rows_bad/len(x)}
    return stats

def main():
    start=time.monotonic(); torch.set_num_threads(4)
    (R/'artifacts').mkdir(exist_ok=True)
    synthetic=np.array([[1e-8,-.1,80/1799,-79.99999,80,-80,81,-81,0,0], [2,-1,1,-2,0,0,0,0,0,0]],np.float32)
    st=tam(synthetic)
    assert all(np.array_equal(st[i,0],author_scalar(row)) for i,row in enumerate(synthetic))
    m=json.loads((ROOT/'configs/datasets.json').read_text())
    base=Path(m['data_root'])/m['datasets']['proteus_temporal']['path']
    prior=torch.load(PRIOR,map_location='cpu',weights_only=False)
    result={}; audits={}; raw_hash={}
    for role, oldrole, filename in [('source','source150','train.npz'),('valid','valid','valid.npz')]:
        old=prior[oldrole]; rows=old['rows'].numpy()
        with np.load(base/filename,allow_pickle=False) as f:
            x=f['X'][rows,:5000].astype(np.float32); y=f['y'][rows].astype(np.int64)
        d=np.sign(x).astype(np.float32)
        assert np.array_equal(d,old['directions'].numpy()), 'prior direction mismatch'
        assert np.array_equal(y,old['labels'].numpy()), 'prior label mismatch'
        audits[role]=audit(x)
        t=tam(x)
        assert np.array_equal(t.sum((1,2,3)),(x!=0).sum(1))
        assert np.array_equal(t[:,0,0].sum(1),(x>0).sum(1))
        assert np.array_equal(t[:,0,1].sum(1),(x<0).sum(1))
        check=np.unique(np.linspace(0,len(x)-1,64,dtype=int))
        assert all(np.array_equal(t[i,0],author_scalar(x[i])) for i in check)
        audits[role]['scalar_parity_rows']=len(check)
        audits[role]['count_conservation']=True
        result[role]={'direction':torch.from_numpy(d[:,None,:]),'tam':torch.from_numpy(t),'labels':torch.from_numpy(y),'rows':old['rows'].clone()}
        raw_hash[str(base/filename)]=sha(base/filename)
    torch.save(result,R/'artifacts/native_prepared.pt')
    torch.manual_seed(1729)
    model=RFNative(num_classes=102).eval()
    with torch.inference_mode(): out=model(torch.from_numpy(st))
    assert out.shape==(2,102) and torch.isfinite(out).all()
    audits.update({'synthetic_scalar_parity':True,'cpu_forward_shape':list(out.shape),'parameters':sum(p.numel() for p in model.parameters()),'future_access':False,'sorting':False,'training':False,'elapsed_seconds':time.monotonic()-start})
    save(R/'artifacts/tam_audit.json',audits)
    paths=[PRIOR,ROOT/'configs/datasets.json',R/'prepare.py',R/'rf_native.py',R/'PLAN_PREP.md',R/'artifacts/native_prepared.pt',R/'artifacts/tam_audit.json']
    save(R/'artifacts/prep_manifest.json',{'raw_hashes':raw_hash,'hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths},'input_rule':'first5000 stored order cast float32; sign direction; float64 abs time arithmetic; no sort; TAM counts 2x1800/80sec'})
    (R/'RF_PREPARATION.md').write_text('# RF preparation completed\n\nAll fixed-row label/direction, padding, finite, per-channel count conservation, scalar author-algorithm parity and synthetic CPU forward checks passed. No optimizer, training or future access. Official model unchanged except RFNative factory.\n\nTiming-inclusive condition uses observed signed absolute timestamps without sorting. Mixed-direction backtracking remains a known data-semantic limitation; within-direction statistics are recorded, not corrected. Clipping >=80 seconds is author-defined last-bin accumulation. This interface does not establish timing causal validity or performance.\n\n```json\n'+json.dumps(audits,indent=2)+'\n```\n')
    print(json.dumps(audits,indent=2))

if __name__=='__main__': main()
