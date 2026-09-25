"""Frozen, read-only checkpoint intervention diagnostic; no model training."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

R = Path(__file__).resolve().parent
ROOT = R.parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, obj):
    (R / 'artifacts' / name).write_text(json.dumps(obj, indent=2) + '\n')

def metric(y, p):
    cm = np.bincount(y * 102 + p, minlength=102**2).reshape(102, 102)
    den = cm.sum(0) + cm.sum(1)
    return dict(accuracy=float((y == p).mean()), macro_f1=float(np.divide(2*cm.diagonal(), den, out=np.zeros(102), where=den != 0).mean()), zero_recall_classes=int((cm.diagonal()==0).sum()))

def main():
    start = time.perf_counter()
    cfg = json.loads((R/'config.json').read_text())
    assert cfg['status']=='frozen' and not cfg['gpu']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    old = ROOT/'runs'/cfg['source_run']
    seal = json.loads((old/'artifacts/output_seal.json').read_text())
    paths = [old/'artifacts'/f'{role}_inputs.npz' for role in cfg['roles']]
    paths += [old/'artifacts/predictions.npz', old/'artifacts/manifest.json']
    paths += [old/'checkpoints'/f'hybrid_{s}.pt' for s in cfg['seeds']]
    for p in paths:
        assert sha(p)==seal[str(p.relative_to(ROOT))], str(p)
    preseal = json.loads((old/'artifacts/pretraining_seal.json').read_text())
    assert sha(old/'train.py') == preseal[str((old/'train.py').relative_to(ROOT))]
    manifest = json.loads((old/'artifacts/manifest.json').read_text())
    assert not set(manifest['direction_hashes']['source']) & set(manifest['direction_hashes']['valid'])
    save('input_seal.json', {str(p.relative_to(ROOT)):sha(p) for p in paths+[old/'train.py', R/'config.json', R/'PLAN.md', R/'diagnose.py', R/'verify.py']})
    spec = importlib.util.spec_from_file_location('explicit_local_typed_model', old/'train.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    metrics=[]; scales=[]; predictions={}; labels={}
    with np.load(old/'artifacts/predictions.npz', allow_pickle=False) as z:
        historical={k:z[k].copy() for k in z.files if k.startswith('hybrid_')}
    for role in cfg['roles']:
        with np.load(old/'artifacts'/f'{role}_inputs.npz', allow_pickle=False) as z:
            arrays=tuple(torch.from_numpy(z[k].copy()) for k in ('direction','exact','coarse'))
            labels[role]=z['labels'].copy()
        for seed in cfg['seeds']:
            model=module.TokenNet('hybrid', seed)
            model.load_state_dict(torch.load(old/'checkpoints'/f'hybrid_{seed}.pt', map_location='cpu', weights_only=True)); model.eval()
            sums={}; nums={}
            with torch.inference_mode():
                for i in range(0,len(arrays[0]),128):
                    d,e,b=(x[i:i+128] for x in arrays); mask=d!=0
                    c=model.continuous(e.unsqueeze(-1)); bias=model.continuous.bias.expand_as(c)
                    branches={'bucket':model.bucket(b), 'continuous':c, 'bias':bias, 'variable':c-bias, 'direction':model.direction(d), 'position':model.position(torch.arange(512)).unsqueeze(0).expand(len(d),-1,-1)}
                    for name,val in branches.items():
                        v=val[mask].double(); sums[name]=sums.get(name,0.)+float(v.square().sum()); nums[name]=nums.get(name,0)+v.numel()
            scales.append(dict(role=role,seed=seed, rms={k:(sums[k]/nums[k])**.5 for k in sums}, valid_tokens=int((arrays[0]!=0).sum())))
            for condition in cfg['interventions']:
                hook=None
                if condition=='off': hook=model.continuous.register_forward_hook(lambda m,a,o: torch.zeros_like(o))
                if condition=='bias_only': hook=model.continuous.register_forward_hook(lambda m,a,o: m.bias.expand_as(o))
                with torch.inference_mode():
                    pred=torch.cat([model(*(x[i:i+128] for x in arrays)).argmax(1) for i in range(0,len(arrays[0]),128)]).numpy()
                if hook is not None: hook.remove()
                key=f'{role}_{seed}_{condition}'; predictions[key]=pred
                full=predictions[f'{role}_{seed}_full']
                if role=='valid' and condition=='full': np.testing.assert_array_equal(pred,historical[f'hybrid_{seed}'])
                metrics.append(dict(key=key,role=role,seed=seed,condition=condition,changed_fraction=float((pred!=full).mean()),**metric(labels[role],pred)))
    np.savez_compressed(R/'artifacts/predictions.npz',**predictions)
    np.savez_compressed(R/'artifacts/labels.npz',**labels)
    save('metrics.json',metrics); save('scales.json',scales)
    save('execution.json',dict(seconds=time.perf_counter()-start,gpu=False,threads=1,full_valid_prediction_match=True))
    outputs=[p for p in (R/'artifacts').iterdir() if p.name!='output_seal.json']
    save('output_seal.json',{str(p.relative_to(ROOT)):sha(p) for p in outputs})
    print(json.dumps({'metrics':metrics,'scales':scales},indent=2))

if __name__=='__main__': main()
