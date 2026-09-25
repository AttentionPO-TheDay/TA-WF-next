"""CPU matched-processing exact-length information control. Explicit copy of local gated-fusion run."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from ta_wf_next.traffic_views import generate_views

R = Path(__file__).resolve().parent
ROOT = R.parents[1]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, obj):
    (R / 'artifacts' / name).write_text(json.dumps(obj, indent=2))

def metric(y, p):
    cm = np.bincount(y * 102 + p, minlength=10404).reshape(102, 102)
    den = cm.sum(0) + cm.sum(1)
    return dict(accuracy=float((y == p).mean()), macro_f1=float(np.divide(
        2 * cm.diagonal(), den, out=np.zeros(102), where=den != 0).mean()))

class TokenNet(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        self.arm = arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.direction = nn.Embedding(3,16,padding_idx=0)
            self.position = nn.Embedding(512,32)
            self.c1 = nn.Conv1d(32,32,5,padding=2)
            self.c2 = nn.Conv1d(32,32,5,padding=2)
            self.head = nn.Linear(512,102)
            torch.manual_seed(seed+10000)
            torch.manual_seed(seed+20000)
            self.bucket = nn.Embedding(14,16,padding_idx=0)
            torch.manual_seed(seed+30000)
            self.bucket_process = nn.Sequential(nn.Linear(16,16),nn.GELU())
            if arm != 'baseline':
                torch.manual_seed(seed+40000)
                self.adapter = nn.Sequential(nn.Linear(32,8),nn.GELU(),nn.Linear(8,32))
                if arm == 'adapter_zero':
                    nn.init.zeros_(self.adapter[2].weight)
                    nn.init.zeros_(self.adapter[2].bias)

    def forward(self,d,e,b):
        mask = (d!=0).unsqueeze(-1)
        be = self.bucket(b); le = be + self.bucket_process(be)
        z = torch.cat([self.direction(d),le],-1)
        if self.arm != 'baseline': z = z + self.adapter(z)
        z = (z+self.position(torch.arange(512)))*mask
        for conv in (self.c1,self.c2):
            z = torch.relu(conv(z.transpose(1,2))).transpose(1,2)*mask
        pooled = z.reshape(-1,16,32,32).sum(2)/mask.reshape(-1,16,32,1).sum(2).clamp_min(1)
        return self.head(pooled.flatten(1))

def tests():
    torch.set_num_threads(2)
    d=torch.zeros(2,512,dtype=torch.long); d[:,:8]=2
    e=torch.zeros(2,512); e[:,:8]=0.2
    b=torch.zeros_like(d); b[:,:8]=3
    common=None
    for arm in ('baseline','adapter_zero','adapter_random'):
        m=TokenNet(arm,1729)
        state={k:v for k,v in m.state_dict().items() if k.split('.')[0] in ('direction','position','c1','c2','head','bucket','bucket_process')}
        if common is not None:
            for k in state: torch.testing.assert_close(state[k],common[k],rtol=0,atol=0)
        if arm == 'baseline': common={k:v.clone() for k,v in state.items()}
        ee=e.clone(); ee[:,8:]=0.9
        bb=b.clone(); bb[:,8:]=12
        torch.testing.assert_close(m(d,e,b),m(d,ee,bb),rtol=0,atol=0)
        torch.testing.assert_close(m(d,e,b),m(d,e+0.3,b),rtol=0,atol=0)
        m(d,e,b).square().sum().backward()
        assert m.bucket.weight.grad.abs().sum()>0
        assert m.bucket_process[0].weight.grad.abs().sum()>0
        if arm != 'baseline':
            assert m.adapter[2].weight.grad.abs().sum()>0
            torch.optim.SGD(m.parameters(),lr=0.01).step()
            m.zero_grad(); m(d,e,b).square().sum().backward()
            assert m.adapter[0].weight.grad.abs().sum()>0
    base=TokenNet('baseline',1729); adapted=TokenNet('adapter_zero',1729)
    torch.testing.assert_close(base(d,e,b),adapted(d,e,b),rtol=0,atol=0)
    assert sum(p.numel() for p in adapted.parameters())==sum(p.numel() for p in TokenNet('adapter_random',1729).parameters())
    print('PASS common init, padding, coarse-only permissions, two-step adapter gradients, identity initialization, capacity')

def main():
    cfg = json.loads((R / 'config.json').read_text())
    assert cfg['status'] == 'frozen' and cfg['gpu'] is False
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert cfg['roles'] == ['source', 'valid']
    assert not list((R / 'checkpoints').glob('*.pt'))
    torch.set_num_threads(cfg['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    old = ROOT / 'runs' / cfg['source_run']
    seal = json.loads((old / 'artifacts/output_seal.json').read_text())
    paths = [old / 'artifacts' / f'{role}_tokens.npz' for role in cfg['roles']]
    paths += [old / 'artifacts/manifest.json']
    for p in paths: assert sha(p) == seal[str(p.relative_to(ROOT))]
    manifest = json.loads(paths[-1].read_text())
    assert not set(manifest['direction_hashes']['source']) & set(manifest['direction_hashes']['valid'])
    data = Path(json.loads((ROOT / 'configs/datasets.json').read_text())['data_root'])
    arrays = {}; ys = {}
    for role in cfg['roles']:
        item = manifest['sampling'][role]
        with np.load(old / 'artifacts' / f'{role}_tokens.npz', allow_pickle=False) as z:
            a = tuple(z[k].copy() for k in ('direction', 'exact', 'coarse'))
        with np.load(data / item['path'], allow_pickle=False) as z:
            raw = z['X'][item['rows'], :5000]
            ys[role] = z['y'][item['rows']].astype(np.int64)
        hashes = [hashlib.sha256(np.sign(row).astype(np.int8).tobytes()).hexdigest() for row in raw]
        assert hashes == manifest['direction_hashes'][role]
        # Reconstruct every token through the current generator, not rounded floats.
        for i, row in enumerate(raw):
            runs = generate_views(row, input_kind='signed_timestamp', budget=5000, window_sizes=()).runs.runs[:512]
            n = len(runs); assert n > 0
            np.testing.assert_array_equal(a[0][i, :n], [1 if r.direction < 0 else 2 for r in runs])
            np.testing.assert_allclose(a[1][i, :n], [np.log1p(r.count) / np.log(5001) for r in runs], rtol=1e-6)
            np.testing.assert_array_equal(a[2][i, :n], [r.count.bit_length() for r in runs])
            assert all(not x[i, n:].any() for x in a)
        assert set(ys[role]) == set(range(102))
        np.savez_compressed(R / 'artifacts' / f'{role}_inputs.npz', direction=a[0], exact=a[1], coarse=a[2], labels=ys[role])
        arrays[role] = tuple(torch.from_numpy(x) for x in a)
    save('manifest.json', manifest)
    code = [R / f for f in ('PLAN.md', 'config.json', 'train.py', 'verify.py')]
    code += [ROOT / 'configs/datasets.json', ROOT / 'src/ta_wf_next/traffic_views.py', ROOT / 'src/ta_wf_next/burst_tokens.py']
    save('pretraining_seal.json', {str(p.relative_to(ROOT)): sha(p) for p in code + paths})
    y = torch.from_numpy(ys['source']); histories = {}; info = {}; preds = {}; rows = []
    vd = arrays['valid']
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            key = f'{arm}_{seed}'; model = TokenNet(arm, seed)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            generator = torch.Generator().manual_seed(seed)
            hist = []; best = -1.; start = time.perf_counter()
            for ep in range(1, cfg['epochs'] + 1):
                model.train(); total = correct = 0
                for ids in torch.randperm(len(y), generator=generator).split(cfg['batch_size']):
                    opt.zero_grad(set_to_none=True)
                    q = model(*(x[ids] for x in arrays['source']))
                    loss = nn.functional.cross_entropy(q, y[ids]); assert torch.isfinite(loss)
                    loss.backward(); opt.step()
                    total += loss.item() * len(ids); correct += int((q.argmax(1) == y[ids]).sum())
                model.eval()
                with torch.inference_mode():
                    p = torch.cat([model(*(x[i:i+128] for x in vd)).argmax(1) for i in range(0, len(vd[0]), 128)]).numpy()
                m = metric(ys['valid'], p)
                hist.append(dict(epoch=ep, loss=total / len(y), train_accuracy=correct / len(y), **m))
                if m['macro_f1'] > best:
                    best = m['macro_f1']; state = {k: v.clone() for k, v in model.state_dict().items()}; bestep = ep; bestpred = p.copy()
            model.load_state_dict(state)
            torch.save(state, R / 'checkpoints' / f'{key}.pt')
            reload = TokenNet(arm, seed); reload.load_state_dict(torch.load(R / 'checkpoints' / f'{key}.pt', weights_only=True)); reload.eval()
            with torch.inference_mode():
                rp = torch.cat([reload(*(x[i:i+128] for x in vd)).argmax(1) for i in range(0, len(vd[0]), 128)]).numpy()
            np.testing.assert_array_equal(rp, bestpred)
            preds[key] = rp; histories[key] = hist
            info[key] = dict(best_epoch=bestep, parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-start)
            rows.append(dict(key=key, arm=arm, seed=seed, **metric(ys['valid'], rp), **info[key]))
            save('history_partial.json', histories); print(key, rows[-1], flush=True)
    assert max(r['parameters'] for r in rows) / min(r['parameters'] for r in rows) < 1.01
    np.savez_compressed(R / 'artifacts/predictions.npz', **preds)
    save('history.json', histories); save('metrics.json', rows)
    outputs = list((R / 'checkpoints').glob('*.pt')) + list((R / 'artifacts').glob('*.npz'))
    outputs += [R / 'artifacts' / f for f in ('history.json', 'metrics.json', 'manifest.json')]
    save('output_seal.json', {str(p.relative_to(ROOT)): sha(p) for p in outputs})
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    tests() if '--test' in sys.argv else main()
