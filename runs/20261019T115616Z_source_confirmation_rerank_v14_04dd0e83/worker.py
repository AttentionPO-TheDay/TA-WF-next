import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import json
import time
import numpy as np
import torch
from runtime import C, RUN, ROOT, save, read, verify, setup, selected, sha
from models import MainNet, ViewNet, attach, input_features


def tam(t):
    nz = t != 0
    bins = (t.abs().double()*1799/80).floor().long().clamp_max(1799) + (t < 0).long()*1800
    bins += torch.arange(len(t))[:, None]*3600
    return torch.bincount(bins[nz], minlength=len(t)*3600).reshape(len(t), 2, 1800).float()


def thin(t, rows, percent, salt=0):
    valid = t != 0
    idx = torch.arange(t.shape[1])[None, :].expand(len(t), -1)
    h = (rows[:, None]*1000003 + idx*9176 + 1103515245 + salt) % 2147483647
    h = ((h ^ (h >> 13))*1274126177) % 2147483647
    keep = valid & ((h % 100 >= percent) | (valid.long().cumsum(-1) == 1))
    order = torch.argsort(torch.where(keep, idx, idx+t.shape[1]), dim=-1)
    return t.gather(1, order)*(idx < keep.sum(-1)[:, None])


def perturb(d, case, ci, role):
    if case['name'] == 'clean':
        return d
    t = d['timestamps'].clone()*case['time']
    if case['thin_percent']:
        t = thin(t, d['rows'], case['thin_percent'], ci*104729 + (97 if role == 'valid' else 0))
    if case['tail'] < 1:
        valid = t != 0
        count = torch.ceil(valid.sum(-1)*case['tail']).long()
        t *= valid.long().cumsum(-1) <= count[:, None]
    return dict(timestamps=t, tam=tam(t), rows=d['rows'], labels=d['labels'])


def tiny(t, rows, view):
    if view == 'original':
        return t
    if view in ['time095', 'time105']:
        return t*(.95 if view == 'time095' else 1.05)
    if view == 'thin01':
        return thin(t, rows, 1)
    assert view == 'tail98'
    valid = t != 0
    count = (valid.sum(-1)*98+99)//100
    return t*(valid.long().cumsum(-1) <= count[:, None])


def main_model(seed, sl=None):
    m = MainNet().cuda().eval()
    state = torch.load(ROOT/C['source_run']/'checkpoints'/f'pure_source_wide_bn_{seed}.pth', weights_only=True, map_location='cpu')
    m.load_state_dict(state['state_dict'], strict=True)
    if sl:
        a = torch.load(ROOT/C['v9']/'checkpoints'/f'{seed}_{sl}_smooth_balance.pt', weights_only=True, map_location='cpu')
        attach(m, a['adapter'])
    return m.requires_grad_(False)


def aux_model(seed):
    m = ViewNet().cuda().eval()
    state = torch.load(ROOT/C['v10']/'checkpoints'/f'{seed}_direction_iat.pt', weights_only=True, map_location='cpu')
    assert state['seed'] == seed and state['arm'] == 'direction_iat'
    m.load_state_dict(state['model'], strict=True)
    return m.requires_grad_(False)


@torch.no_grad()
def main_predict(m, d):
    return np.concatenate([m(b.cuda()).cpu().numpy() for b in d['tam'].split(C['budget']['main_batch'])])


@torch.no_grad()
def aux_predict(m, d, view):
    out = []
    batch = C['budget']['aux_batch']
    for start in range(0, len(d['rows']), batch):
        t = tiny(d['timestamps'][start:start+batch], d['rows'][start:start+batch], view)
        out.append(m(input_features(t).cuda()).cpu().numpy())
    return np.concatenate(out)


def margin(z, c):
    x = torch.as_tensor(z, device='cuda')
    x = x/x.square().mean(-1, keepdim=True).add(1e-12).sqrt()
    cc = torch.as_tensor(c, device='cuda')
    return (x.gather(1, cc[:, None]).squeeze(1) - x.clone().scatter_(1, cc[:, None], -torch.inf).max(-1).values).cpu().numpy()


def load():
    seed = int(os.environ['SOURCE_SEED'])
    setup(seed)
    verify()
    d = torch.load(ROOT/C['source_run']/'artifacts/prepared.pt', weights_only=True, map_location='cpu')
    models = {'frozen': main_model(seed), **{sl: main_model(seed, sl) for sl in C['slices']}}
    aux = aux_model(seed)
    return seed, d, models, aux


def preflight():
    seed, d, models, aux = load()
    v = d['valid']
    assert torch.equal(tam(v['timestamps']), v['tam'])
    for sl, m in models.items():
        z = main_predict(m, v)
        for name in C['slices'] if sl == 'frozen' else [sl]:
            old = read(ROOT/C['v9']/'artifacts'/f'valid_{seed}_{name}_{"frozen" if sl=="frozen" else "smooth_balance"}.npz')
            assert np.array_equal(z, old['logits']), (seed, sl, 'main identity')
            assert np.array_equal(v['rows'].numpy(), old['rows'])
            if sl != 'frozen':
                base = read(ROOT/C['v9']/'artifacts'/f'valid_{seed}_{name}_frozen.npz')['logits']
                cp = z.argmax(-1)
                assert np.array_equal(margin(z, cp)-margin(base, cp), old['margin_gain']), (seed, sl, 'margin identity')
    z = aux_predict(aux, v, 'original')
    old = read(ROOT/C['v10']/'artifacts'/f'sourcevalid_{seed}_direction_iat.npz')
    assert np.array_equal(z, old['logits']), (seed, 'aux identity')
    for role in ['train', 'valid']:
        source = d['source' if role == 'train' else 'valid']
        ix = selected(d, role)
        expected = json.loads((RUN/'artifacts/source_rows.json').read_text())[role]
        assert source['rows'][ix].tolist() == expected
        small = {k: val[ix[:8]] for k, val in source.items()}
        for ci, case in enumerate(C['perturbations']):
            out = perturb(small, case, ci, role)
            one = perturb({k: val[:1] for k, val in small.items()}, case, ci, role)
            assert torch.equal(out['timestamps'][:1], one['timestamps'])
            assert torch.equal(out['tam'][:1], one['tam'])
            assert torch.isfinite(out['timestamps']).all()
            assert (out['tam'].sum((1, 2)) == (out['timestamps'] != 0).sum(-1)).all()
    assert all(not p.requires_grad for m in list(models.values())+[aux] for p in m.parameters())
    save(f'preflight_{seed}.json', dict(passed=True, full_sourcevalid_main_states=3, full_sourcevalid_aux_states=1, original_logits_exact=True, deterministic_batch_independent_perturbations=True, models_locked=True))


def compute(audit=False):
    begin = time.monotonic()
    seed, d, models, aux = load()
    before = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in list(models.values())+[aux]]
    checked = 0
    for role in ['train', 'valid']:
        source = d['source' if role == 'train' else 'valid']
        ix = selected(d, role)
        if audit:
            ix = ix[:256]
        src = {k: val[ix] for k, val in source.items()}
        for ci, case in enumerate(C['perturbations']):
            assert time.monotonic()-begin < C['budget']['wall_seconds']
            dd = perturb(src, case, ci, role)
            result = {k: dd[k].numpy() for k in ['rows', 'labels']}
            zz = {key: main_predict(m, dd) for key, m in models.items()}
            result['frozen_logits'] = zz['frozen']
            for sl in C['slices']:
                z = zz[sl]
                result[sl+'_logits'] = z
                cp = z.argmax(-1)
                result[sl+'_margin'] = margin(z, cp)-margin(zz['frozen'], cp)
            result['aux_logits'] = np.stack([aux_predict(aux, dd, view) for view in C['views']])
            path = RUN/'artifacts'/f'synthetic_{seed}_{role}_{case["name"]}.npz'
            if audit:
                old = read(path)
                for k, v in result.items():
                    want = old[k][:, :len(ix)] if k == 'aux_logits' else old[k][:len(ix)]
                    assert np.array_equal(want, v), (path.name, k)
            else:
                assert all(np.isfinite(v).all() for v in result.values())
                np.savez_compressed(path, **result)
            checked += 1
            print(json.dumps(dict(seed=seed, role=role, perturbation=case['name'], rows=len(ix), audit=audit)), flush=True)
    for m, old in zip(list(models.values())+[aux], before):
        assert all(torch.equal(v.detach().cpu(), old[k]) for k, v in m.state_dict().items())
    save(f'{"reload_audit" if audit else "source_forward"}_{seed}.json', dict(passed=True, cache_states=checked, probe_rows_per_state=256 if audit else None, models_unchanged=True, seconds=time.monotonic()-begin))


if __name__ == '__main__':
    import sys
    {'preflight': preflight, 'compute': compute, 'audit': lambda: compute(True)}[sys.argv[1]]()
