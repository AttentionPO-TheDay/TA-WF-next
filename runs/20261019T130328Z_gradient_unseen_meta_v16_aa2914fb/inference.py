import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
import json
import time
import numpy as np
import torch
from runtime import C, RUN, ROOT, save, read, verify, setup, sha
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

