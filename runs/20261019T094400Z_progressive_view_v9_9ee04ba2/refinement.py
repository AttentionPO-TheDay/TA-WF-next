"""Detached teacher geometry, separately controlled student anchors; no query state."""
import time
import torch
from torch import nn
from runtime import C, entropy
from methods import attach, streams
from candidate import target_distribution
from momentum_legacy import MomentumState, objective


class RefinedState(MomentumState):
    def __init__(self, e, z, p, condition):
        super().__init__(e, z, p)
        self.condition = condition
        self.options = C['refinement_v6']['arms'].get(condition)
        if condition == 'legacy_control':
            self.options = dict(target='moving_geometry', loss_anchor='moving',
                                prototype_update=True, temperature=.1, radius_preservation=False)
        assert self.options is not None

    def anchored(self):
        a = super().anchored()
        if not self.options['radius_preservation']:
            return a
        center = a.mean(0, keepdim=True)
        d = a - center
        s = self.source - self.source.mean(0, keepdim=True)
        radius = s.norm(dim=-1, keepdim=True)
        unit = torch.where(d.norm(dim=-1, keepdim=True) > 1e-8,
                           nn.functional.normalize(d, dim=-1), nn.functional.normalize(s, dim=-1))
        return center + radius * unit

    def loss_anchors(self):
        return self.anchored() if self.options['loss_anchor'] == 'moving' else self.source

    @torch.no_grad()
    def target(self, qfixed, ix):
        if self.options['target'] == 'qfixed':
            return qfixed[ix].detach()
        p = self.source if self.options['target'] == 'source_geometry' else self.anchored()
        u = nn.functional.normalize(self.features[ix], dim=-1)
        v = nn.functional.normalize(p, dim=-1)
        qgeo = (u @ v.T / self.options['temperature']).softmax(-1)
        mix = C['momentum_v5']['prototype_target_mix']
        return ((1-mix)*qfixed[ix] + mix*qgeo).detach()

    @torch.no_grad()
    def observe(self, ix, x, z, q, w):
        super().update(ix, x, z, q, w, self.options['prototype_update'])


def train(m, t, e, z, votes, source_t, source_e, p, condition):
    cfg = C['adaptation']
    a = attach(m, 'lora')
    state = RefinedState(e, z, p, condition)
    qfixed = target_distribution(z, votes, 'soft_neighbor')
    opt = torch.optim.Adam(a.parameters(), lr=cfg['lr'])
    src = streams(len(source_t), cfg['steps'], cfg['source_batch_size'])
    history = []
    begin = time.monotonic()
    for step, ix in enumerate(streams(len(t), cfg['steps'], cfg['batch_size'])):
        opt.zero_grad(set_to_none=True)
        si = next(src)
        q = state.target(qfixed, ix)
        w, wd = state.weight(ix, True)
        prototypes = state.loss_anchors()
        loss, x, zz, detail = objective(m, a, t[ix], q, w, prototypes,
                                        source_t[si], source_e[si], p, False)
        assert not q.requires_grad and not prototypes.requires_grad and torch.isfinite(loss)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(a.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
        opt.step()
        state.observe(ix, x, zz, q, w)
        with torch.no_grad():
            anchored = state.anchored()
            drift = (anchored-state.source).norm(dim=-1)
            qent = -(q*q.clamp_min(1e-12).log()).sum(-1)
            centered = anchored-anchored.mean(0, keepdim=True)
            sr = (state.source-state.source.mean(0, keepdim=True)).norm(dim=-1)
            history.append(dict(step=step+1, loss=float(loss.detach()), gradient_norm=float(grad),
                prototype_drift_mean=float(drift.mean()), prototype_drift_max=float(drift.max()),
                prototype_radius_ratio=float((centered.norm(dim=-1)/sr.clamp_min(1e-8)).mean()),
                loss_anchor_drift_mean=float((prototypes-state.source).norm(dim=-1).mean()),
                target_entropy=float(qent.mean()),
                target_teacher_disagreement=float((q.argmax(-1)!=z[ix].argmax(-1)).float().mean()),
                student_entropy=float(entropy(zz).mean()), **wd, **detail))
        if time.monotonic()-begin > C['budget']['max_cell_seconds']:
            raise TimeoutError('adaptation cell')
    assert state.steps == cfg['steps']
    return a, state, history
