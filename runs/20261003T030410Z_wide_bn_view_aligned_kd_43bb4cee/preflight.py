"""Frozen-input and CUDA execution checks for the view-aligned KD batch."""
import json, os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import MultiViewModel, span_mask
from common import RUN, config, deterministic, sha


def load_teacher(seed, device):
    model = MultiViewModel('ce_only').to(device)
    ck = torch.load(RUN / 'artifacts' / f'teacher_checkpoint_{seed}.pt',
                    weights_only=True, map_location=device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    model.requires_grad_(False)
    assert all(not p.requires_grad for p in model.parameters())
    assert all(not m.training for m in model.modules())
    return model


def main():
    c = config()
    assert torch.cuda.is_available()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    deterministic('cuda', 2)
    device = torch.device('cuda')
    raw = torch.load(RUN / 'artifacts' / 'prepared.pt', weights_only=False,
                     map_location='cpu')
    assert set(raw) >= {'source', 'valid'}
    assert tuple(raw['source']['tam'].shape) == (c['source_rows'], 2, 1800)
    assert tuple(raw['valid']['tam'].shape) == (c['valid_rows'], 2, 1800)
    assert raw['source']['labels'].shape == (c['source_rows'],)
    assert raw['valid']['labels'].shape == (c['valid_rows'],)

    # Every frozen teacher checkpoint and precomputed clean target is finite
    # and has the declared class dimension.
    teachers = [load_teacher(seed, device) for seed in c['seeds']]
    logits = []
    for seed in c['seeds']:
        with np.load(RUN / 'artifacts' / f'teacher_logits_{seed}.npz') as z:
            assert z['source'].shape == (c['source_rows'], c['num_classes'])
            assert z['valid'].shape == (c['valid_rows'], c['num_classes'])
            assert np.isfinite(z['source']).all() and np.isfinite(z['valid']).all()
            logits.append(z['source'].astype(np.float32, copy=False))
    T = float(c['distillation']['temperature'])
    p_each = []
    for x in logits:
        q = x / T
        q -= q.max(axis=1, keepdims=True)
        q = np.exp(q); q /= q.sum(axis=1, keepdims=True)
        p_each.append(q)
    clean_target = np.mean(np.stack(p_each), axis=0)
    assert clean_target.shape == (c['source_rows'], c['num_classes'])
    assert np.isfinite(clean_target).all()
    np.testing.assert_allclose(clean_target.sum(axis=1), 1.0, rtol=2e-6, atol=2e-6)

    x = raw['source']['timestamps'][:64].to(device)
    clean = raw['source']['tam'][:64].to(device)
    mask_rng = torch.Generator().manual_seed(91731)
    masked = span_mask(clean, mask_rng)
    assert torch.equal(masked, span_mask(clean, torch.Generator().manual_seed(91731)))

    # Matched-view contract: one exact tensor is passed to all teachers and
    # the student; no teacher buffer may update in eval mode.
    before = [{k: v.detach().clone() for k, v in m.named_buffers()} for m in teachers]
    with torch.inference_mode():
        teacher_out = [m(x, masked) for m in teachers]
    assert all(o.shape == (64, c['num_classes']) and torch.isfinite(o).all()
               for o in teacher_out)
    for m, old in zip(teachers, before):
        assert all(torch.equal(old[k], v) for k, v in m.named_buffers())
    probs = torch.stack([F.softmax(o / T, dim=1) for o in teacher_out], dim=0).mean(0)
    confidence = probs.max(1).values
    votes = torch.stack([o.argmax(1) for o in teacher_out], dim=0)
    disagreement = (votes != votes[0:1]).any(0)
    gate = disagreement & (confidence >= float(c['distillation']['confidence_threshold']))
    assert gate.dtype == torch.bool and gate.shape == (64,)
    if int(gate.sum()) > 0:
        assert torch.isfinite(probs[gate]).all()

    # Actual student forward/backward, including generator gradients.
    student = MultiViewModel('ce_only').to(device)
    y = raw['source']['labels'][:64].to(device)
    out = student(x, masked)
    assert out.shape == (64, c['num_classes']) and torch.isfinite(out).all()
    ce = F.cross_entropy(out, y)
    kd = F.kl_div(F.log_softmax(out / T, dim=1), probs,
                  reduction='batchmean') * (T * T)
    loss = ce + float(c['distillation']['weight']) * kd
    assert torch.isfinite(loss)
    loss.backward()
    active = [p for n, p in student.named_parameters()
              if (n.startswith('generator.stages') or n == 'generator.to_features.weight')
              and n.endswith('weight')]
    assert active and all(p.grad is not None and torch.isfinite(p.grad).all()
                          and float(p.grad.norm()) > 0 for p in active)

    result = {
        'passed': True, 'cuda_available': True,
        'gpu_name': torch.cuda.get_device_name(0),
        'source_rows': c['source_rows'], 'valid_rows': c['valid_rows'],
        'teacher_files': len(teachers), 'teacher_eval_frozen': True,
        'clean_target_shape': list(clean_target.shape),
        'clean_target_probability_mean': True,
        'matched_input_equal': True, 'matched_teacher_outputs_finite': True,
        'gated_condition_runnable': True, 'gate_count_probe': int(gate.sum()),
        'student_forward_backward': True, 'generator_gradients_active': True,
        'future_access': False,
    }
    (RUN / 'artifacts' / 'preflight.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
