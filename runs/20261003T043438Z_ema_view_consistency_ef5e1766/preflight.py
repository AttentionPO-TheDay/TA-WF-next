"""Preflight for online EMA consistency; no fixed teacher artifacts are loaded."""
import json, os
import torch
from torch.nn import functional as F
from model import MultiViewModel, span_mask, consistency_weight
from common import RUN, config, deterministic


def main():
    c = config(); assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    deterministic('cuda', 2); device = torch.device('cuda')
    raw = torch.load(RUN / 'artifacts/prepared.pt', weights_only=False, map_location='cpu')
    assert tuple(raw['source']['tam'].shape) == (15300, 2, 1800)
    assert tuple(raw['valid']['tam'].shape) == (510, 2, 1800)
    torch.manual_seed(21729); torch.cuda.manual_seed_all(21729)
    student = MultiViewModel('ce_only').to(device)
    ema = MultiViewModel('ce_only').to(device); ema.load_state_dict(student.state_dict()); ema.requires_grad_(False); ema.eval()
    assert all(not p.requires_grad for p in ema.parameters()) and all(not m.training for m in ema.modules())
    x = raw['source']['timestamps'][:64].to(device); clean = raw['source']['tam'][:64].to(device); y = raw['source']['labels'][:64].to(device)
    masked = span_mask(clean, torch.Generator().manual_seed(91731))
    assert torch.equal(masked, span_mask(clean, torch.Generator().manual_seed(91731)))
    with torch.no_grad():
        same_target = F.softmax(ema(x, masked) / c['ema_consistency']['temperature'], dim=1).detach()
        cross_target = F.softmax(ema(x, clean) / c['ema_consistency']['temperature'], dim=1).detach()
    assert same_target.shape == cross_target.shape == (64, 102)
    assert torch.isfinite(same_target).all() and torch.isfinite(cross_target).all()
    out = student(x, masked); ce = F.cross_entropy(out, y)
    kd = F.kl_div(F.log_softmax(out / 2, dim=1), same_target, reduction='batchmean') * 4
    loss = ce + consistency_weight(1, .25, 3200) * kd; loss.backward()
    active = [p for n, p in student.named_parameters() if (n.startswith('generator.stages') or n == 'generator.to_features.weight') and n.endswith('weight')]
    assert active and all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm()) > 0 for p in active)
    assert all(p.grad is None for p in ema.parameters())
    result = {'passed': True, 'cuda_available': True, 'gpu_name': torch.cuda.get_device_name(0), 'source_rows': 15300, 'valid_rows': 510, 'fixed_teacher_loaded': False, 'ema_eval_frozen': True, 'same_view_target_shape': [64, 102], 'cross_view_target_shape': [64, 102], 'student_forward_backward': True, 'ema_gradient_free': True, 'future_access': False}
    (RUN / 'artifacts/preflight.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
