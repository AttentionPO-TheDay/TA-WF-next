import json, os
from pathlib import Path
import numpy as np
import torch
from model import MultiViewModel, span_mask
from common import RUN, config, deterministic

def main():
    c = config()
    assert torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    raw = torch.load(RUN / 'artifacts/prepared.pt', weights_only=False, map_location='cpu')
    assert raw['source']['tam'].shape == (15300, 2, 1800)
    assert raw['valid']['tam'].shape == (510, 2, 1800)
    teacher = []
    for seed in c['seeds']:
        with np.load(RUN / 'artifacts' / f'teacher_logits_{seed}.npz') as z:
            assert z['source'].shape == (15300, 102) and z['valid'].shape == (510, 102)
            assert np.isfinite(z['source']).all() and np.isfinite(z['valid']).all()
            teacher.append(z['source'])
    avg = np.mean(np.stack(teacher), axis=0)
    assert np.isfinite(avg).all()
    deterministic('cuda', 2)
    torch.manual_seed(21729); torch.cuda.manual_seed_all(21729)
    model = MultiViewModel('ensemble_kd').cuda()
    x, t = raw['source']['timestamps'][:64].cuda(), raw['source']['tam'][:64].cuda()
    with torch.no_grad():
        logits = model(x, t)
        assert logits.shape == (64, 102) and torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(model(x, t), raw['source']['labels'][:64].cuda())
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        if name.startswith('generator.stages') or name == 'generator.to_features.weight':
            assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm()) > 0
    out = {'passed': True, 'cuda_available': True, 'gpu_name': torch.cuda.get_device_name(0), 'source_rows': 15300, 'valid_rows': 510,
           'teacher_files': len(teacher), 'teacher_mean_shape': list(avg.shape), 'future_access': False,
           'model_forward_and_backward': True}
    (RUN / 'artifacts' / 'preflight.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out, ensure_ascii=False))

if __name__ == '__main__': main()
