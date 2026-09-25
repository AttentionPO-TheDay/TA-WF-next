from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import (
    GeneratorTokenTransformer, batch_from_views, finetune_step,
    freeze_except_adapter, mask_observation_spans, pretrain_step,
    tta_adapter_step,
)


def state_hash(model, *, exclude_adapter=False):
    h = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        if exclude_adapter and name.startswith('adapter.'):
            continue
        h.update(name.encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    config = json.loads((RUN / 'config.json').read_text())
    if config.get('status') != 'frozen' or config.get('device') != 'cpu':
        raise RuntimeError('requires frozen CPU smoke configuration')
    out = RUN / 'artifacts' / 'smoke.json'
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    data_config = json.loads((ROOT / 'configs/datasets.json').read_text())
    path = Path(data_config['data_root']) / data_config['datasets']['proteus_temporal']['path'] / 'train.npz'
    indices = np.asarray(config['rows'], dtype=np.int64)
    with np.load(path, allow_pickle=False) as archive:
        rows = archive['X'][indices]
        labels = archive['y'][indices].astype(np.int64)
    assert rows.shape[1] >= 5000 and len(rows) == 4
    views = tuple(generate_views(row[:5000], input_kind='signed_timestamp', budget=5000,
                                 window_sizes=(50, 250)) for row in rows)
    batch = batch_from_views(views, packet_patch=50, max_runs=128)
    model = GeneratorTokenTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pretrain = pretrain_step(model, batch, optimizer, span_length=50)
    finetune = finetune_step(model, batch, torch.from_numpy(labels), optimizer)
    teacher = copy.deepcopy(model)
    student = copy.deepcopy(model)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    freeze_except_adapter(student)
    teacher_before = state_hash(teacher)
    backbone_before = state_hash(student, exclude_adapter=True)
    adapter_before = state_hash(student.adapter)
    perturbed, selected = mask_observation_spans(
        batch, torch.tensor([0, 50, 100, 150]), torch.full((4,), 50),
    )
    tta_optimizer = torch.optim.AdamW(student.adapter.parameters(), lr=1e-3)
    tta = tta_adapter_step(student, teacher, batch, perturbed, tta_optimizer,
                           confidence_threshold=0.8, consistency_weight=0.1)
    result = {
        'source_rows': indices.tolist(),
        'source_label_use': 'finetune step only',
        'packet_token_shape': list(batch.packet.shape),
        'run_token_shape': list(batch.runs.shape),
        'window_token_shape': list(batch.windows.shape),
        'observed_counts': [v.runs.observed_count for v in views],
        'pretrain': pretrain,
        'finetune': finetune,
        'tta': tta,
        'tta_masked_tokens': {name: int(mask.sum()) for name, mask in selected.items()},
        'teacher_unchanged': teacher_before == state_hash(teacher),
        'student_non_adapter_unchanged': backbone_before == state_hash(student, exclude_adapter=True),
        'student_adapter_changed': adapter_before != state_hash(student.adapter),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'checkpoint_saved': False,
        'performance_scored': False,
    }
    if not all(result[k] for k in ('teacher_unchanged', 'student_non_adapter_unchanged', 'student_adapter_changed')):
        raise AssertionError('TTA freeze/update invariant failed')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
