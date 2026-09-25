from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import (
    GeneratorTokenBatch, GeneratorTokenTransformer, batch_from_views,
    finetune_step, pretrain_step,
)

PRIOR = ROOT / 'runs/20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37/artifacts/manifest.json'
DATA_ROOT = Path(json.loads((ROOT / 'configs/datasets.json').read_text())['data_root'])
ROLES = ('source', 'valid', 'day14', 'day30', 'day90', 'day150', 'day270')
SEEDS = (1729, 3407, 2026)


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    cm = np.bincount(y * 102 + prediction, minlength=102 * 102).reshape(102, 102)
    tp = np.diag(cm).astype(np.float64)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros(102), where=den != 0)
    return {'accuracy': float((y == prediction).mean()), 'macro_f1': float(f1.mean())}


def views_to_batch(rows: np.ndarray) -> GeneratorTokenBatch:
    views = tuple(generate_views(row[:5000], input_kind='signed_timestamp', budget=5000,
                                 window_sizes=(50, 250)) for row in rows)
    return batch_from_views(views, packet_patch=50, max_runs=128)


def tensor_dict(batch: GeneratorTokenBatch) -> dict[str, torch.Tensor]:
    result = {name: getattr(batch, name) for name in
              ('packet', 'packet_mask', 'runs', 'runs_mask', 'windows', 'windows_mask')}
    result.update({f'spans_{name}': batch.spans[name] for name in ('packet', 'runs', 'windows')})
    return result


def from_tensor_dict(values: dict[str, torch.Tensor]) -> GeneratorTokenBatch:
    return GeneratorTokenBatch(
        values['packet'], values['packet_mask'], values['runs'], values['runs_mask'],
        values['windows'], values['windows_mask'],
        {name: values[f'spans_{name}'] for name in ('packet', 'runs', 'windows')},
    )


def subset(batch: GeneratorTokenBatch, indices: np.ndarray | torch.Tensor) -> GeneratorTokenBatch:
    idx = torch.as_tensor(indices, dtype=torch.long)
    return GeneratorTokenBatch(
        *(getattr(batch, name)[idx] for name in
          ('packet', 'packet_mask', 'runs', 'runs_mask', 'windows', 'windows_mask')),
        {name: batch.spans[name][idx] for name in ('packet', 'runs', 'windows')},
    )


def packet_only(batch: GeneratorTokenBatch) -> GeneratorTokenBatch:
    return GeneratorTokenBatch(
        batch.packet, batch.packet_mask,
        torch.zeros_like(batch.runs), torch.zeros_like(batch.runs_mask),
        torch.zeros_like(batch.windows), torch.zeros_like(batch.windows_mask),
        batch.spans,
    )


def prepare() -> None:
    output = RUN / 'artifacts' / 'prepared.pt'
    if output.exists():
        return
    manifest = json.loads(PRIOR.read_text())['sampling']
    prepared: dict[str, object] = {}
    direction_hashes = {}
    for role in ('source', 'valid'):
        item = manifest[role]
        with np.load(DATA_ROOT / item['path'], allow_pickle=False) as archive:
            rows = archive['X'][item['rows']].astype(np.float32)
            labels = archive['y'][item['rows']].astype(np.int64)
        if rows.shape != (item['count'], rows.shape[1]) or rows.shape[1] < 5000:
            raise ValueError(f'unexpected {role} shape {rows.shape}')
        batch = views_to_batch(rows)
        prepared[role] = {'values': tensor_dict(batch), 'labels': torch.from_numpy(labels)}
        direction_hashes[role] = {hashlib.sha256(np.sign(row[:5000]).astype(np.int8).tobytes()).hexdigest() for row in rows}
        print(f'prepared {role}: {len(rows)}', flush=True)
    if direction_hashes['source'] & direction_hashes['valid']:
        raise ValueError('source/valid direction overlap')
    source_y = prepared['source']['labels']
    fewshot = []
    for label in range(102):
        fewshot.extend(torch.where(source_y == label)[0][:5].tolist())
    fewshot = np.asarray(fewshot, dtype=np.int64)
    prepared['fewshot'] = torch.from_numpy(fewshot)
    prepared['source_pretrain'] = torch.arange(len(source_y), dtype=torch.long)
    torch.save(prepared, output)
    print(json.dumps({'prepared': str(output), 'fewshot': len(fewshot),
                      'source_pretrain': len(prepared['source_pretrain'])}, indent=2))


def load_prepared() -> dict[str, object]:
    return torch.load(RUN / 'artifacts' / 'prepared.pt', map_location='cpu', weights_only=False)


def make_model() -> GeneratorTokenTransformer:
    return GeneratorTokenTransformer(d_model=64, nhead=4, layers=2, dim_feedforward=128,
                                     max_tokens=512, num_classes=102, adapter_bottleneck=16)


def batch_for(prepared, role: str, condition: str) -> tuple[GeneratorTokenBatch, np.ndarray]:
    entry = prepared[role]
    batch = from_tensor_dict(entry['values'])
    if condition == 'packet_scratch':
        batch = packet_only(batch)
    return batch, entry['labels'].numpy()


def batched_indices(indices: np.ndarray, batch_size: int, seed: int, shuffle: bool = True):
    generator = torch.Generator().manual_seed(seed)
    order = torch.as_tensor(indices, dtype=torch.long)
    if shuffle:
        order = order[torch.randperm(len(order), generator=generator)]
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size].numpy()


def predict(model, batch: GeneratorTokenBatch, device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval(); values = []
    with torch.inference_mode():
        for idx in range(0, batch.packet.shape[0], batch_size):
            output = model(subset(batch, np.arange(idx, min(idx + batch_size, batch.packet.shape[0]))).to(device))
            values.append(output['logits'].argmax(1).cpu().numpy())
    return np.concatenate(values)


def transfer_backbone(model: GeneratorTokenTransformer, state: dict[str, torch.Tensor]) -> None:
    allowed = {name: value for name, value in state.items()
               if not name.startswith('classifier.') and not name.startswith('reconstruction.')}
    missing, unexpected = model.load_state_dict(allowed, strict=False)
    if any(name.startswith('projections.') or name.startswith('encoder.') for name in missing):
        raise AssertionError(f'pretrained backbone missing: {missing}')
    if unexpected:
        raise AssertionError(f'unexpected pretrained keys: {unexpected}')


def train_condition(condition: str, device_name: str) -> None:
    config = json.loads((RUN / 'config.json').read_text())
    if config.get('status') != 'frozen' or not torch.cuda.is_available():
        raise RuntimeError('frozen config and CUDA are required')
    prepared = load_prepared()
    device = torch.device(device_name)
    torch.set_num_threads(config['threads'])
    source_batch, source_y = batch_for(prepared, 'source', condition)
    valid_batch, valid_y = batch_for(prepared, 'valid', condition)
    fewshot = prepared['fewshot'].numpy()
    pretrain_idx = prepared['source_pretrain'].numpy()
    histories: dict[str, object] = {}
    metric_rows = []
    predictions: dict[str, np.ndarray] = {}
    checkpoints = RUN / 'checkpoints'
    started = time.monotonic()
    for seed in SEEDS:
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        pretrained_state = None
        pretrain_history = []
        if condition == 'multiview_pretrained':
            pre_model = make_model().to(device)
            pre_optimizer = torch.optim.AdamW(pre_model.parameters(), lr=0.001, weight_decay=0.0001)
            pre_batch = subset(source_batch, pretrain_idx)
            for epoch in range(1, 9):
                epoch_records = []
                for idx in batched_indices(np.arange(len(pretrain_idx)), 64, seed + epoch):
                    record = pretrain_step(pre_model, subset(pre_batch, idx).to(device), pre_optimizer, span_length=50)
                    epoch_records.append(record)
                pretrain_history.append({'epoch': epoch, 'loss': float(np.mean([x['loss'] for x in epoch_records])),
                                         'masked_tokens': int(sum(x['masked_tokens'] for x in epoch_records))})
            pretrained_state = {name: value.detach().cpu().clone() for name, value in pre_model.state_dict().items()}
            del pre_model, pre_optimizer
            torch.cuda.empty_cache()
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = make_model().to(device)
        if pretrained_state is not None:
            transfer_backbone(model, pretrained_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        train_batch = subset(source_batch, fewshot)
        train_y = torch.from_numpy(source_y[fewshot])
        history = []
        best_f1 = -1.0; best_epoch = 0; best_state = None
        for epoch in range(1, 26):
            records = []
            for idx in batched_indices(np.arange(len(fewshot)), 64, seed + 1000 + epoch):
                labels = train_y[idx]
                records.append(finetune_step(model, subset(train_batch, idx).to(device), labels.to(device), optimizer))
            prediction = predict(model, valid_batch, device)
            score = metric(valid_y, prediction)
            record = {'epoch': epoch, 'train_loss': float(np.mean([x['loss'] for x in records])),
                      'train_accuracy': float(np.mean([x['batch_accuracy'] for x in records])), **score}
            history.append(record)
            if score['macro_f1'] > best_f1:
                best_f1 = score['macro_f1']; best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        assert best_state is not None
        model.load_state_dict(best_state)
        key = f'{condition}_{seed}'
        torch.save({'state_dict': best_state, 'condition': condition, 'seed': seed,
                    'best_epoch': best_epoch, 'pretrain_history': pretrain_history}, checkpoints / f'{key}.pt')
        histories[key] = {'history': history, 'best_epoch': best_epoch, 'pretrain_history': pretrain_history}
        for role in ('source', 'valid'):
            role_batch, role_y = batch_for(prepared, role, condition)
            prediction = predict(model, role_batch, device)
            predictions[f'{role}_{key}'] = prediction
            metric_rows.append({'condition': condition, 'seed': seed, 'role': role,
                                'best_epoch': best_epoch, **metric(role_y, prediction)})
        print(f'{condition} seed {seed}: valid F1 {best_f1:.5f} epoch {best_epoch}', flush=True)
        if time.monotonic() - started > config['time_limit_seconds']:
            raise TimeoutError('experiment time limit exceeded')
    (RUN / 'artifacts' / f'history_{condition}.json').write_text(json.dumps(histories, indent=2))
    with (RUN / 'artifacts' / f'metrics_{condition}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['condition', 'seed', 'role', 'best_epoch', 'accuracy', 'macro_f1'])
        writer.writeheader(); writer.writerows(metric_rows)
    np.savez_compressed(RUN / 'artifacts' / f'predictions_{condition}.npz', **predictions)
    print(f'COMPLETE {condition}', flush=True)


def score_future(device_name: str) -> None:
    """Evaluate only after all nine checkpoints have been selected on valid."""
    config = json.loads((RUN / 'config.json').read_text())
    if config.get('status') != 'frozen':
        raise RuntimeError('frozen config required')
    expected = [RUN / 'checkpoints' / f'{condition}_{seed}.pt'
                for condition in config['conditions'] for seed in config['seeds']]
    if not all(path.exists() for path in expected):
        raise FileNotFoundError('all selected checkpoints are required before future scoring')
    output = RUN / 'artifacts' / 'metrics_future.csv'
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(device_name)
    sampling = json.loads(PRIOR.read_text())['sampling']
    rows = []
    predictions = {}
    for role in ROLES[2:]:
        item = sampling[role]
        path = DATA_ROOT / item['path']
        with np.load(path, allow_pickle=False) as archive:
            raw_x = archive['X'][item['rows']].astype(np.float32)
        batch = views_to_batch(raw_x)
        role_predictions = {}
        for condition in config['conditions']:
            condition_batch = packet_only(batch) if condition == 'packet_scratch' else batch
            for seed in config['seeds']:
                checkpoint = torch.load(RUN / 'checkpoints' / f'{condition}_{seed}.pt',
                                        map_location='cpu', weights_only=True)
                model = make_model().to(device).eval()
                model.load_state_dict(checkpoint['state_dict'])
                key = f'{role}_{condition}_{seed}'
                role_predictions[key] = predict(model, condition_batch, device)
                predictions[key] = role_predictions[key]
        # Query labels are accessed only after every condition has predicted this role.
        with np.load(path, allow_pickle=False) as archive:
            labels = archive['y'][item['rows']].astype(np.int64)
        for condition in config['conditions']:
            for seed in config['seeds']:
                key = f'{role}_{condition}_{seed}'
                checkpoint = torch.load(RUN / 'checkpoints' / f'{condition}_{seed}.pt',
                                        map_location='cpu', weights_only=True)
                rows.append({'condition': condition, 'seed': seed, 'role': role,
                             'best_epoch': checkpoint['best_epoch'],
                             **metric(labels, role_predictions[key])})
        print(f'scored {role}', flush=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['condition', 'seed', 'role', 'best_epoch', 'accuracy', 'macro_f1'])
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(RUN / 'artifacts' / 'predictions_future.npz', **predictions)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--score-future', action='store_true')
    parser.add_argument('--condition', choices=('packet_scratch', 'multiview_scratch', 'multiview_pretrained'))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.score_future:
        score_future(args.device)
        return
    prepare()
    if args.condition:
        train_condition(args.condition, args.device)


if __name__ == '__main__':
    main()
