from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ta_wf_next.hierarchical_transformer import HierarchicalViewTransformer
from ta_wf_next.transformer_proto import GeneratorTokenBatch, GeneratorTokenTransformer

SOURCE_PREPARED = ROOT / 'runs/20260925T130044Z_transformer_temporal_pretrain_fewshot_7098625e/artifacts/prepared.pt'
SEEDS = (1729, 3407, 2026)


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    cm = np.bincount(y * 102 + prediction, minlength=102 * 102).reshape(102, 102)
    tp = np.diag(cm).astype(np.float64)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros(102), where=den != 0)
    return {'accuracy': float((y == prediction).mean()), 'macro_f1': float(f1.mean())}


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


def load_prepared() -> dict[str, object]:
    local = RUN / 'artifacts' / 'prepared.pt'
    if hashlib.sha256(local.read_bytes()).hexdigest() != hashlib.sha256(SOURCE_PREPARED.read_bytes()).hexdigest():
        raise ValueError('prepared artifact differs from fixed source/valid data')
    prepared = torch.load(local, map_location='cpu', weights_only=False)
    if set(prepared) != {'source', 'valid', 'fewshot', 'source_pretrain'}:
        raise ValueError('unexpected prepared roles')
    return prepared


def make_model(condition: str) -> torch.nn.Module:
    if condition == 'hierarchical_multiview':
        return HierarchicalViewTransformer(d_model=52, nhead=4, dim_feedforward=104, num_classes=102)
    if condition == 'flat_multiview':
        return GeneratorTokenTransformer(d_model=64, nhead=4, layers=2, dim_feedforward=128,
                                         max_tokens=512, num_classes=102, adapter_bottleneck=16)
    raise ValueError(condition)


def batch_for(prepared: dict[str, object], role: str) -> tuple[GeneratorTokenBatch, np.ndarray]:
    entry = prepared[role]
    return from_tensor_dict(entry['values']), entry['labels'].numpy()


def batched_indices(indices: np.ndarray, batch_size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    order = torch.as_tensor(indices, dtype=torch.long)
    order = order[torch.randperm(len(order), generator=generator)]
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size].numpy()


def predict(model: torch.nn.Module, batch: GeneratorTokenBatch, batch_size: int = 64) -> np.ndarray:
    model.eval(); values = []
    with torch.inference_mode():
        for start in range(0, batch.packet.shape[0], batch_size):
            output = model(subset(batch, np.arange(start, min(start + batch_size, batch.packet.shape[0]))))
            logits = output['logits'] if isinstance(output, dict) else output
            values.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(values)


def fit_step(model: torch.nn.Module, batch: GeneratorTokenBatch, labels: torch.Tensor,
             optimizer: torch.optim.Optimizer) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    logits = output['logits'] if isinstance(output, dict) else output
    loss = F.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite supervised loss')
    loss.backward()
    optimizer.step()
    return {'loss': float(loss.detach()), 'batch_accuracy': float((logits.argmax(1) == labels).float().mean())}


def train_condition(condition: str) -> None:
    config = json.loads((RUN / 'config.json').read_text())
    if config.get('status') != 'frozen' or condition not in config['conditions']:
        raise RuntimeError('frozen config and listed condition are required')
    prepared = load_prepared()
    torch.set_num_threads(config['threads'])
    source_batch, source_y = batch_for(prepared, 'source')
    valid_batch, valid_y = batch_for(prepared, 'valid')
    fewshot = prepared['fewshot'].numpy()
    train_batch = subset(source_batch, fewshot)
    train_y = source_y[fewshot]
    labels = torch.from_numpy(train_y)
    started = time.monotonic()
    for seed in config['seeds']:
        key = f'{condition}_{seed}'
        checkpoint_path = RUN / 'checkpoints' / f'{key}.pt'
        history_path = RUN / 'artifacts' / f'history_{key}.json'
        metric_path = RUN / 'artifacts' / f'metrics_{key}.json'
        prediction_path = RUN / 'artifacts' / f'predictions_{key}.npz'
        paths = (checkpoint_path, history_path, metric_path, prediction_path)
        if all(path.exists() for path in paths):
            print(f'skip completed {key}', flush=True); continue
        if any(path.exists() for path in paths):
            raise RuntimeError(f'partial artifacts require review: {key}')
        np.random.seed(seed); torch.manual_seed(seed)
        model = make_model(condition)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        history = []
        best_f1 = -1.0; best_epoch = 0; best_state = None
        for epoch in range(1, config['finetune_epochs'] + 1):
            records = []
            for idx in batched_indices(np.arange(len(fewshot)), config['batch_size'], seed + 1000 + epoch):
                records.append(fit_step(model, subset(train_batch, idx), labels[idx], optimizer))
            record = {'epoch': epoch, 'train_loss': float(np.mean([x['loss'] for x in records])),
                      'online_train_accuracy': float(np.mean([x['batch_accuracy'] for x in records]))}
            if epoch % config['eval_every'] == 0:
                train_prediction = predict(model, train_batch)
                valid_prediction = predict(model, valid_batch)
                train_score = metric(train_y, train_prediction)
                valid_score = metric(valid_y, valid_prediction)
                record.update({'train_eval_accuracy': train_score['accuracy'],
                               'train_eval_macro_f1': train_score['macro_f1'],
                               'valid_accuracy': valid_score['accuracy'],
                               'valid_macro_f1': valid_score['macro_f1'],
                               'valid_predicted_classes': int(len(np.unique(valid_prediction)))})
                if valid_score['macro_f1'] > best_f1:
                    best_f1 = valid_score['macro_f1']; best_epoch = epoch
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            history.append(record)
            if epoch % 10 == 0:
                print(f'{key} epoch {epoch}: train_eval_acc={record["train_eval_accuracy"]:.4f} '
                      f'valid_f1={record["valid_macro_f1"]:.4f}', flush=True)
            if time.monotonic() - started > config['time_limit_seconds']:
                raise TimeoutError('experiment time limit exceeded')
        assert best_state is not None
        model.load_state_dict(best_state)
        train_prediction = predict(model, train_batch)
        valid_prediction = predict(model, valid_batch)
        report = {'condition': condition, 'seed': seed, 'best_epoch': best_epoch,
                  'parameters': sum(parameter.numel() for parameter in model.parameters()),
                  'train': metric(train_y, train_prediction), 'valid': metric(valid_y, valid_prediction),
                  'train_predicted_classes': int(len(np.unique(train_prediction))),
                  'valid_predicted_classes': int(len(np.unique(valid_prediction)))}
        tmp = checkpoint_path.with_suffix('.pt.tmp')
        torch.save({'state_dict': best_state, 'condition': condition, 'seed': seed,
                    'best_epoch': best_epoch}, tmp); tmp.replace(checkpoint_path)
        history_path.write_text(json.dumps({'history': history, 'best_epoch': best_epoch}, indent=2) + '\n')
        metric_path.write_text(json.dumps(report, indent=2) + '\n')
        np.savez_compressed(prediction_path, train=train_prediction, valid=valid_prediction)
        print(f'{key}: best valid F1 {best_f1:.5f} epoch {best_epoch}', flush=True)
    print(f'COMPLETE {condition}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=('hierarchical_multiview', 'flat_multiview'), required=True)
    args = parser.parse_args()
    train_condition(args.condition)


if __name__ == '__main__':
    main()
