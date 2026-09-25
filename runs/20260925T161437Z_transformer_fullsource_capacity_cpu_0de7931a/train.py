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


def metric(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    count = 102
    confusion = np.bincount(labels * count + prediction, minlength=count * count).reshape(count, count)
    tp = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(count), where=denominator != 0)
    return {'accuracy': float((labels == prediction).mean()), 'macro_f1': float(f1.mean())}


def load_data(config: dict) -> dict:
    path = ROOT / config['source_artifact']
    if hashlib.sha256(path.read_bytes()).hexdigest() != config['source_artifact_sha256']:
        raise ValueError('fixed prepared artifact SHA-256 mismatch')
    data = torch.load(path, map_location='cpu', weights_only=False)
    if set(data) != {'source', 'valid', 'fewshot', 'source_pretrain'}:
        raise ValueError('prepared artifact roles changed')
    if len(data['source']['labels']) != config['train_label_count'] or len(data['valid']['labels']) != config['valid_label_count']:
        raise ValueError('source/valid count mismatch')
    return data


def make_batch(values: dict[str, torch.Tensor]) -> GeneratorTokenBatch:
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


def make_model(condition: str) -> torch.nn.Module:
    if condition == 'flat_multiview':
        return GeneratorTokenTransformer(d_model=64, nhead=4, layers=2, dim_feedforward=128,
                                         max_tokens=512, num_classes=102, adapter_bottleneck=16)
    if condition == 'hierarchical_multiview':
        return HierarchicalViewTransformer(d_model=52, nhead=4, dim_feedforward=104, num_classes=102)
    raise ValueError(condition)


def logits_for(model: torch.nn.Module, batch: GeneratorTokenBatch) -> torch.Tensor:
    output = model(batch)
    return output['logits'] if isinstance(output, dict) else output


def shuffled_batches(size: int, batch_size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator)
    for start in range(0, size, batch_size):
        yield order[start:start + batch_size]


def predict(model: torch.nn.Module, batch: GeneratorTokenBatch, batch_size: int = 64) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, batch.packet.shape[0], batch_size):
            indices = torch.arange(start, min(start + batch_size, batch.packet.shape[0]))
            predictions.append(logits_for(model, subset(batch, indices)).argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def train(condition: str) -> None:
    config = json.loads((RUN / 'config.json').read_text())
    if config['status'] != 'frozen' or config['device'] != 'cpu' or condition not in config['conditions']:
        raise RuntimeError('frozen CPU config and listed condition required')
    torch.set_num_threads(config['threads_per_condition'])
    prepared = load_data(config)
    source = make_batch(prepared['source']['values'])
    valid = make_batch(prepared['valid']['values'])
    source_y = prepared['source']['labels'].numpy()
    valid_y = prepared['valid']['labels'].numpy()
    started = time.monotonic()
    for seed in config['seeds']:
        key = f'{condition}_{seed}'
        checkpoint_path = RUN / 'checkpoints' / f'{key}.pt'
        history_path = RUN / 'artifacts' / f'history_{key}.json'
        metric_path = RUN / 'artifacts' / f'metrics_{key}.json'
        prediction_path = RUN / 'artifacts' / f'predictions_{key}.npz'
        paths = (checkpoint_path, history_path, metric_path, prediction_path)
        if all(path.exists() for path in paths):
            print(f'skip completed {key}', flush=True)
            continue
        if any(path.exists() for path in paths):
            raise RuntimeError(f'partial seed artifacts require integrity review: {key}')
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = make_model(condition)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        history = []
        best_f1 = -1.0
        best_epoch = 0
        best_state = None
        for epoch in range(1, config['finetune_epochs'] + 1):
            model.train()
            losses = []
            correct = 0
            for indices in shuffled_batches(len(source_y), config['batch_size'], seed + 1000 + epoch):
                batch = subset(source, indices)
                target = prepared['source']['labels'][indices]
                optimizer.zero_grad(set_to_none=True)
                logits = logits_for(model, batch)
                loss = F.cross_entropy(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{key} epoch {epoch} nonfinite loss')
                correct += int((logits.argmax(1) == target).sum())
                losses.append(float(loss.detach()))
                loss.backward()
                optimizer.step()
            record = {'epoch': epoch, 'train_loss': float(np.mean(losses)),
                      'online_train_accuracy': correct / len(source_y)}
            if epoch % config['eval_every'] == 0:
                train_prediction = predict(model, source)
                valid_prediction = predict(model, valid)
                train_score = metric(source_y, train_prediction)
                valid_score = metric(valid_y, valid_prediction)
                record.update({'train_eval_accuracy': train_score['accuracy'],
                               'train_eval_macro_f1': train_score['macro_f1'],
                               'valid_accuracy': valid_score['accuracy'],
                               'valid_macro_f1': valid_score['macro_f1'],
                               'valid_predicted_classes': int(len(np.unique(valid_prediction)))})
                if valid_score['macro_f1'] > best_f1:
                    best_f1 = valid_score['macro_f1']
                    best_epoch = epoch
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            history.append(record)
            if epoch % 10 == 0:
                print(f'{key} epoch {epoch}: train_acc={record["train_eval_accuracy"]:.4f} '
                      f'valid_acc={record["valid_accuracy"]:.4f} valid_f1={record["valid_macro_f1"]:.4f}', flush=True)
            if time.monotonic() - started > config['time_limit_seconds_per_condition']:
                raise TimeoutError(f'{condition} time budget exceeded before seed completion')
        if best_state is None:
            raise AssertionError('no validation checkpoints')
        model.load_state_dict(best_state)
        train_prediction = predict(model, source)
        valid_prediction = predict(model, valid)
        report = {'condition': condition, 'seed': seed, 'best_epoch': best_epoch,
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'source': metric(source_y, train_prediction), 'valid': metric(valid_y, valid_prediction),
                  'valid_predicted_classes': int(len(np.unique(valid_prediction)))}
        temporary = checkpoint_path.with_suffix('.pt.tmp')
        torch.save({'condition': condition, 'seed': seed, 'best_epoch': best_epoch,
                    'state_dict': best_state}, temporary)
        temporary.replace(checkpoint_path)
        history_path.write_text(json.dumps({'history': history, 'best_epoch': best_epoch}, indent=2) + '\n')
        metric_path.write_text(json.dumps(report, indent=2) + '\n')
        np.savez_compressed(prediction_path, source=train_prediction, valid=valid_prediction)
        print(f'{key}: best valid accuracy={report["valid"]["accuracy"]:.4f} '
              f'F1={best_f1:.4f} epoch={best_epoch}', flush=True)
    print(f'COMPLETE {condition}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=('flat_multiview', 'hierarchical_multiview'), required=True)
    args = parser.parse_args()
    train(args.condition)


if __name__ == '__main__':
    main()
