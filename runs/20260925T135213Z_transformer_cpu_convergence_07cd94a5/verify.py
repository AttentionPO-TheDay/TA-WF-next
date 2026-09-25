from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
import train


def main() -> None:
    config = json.loads((RUN / 'config.json').read_text())
    prepared = train.load_prepared()
    fewshot = prepared['fewshot'].numpy()
    source_y = prepared['source']['labels'].numpy()
    assert len(fewshot) == 510
    assert np.all(np.bincount(source_y[fewshot], minlength=102) == 5)
    assert len(prepared['source_pretrain']) == 2040
    assert set(prepared) == {'source', 'valid', 'fewshot', 'source_pretrain'}
    torch.set_num_threads(config['threads'])
    errors = []
    checked = 0
    parameters = set()
    for condition in config['conditions']:
        batches = {role: train.batch_for(prepared, role, condition) for role in ('source', 'valid')}
        train_batch = train.subset(batches['source'][0], fewshot)
        train_y = batches['source'][1][fewshot]
        valid_batch, valid_y = batches['valid']
        for seed in config['seeds']:
            key = f'{condition}_{seed}'
            checkpoint = RUN / 'checkpoints' / f'{key}.pt'
            history_path = RUN / 'artifacts' / f'history_{key}.json'
            metric_path = RUN / 'artifacts' / f'metrics_{key}.json'
            prediction_path = RUN / 'artifacts' / f'predictions_{key}.npz'
            if not all(path.is_file() for path in (checkpoint, history_path, metric_path, prediction_path)):
                errors.append(f'{key}:missing')
                continue
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = train.make_model().eval()
            model.load_state_dict(state['state_dict'])
            parameters.add(sum(parameter.numel() for parameter in model.parameters()))
            history = json.loads(history_path.read_text())
            report = json.loads(metric_path.read_text())
            evaluations = [row for row in history['history'] if 'valid_macro_f1' in row]
            if [row['epoch'] for row in evaluations] != list(range(5, 101, 5)):
                errors.append(f'{key}:evaluation-schedule')
            earliest = max(evaluations, key=lambda row: row['valid_macro_f1'])['epoch']
            if state['best_epoch'] != earliest or report['best_epoch'] != earliest:
                errors.append(f'{key}:selection')
            predictions = np.load(prediction_path)
            for role, batch, labels in (('train', train_batch, train_y), ('valid', valid_batch, valid_y)):
                actual = train.predict(model, batch, torch.device('cpu'))
                if not np.array_equal(actual, predictions[role]):
                    errors.append(f'{key}:{role}:prediction')
                metric = train.metric(labels, actual)
                if any(abs(metric[name] - report[role][name]) > 1e-12 for name in metric):
                    errors.append(f'{key}:{role}:metric')
                if len(np.unique(actual)) != report[f'{role}_predicted_classes']:
                    errors.append(f'{key}:{role}:classes')
                checked += 1
            predictions.close()
    if parameters and parameters != {110400}:
        errors.append('parameters')
    digest = hashlib.sha256((RUN / 'artifacts' / 'prepared.pt').read_bytes()).hexdigest()
    result = {'prediction_sets_checked': checked, 'expected_prediction_sets': 18,
              'errors': errors, 'prepared_sha256': digest, 'parameters': sorted(parameters)}
    (RUN / 'artifacts' / 'integrity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    assert not errors and checked == 18


if __name__ == '__main__':
    main()
