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
    assert len(fewshot) == 510 and np.all(np.bincount(source_y[fewshot], minlength=102) == 5)
    errors = []
    checked = 0
    parameters = set()
    for condition in config['conditions']:
        source_batch, source_labels = train.batch_for(prepared, 'source')
        valid_batch, valid_labels = train.batch_for(prepared, 'valid')
        train_batch = train.subset(source_batch, fewshot)
        train_labels = source_labels[fewshot]
        for seed in config['seeds']:
            key = f'{condition}_{seed}'
            checkpoint_path = RUN / 'checkpoints' / f'{key}.pt'
            history_path = RUN / 'artifacts' / f'history_{key}.json'
            metric_path = RUN / 'artifacts' / f'metrics_{key}.json'
            prediction_path = RUN / 'artifacts' / f'predictions_{key}.npz'
            if not all(path.is_file() for path in (checkpoint_path, history_path, metric_path, prediction_path)):
                errors.append(f'{key}:missing')
                continue
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model = train.make_model(condition).eval()
            model.load_state_dict(state['state_dict'])
            parameters.add(sum(p.numel() for p in model.parameters()))
            history = json.loads(history_path.read_text())
            report = json.loads(metric_path.read_text())
            evaluations = [row for row in history['history'] if 'valid_macro_f1' in row]
            if [row['epoch'] for row in evaluations] != list(range(5, 101, 5)):
                errors.append(f'{key}:schedule')
            expected_epoch = max(evaluations, key=lambda row: row['valid_macro_f1'])['epoch']
            if state['best_epoch'] != expected_epoch or report['best_epoch'] != expected_epoch:
                errors.append(f'{key}:selection')
            saved = np.load(prediction_path)
            for role, batch, labels in (('train', train_batch, train_labels), ('valid', valid_batch, valid_labels)):
                prediction = train.predict(model, batch)
                if not np.array_equal(prediction, saved[role]):
                    errors.append(f'{key}:{role}:prediction')
                scores = train.metric(labels, prediction)
                if any(abs(scores[name] - report[role][name]) > 1e-12 for name in scores):
                    errors.append(f'{key}:{role}:metric')
                checked += 1
            saved.close()
    digest = hashlib.sha256((RUN / 'artifacts' / 'prepared.pt').read_bytes()).hexdigest()
    result = {'prediction_sets_checked': checked, 'expected_prediction_sets': 6,
              'errors': errors, 'prepared_sha256': digest, 'parameters': sorted(parameters)}
    (RUN / 'artifacts' / 'integrity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    assert not errors and checked == 6


if __name__ == '__main__':
    main()
