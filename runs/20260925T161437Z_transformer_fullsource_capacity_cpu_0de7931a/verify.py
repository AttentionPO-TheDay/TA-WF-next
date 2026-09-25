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
    prepared = train.load_data(config)
    torch.set_num_threads(config['threads_per_condition'])
    source = train.make_batch(prepared['source']['values'])
    valid = train.make_batch(prepared['valid']['values'])
    source_y = prepared['source']['labels'].numpy()
    valid_y = prepared['valid']['labels'].numpy()
    errors = []
    checked = 0
    parameter_counts = {}
    for condition in config['conditions']:
        parameter_counts[condition] = set()
        for seed in config['seeds']:
            key = f'{condition}_{seed}'
            checkpoint_path = RUN / 'checkpoints' / f'{key}.pt'
            history_path = RUN / 'artifacts' / f'history_{key}.json'
            metric_path = RUN / 'artifacts' / f'metrics_{key}.json'
            predictions_path = RUN / 'artifacts' / f'predictions_{key}.npz'
            if not all(path.is_file() for path in (checkpoint_path, history_path, metric_path, predictions_path)):
                errors.append(f'{key}:missing')
                continue
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model = train.make_model(condition).eval()
            model.load_state_dict(state['state_dict'])
            parameter_counts[condition].add(sum(p.numel() for p in model.parameters()))
            history = json.loads(history_path.read_text())
            report = json.loads(metric_path.read_text())
            evals = [row for row in history['history'] if 'valid_macro_f1' in row]
            if [row['epoch'] for row in evals] != list(range(5, 101, 5)):
                errors.append(f'{key}:evaluation-schedule')
            best = max(evals, key=lambda row: row['valid_macro_f1'])['epoch']
            if best != history['best_epoch'] or best != report['best_epoch'] or best != state['best_epoch']:
                errors.append(f'{key}:checkpoint-selection')
            saved = np.load(predictions_path)
            for role, batch, labels in (('source', source, source_y), ('valid', valid, valid_y)):
                prediction = train.predict(model, batch)
                if not np.array_equal(prediction, saved[role]):
                    errors.append(f'{key}:{role}:prediction')
                score = train.metric(labels, prediction)
                if any(abs(score[name] - report[role][name]) > 1e-12 for name in score):
                    errors.append(f'{key}:{role}:metric')
                checked += 1
            saved.close()
    digest = hashlib.sha256((train.ROOT / config['source_artifact']).read_bytes()).hexdigest()
    result = {'prediction_sets_checked': checked, 'expected_prediction_sets': 12,
              'errors': errors, 'source_artifact_sha256': digest,
              'parameters': {key: sorted(value) for key, value in parameter_counts.items()}}
    (RUN / 'artifacts' / 'integrity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    assert not errors and checked == 12


if __name__ == '__main__':
    main()
