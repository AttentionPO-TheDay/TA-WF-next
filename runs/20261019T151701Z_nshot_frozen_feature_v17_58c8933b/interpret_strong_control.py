"""Post-completion interpretation of cached predictions; never fits or selects.

Reproduces strong_control_interpretation.json without modifying primary gates.
Run after audit.py. Optional --write updates only this interpretation artifact.
"""
import argparse
import json
import numpy as np
from runner import C, ART, read_npz
from maintenance import frontier, best_under_damage, least_damage_for_recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    assert json.loads((ART / 'audit.json').read_text())['passed']
    scores = {r['key']: r for r in json.loads((ART / 'scores.json').read_text())}
    rows = []
    for seed in C['model_seeds']:
        for role in C['targets']:
            y = read_npz(ART / f'evaluation_labels_{role}.npz')['labels']
            base = read_npz(ART / f'pred_{seed}_{role}_frozen.npz')['predictions']
            correct = int((base == y).sum())
            wrong = len(y) - correct
            for draw in C['support_seeds']:
                for cap in C['nominal_shots']:
                    curves, pair = {}, {}
                    for arm in ['shrink_proto', 'head_l2']:
                        key = f'{seed}_{role}_{draw}_cap{cap}_{arm}'
                        prediction = read_npz(ART / f'pred_{key}.npz')
                        curves[arm] = frontier(y, prediction['predictions'], base, prediction['gain'])
                        pair[arm] = scores[key]
                    head = best_under_damage(curves['head_l2'], int(correct * .01))
                    shrink = best_under_damage(curves['shrink_proto'], head['D'])
                    match = least_damage_for_recovery(curves['shrink_proto'], head['R'])
                    rows.append({'seed': seed, 'role': role, 'draw': draw, 'cap': cap,
                                 'F1_gain_shrink_minus_head_pp': 100 * (pair['shrink_proto']['macro_f1'] - pair['head_l2']['macro_f1']),
                                 'head_at1': head, 'shrink_same_head_D': shrink,
                                 'same_damage_Recovery_gain_pp': 100 * (shrink['R'] - head['R']) / wrong,
                                 'shrink_same_head_R': match,
                                 'same_R_feasible_at_NAR1': bool(match is not None and match['D'] <= correct * .01)})
    summary = []
    for role in C['targets']:
        for cap in C['nominal_shots']:
            rs = [r for r in rows if r['role'] == role and r['cap'] == cap]
            summary.append({'role': role, 'cap': cap,
                            'F1_gain_shrink_minus_head_pp': float(np.mean([r['F1_gain_shrink_minus_head_pp'] for r in rs])),
                            'same_damage_Recovery_gain_pp': float(np.mean([r['same_damage_Recovery_gain_pp'] for r in rs])),
                            'same_R_feasible_states': sum(r['same_R_feasible_at_NAR1'] for r in rs), 'states': len(rs)})
    report = {'role': 'post-completion interpretation only, cached predictions; no fit/selection or primary gate changes',
              'summary': summary, 'cells': rows}
    if args.write:
        (ART / 'strong_control_interpretation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    else:
        assert report == json.loads((ART / 'strong_control_interpretation.json').read_text())
    print(json.dumps({'reproduced': True, 'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
