"""Launch and aggregate the frozen EMA consistency batch."""
import json, os, signal, subprocess, sys, time, traceback
import numpy as np
import torch
from common import RUN, ROOT, config, verify_freeze, save


def update(state, summary):
    subprocess.run([sys.executable, str(ROOT / 'scripts/experiment.py'), 'status', '--id', RUN.name, '--state', state, '--summary', summary], check=True)


def main():
    c = config(); verify_freeze(); assert torch.cuda.is_available()
    seeds, kinds = c['seeds'], c['conditions']; todo = [(k, s) for s in seeds for k in kinds]
    active, done = [], []; start = time.monotonic()
    update('running', 'GPU0正式训练启动；CE/EMA同视图/EMA跨视图×3seed，未来关闭。')
    try:
        while todo or active:
            if time.monotonic() - start > c['pipeline_seconds']:
                raise TimeoutError('whole pipeline budget exceeded')
            while todo and len(active) < c['max_gpu_parallel']:
                kind, seed = todo.pop(0); key = f'{kind}_{seed}'
                log = (RUN / 'logs' / f'{key}.log').open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                p = subprocess.Popen([sys.executable, '-u', str(RUN / 'worker.py'), '--kind', kind, '--seed', str(seed), '--device', 'cuda'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((key, p, log, time.monotonic()))
            for item in list(active):
                key, p, log, launched = item
                if p.poll() is not None:
                    log.close(); active.remove(item)
                    if p.returncode != 0: raise RuntimeError(f'{key} failed with {p.returncode}')
                    assert (RUN / 'artifacts' / key / 'report.json').exists(); done.append(key)
                elif time.monotonic() - launched > c['gpu_job_seconds'] + 300:
                    raise TimeoutError(f'{key} watchdog exceeded')
            save(RUN / 'artifacts/progress.json', {'status': 'running', 'active': [x[0] for x in active], 'completed': done, 'pending': [f'{k}_{s}' for k, s in todo], 'elapsed_seconds': time.monotonic() - start})
            time.sleep(5)
        reports = [json.loads((RUN / 'artifacts' / key / 'report.json').read_text()) for key in done]
        means = {k: {m: float(np.mean([r['best']['valid'][m] for r in reports if r['kind'] == k])) for m in ('accuracy', 'macro_f1')} for k in kinds}
        comparisons = {}
        for k in kinds:
            if k == 'ce_only': continue
            da = [next(r['best']['valid']['accuracy'] for r in reports if r['kind'] == k and r['seed'] == s) - next(r['best']['valid']['accuracy'] for r in reports if r['kind'] == 'ce_only' and r['seed'] == s) for s in seeds]
            df = [next(r['best']['valid']['macro_f1'] for r in reports if r['kind'] == k and r['seed'] == s) - next(r['best']['valid']['macro_f1'] for r in reports if r['kind'] == 'ce_only' and r['seed'] == s) for s in seeds]
            comparisons[f'{k} - ce_only'] = {'accuracy_mean_delta_pp': float(np.mean(da) * 100), 'accuracy_per_seed_pp': [float(v * 100) for v in da], 'macro_f1_mean_delta_pp': float(np.mean(df) * 100), 'positive_seed_count': int(sum(v > 0 for v in da)), 'pass': bool(np.mean(da) >= .01 and min(da) > 0 and np.mean(df) >= 0)}
        summary = {'means': means, 'comparisons': comparisons, 'reports': reports, 'future_access': False, 'training_tasks_completed': len(reports), 'fixed_teacher_loaded': False, 'valid_selection': 'predeclared checkpoint selection only'}
        save(RUN / 'artifacts/summary.json', summary)
        rows = ['# EMA 视图一致性结果', '', 'GPU0 三条件×三 seed 正式训练完成。', '', '| 条件 | valid accuracy | valid Macro-F1 | ≥90% |', '|---|---:|---:|---|']
        for k in kinds: rows.append(f"| {k} | {means[k]['accuracy']*100:.3f}% | {means[k]['macro_f1']*100:.3f}% | {means[k]['accuracy'] >= .9} |")
        rows += ['', '```json', json.dumps(comparisons, ensure_ascii=False, indent=2), '```', '', '本轮无固定教师 checkpoint、无教师 logits；EMA 副本仅由各学生在线更新。未来/WTT-Time/AWF/TTA关闭。']
        (RUN / 'RESULTS.md').write_text('\n'.join(rows) + '\n')
        save(RUN / 'artifacts/progress.json', {'status': 'completed', 'completed': done, 'elapsed_seconds': time.monotonic() - start})
        update('completed', '9项GPU训练完成；CE/EMA同视图/EMA跨视图汇总见RESULTS.md；未来关闭。')
    except BaseException:
        for key, p, log, _ in active:
            if p.poll() is None: os.killpg(p.pid, signal.SIGTERM)
            log.close()
        err = traceback.format_exc(); (RUN / 'logs/error.txt').write_text(err); save(RUN / 'artifacts/progress.json', {'status': 'failed', 'completed': done, 'error': err}); update('failed', 'GPU0训练中止，部分产物保留，见logs/error.txt。'); raise


if __name__ == '__main__': main()
