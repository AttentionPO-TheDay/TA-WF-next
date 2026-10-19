"""Preflight, three bounded seed workers, independent audit, then figures."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from runtime import RUN, ROOT, C, save, registry, setup, verify_freeze, metrics, sha


def main():
    setup()
    verify_freeze()
    with (ROOT / '.pure_temporal_core.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry('running', '阶段1与9项frozen复用审计通过结果；正在预检四种适应及coadapt诊断，预检后GPU0/1/2三seed并行。')
        subprocess.run([sys.executable, '-u', str(RUN / 'preflight.py')], check=True, timeout=600)
        children, handles = [], []
        try:
            for seed in C['seeds']:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(C['budget']['seed_gpu_map'][str(seed)]), TEMPORAL_SEED=str(seed))
                handle = (RUN / 'logs' / f'worker_{seed}.out').open('ab', buffering=0)
                handles.append(handle)
                children.append(subprocess.Popen([sys.executable, '-u', str(RUN / 'runner.py')], env=env,
                                                  stdout=handle, stderr=subprocess.STDOUT))
            begun = time.monotonic()
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('worker failed; inspect per-seed failure files')
                if time.monotonic() - begun > C['budget']['max_wall_seconds']:
                    raise TimeoutError('parallel worker budget exhausted')
                size = 0
                for p in RUN.rglob('*'):
                    try:
                        if p.is_file():
                            size += p.stat().st_size
                    except FileNotFoundError:
                        pass
                if size > C['budget']['max_output_gib'] * 1024 ** 3:
                    raise TimeoutError('run output budget exhausted')
                time.sleep(2)
            assert all(child.returncode == 0 for child in children)
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=30)
            for handle in handles:
                handle.close()
        rows = []
        for seed in C['seeds']:
            rows.extend(json.loads((RUN / 'artifacts' / f'scores_seed_{seed}.json').read_text()))
        baseline_root = ROOT / 'runs' / C['frozen_baseline_run']
        baseline_scores = json.loads((baseline_root / 'artifacts/scores.json').read_text())
        for b in baseline_scores:
            path = baseline_root / 'artifacts' / f"pred_{b['seed']}_{b['role']}_frozen.npz"
            rows.append({**b, 'negative_adaptation_rate': 0.0, 'recovery_rate': 0.0, 'delta_f1_pp': 0.0,
                         'origin': 'reused_audited_pure_frozen', 'source_run': C['frozen_baseline_run'],
                         'prediction_file': str(path.relative_to(ROOT)), 'prediction_sha256': sha(path),
                         'query_batch_statistics': False})
        assert len(rows) == 54
        save('scores.json', rows)
        subprocess.run([sys.executable, '-u', str(RUN / 'audit.py')], check=True, timeout=1800)
        figure_run = ROOT / C['figure_run']
        subprocess.run([sys.executable, '-u', str(figure_run / 'render.py')], check=True, timeout=300)
        registry('completed', '45项五条件主表（9 frozen复用+36新适应）及9 coadapt诊断完成，45 checkpoint重载/54指标复算通过；三张机理图已导出。')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('controller_failure.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        registry('failed', f'纯净时序发布流水线失败：{type(exc).__name__}: {exc}；产物保留。')
        raise
