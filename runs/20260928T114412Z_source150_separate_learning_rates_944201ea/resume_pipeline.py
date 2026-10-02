"""Recover an interrupted supervisor without repeating completed training.

The frozen train/verify/finalize programs and config are unchanged. This
run-local recovery supervisor records its actions separately from the original
pipeline and applies the same worker, time, and memory limits.
"""
from __future__ import annotations

from collections import deque
import fcntl
import subprocess
import time

from common import *
from pipeline import rss_bytes, stop_process


def main():
    with (RUN / 'artifacts/pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = config_read()
        resource = c['resource']
        assert resource['workers'] * resource['threads_per_worker'] <= 12
        queue = deque()
        reused = []
        for seed in c['seeds']:
            for cond in c['conditions']:
                name = cond['id']
                key = f'{name}_{seed}'
                folder = RUN / 'artifacts' / key
                if (folder / 'verification.json').exists():
                    reused.append(key)
                elif (folder / 'report.json').exists():
                    queue.append(('verify', name, seed))
                else:
                    assert not folder.exists(), f'partial training requires a separate audit: {key}'
                    queue.append(('train', name, seed))
        initial_queue = list(queue)
        atomic_json(RUN / 'artifacts/recovery_launch.json', {
            'started_unix': time.time(), 'reused_verified': reused,
            'initial_queue': initial_queue,
            'reason': 'original supervisor exited after six training reports; no training worker remains',
            'frozen_config_sha256': sha(RUN / 'config.json'),
            'recovery_code_sha256': sha(RUN / 'resume_pipeline.py'),
        })
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                   OMP_NUM_THREADS=str(resource['threads_per_worker']),
                   MKL_NUM_THREADS=str(resource['threads_per_worker']),
                   OPENBLAS_NUM_THREADS='1')
        active = {}
        exit_codes = {}
        reasons = {}
        cancelled = []
        started = time.monotonic()
        while queue or active:
            elapsed = time.monotonic() - started
            if elapsed > resource['pipeline_seconds']:
                cancelled = list(queue)
                queue.clear()
                for key, entry in active.items():
                    reasons[key] = 'global time budget'
                    stop_process(entry['process'])
            while queue and len(active) < resource['workers']:
                kind, name, seed = queue.popleft()
                key = f'{kind}_{name}_{seed}'
                script = 'train.py' if kind == 'train' else 'verify_one.py'
                log = (RUN / 'logs' / f'recovery_{key}.log').open('a')
                process = subprocess.Popen(
                    [sys.executable, '-u', str(RUN / script), '--condition', name,
                     '--seed', str(seed)], cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                active[key] = {'process': process, 'log': log,
                               'started': time.monotonic(), 'kind': kind,
                               'name': name, 'seed': seed}
            usage = {key: rss_bytes(entry['process'].pid)
                     for key, entry in active.items()}
            total = sum(usage.values())
            for key, entry in list(active.items()):
                process = entry['process']
                kind = entry['kind']
                lived = time.monotonic() - entry['started']
                limit = (resource['job_seconds'] + resource['job_grace_seconds']
                         if kind == 'train' else resource['verify_seconds'])
                if process.poll() is None:
                    if lived > limit:
                        reasons[key] = 'job wall limit'
                        stop_process(process)
                    elif usage[key] > resource['rss_gib_per_worker'] * 1024**3:
                        reasons[key] = 'per-process RSS limit'
                        stop_process(process)
                    elif total > resource['total_rss_gib'] * 1024**3:
                        reasons[key] = 'total RSS limit'
                        stop_process(process)
                        total -= usage[key]
                code = process.poll()
                if code is None:
                    continue
                exit_codes[key] = code
                entry['log'].close()
                del active[key]
                if kind == 'train' and code == 0 and elapsed <= resource['pipeline_seconds']:
                    queue.appendleft(('verify', entry['name'], entry['seed']))
            atomic_json(RUN / 'artifacts/recovery_progress.json', {
                'elapsed_seconds': time.monotonic() - started,
                'active': {key: entry['process'].pid for key, entry in active.items()},
                'queued': list(queue), 'exit_codes': exit_codes,
                'stop_reasons': reasons, 'cancelled': cancelled,
            })
            if active:
                time.sleep(2)
        with (RUN / 'logs/recovery_finalize.log').open('w') as log:
            result = subprocess.run([sys.executable, '-u', str(RUN / 'finalize.py')],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(RUN / 'artifacts/recovery_result.json', {
            'exit_codes': exit_codes, 'stop_reasons': reasons,
            'cancelled': cancelled, 'finalize_exit_code': result.returncode,
            'finished_unix': time.time(),
            'elapsed_seconds': time.monotonic() - started,
        })
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
