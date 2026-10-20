"""Detached, reviewable launch; refuse a duplicate live controller."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / 'config.json').read_text())
assert C['status'] == 'frozen'
for relative, expected in json.loads((RUN / 'artifacts/freeze.json').read_text()).items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
with (ROOT / '.paper2_launch.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = RUN / 'artifacts/launch.json'
    if prior.exists():
        pid = json.loads(prior.read_text())['pid']
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:
            command = b''
        assert str(RUN / 'controller.py').encode() not in command, 'controller already running'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(C['budget']['physical_gpu']),
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    command = ['/home/rbf/TA-WF-local-backup/.venv/bin/python3', '-u', str(RUN / 'controller.py')]
    with (RUN / 'logs/controller.out').open('ab', buffering=0) as output:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    prior.write_text(json.dumps({'pid': child.pid, 'command': command, 'physical_gpu': 1,
                                'config_sha256': hashlib.sha256((RUN / 'config.json').read_bytes()).hexdigest()}, indent=2) + '\n')
    print(child.pid)
