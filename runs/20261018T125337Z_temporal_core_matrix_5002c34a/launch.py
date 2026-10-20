"""Detached launch with immutable configuration and duplicate protection."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
from runtime import RUN, ROOT, C, verify_freeze

verify_freeze()
assert C['status'] == 'frozen'
with (ROOT / '.pure_temporal_launch.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch = RUN / 'artifacts/launch.json'
    if launch.exists():
        pid = json.loads(launch.read_text())['pid']
        try:
            prior = Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:
            prior = b''
        assert str(RUN / 'controller.py').encode() not in prior, 'controller already alive'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MPLCONFIGDIR=str(RUN / 'artifacts/matplotlib_cache'))
    command = ['/home/rbf/TA-WF-local-backup/.venv/bin/python3', '-u', str(RUN / 'controller.py')]
    with (RUN / 'logs/controller.out').open('ab', buffering=0) as output:
        child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    launch.write_text(json.dumps({'pid': child.pid, 'command': command, 'seed_gpu_map': C['budget']['seed_gpu_map']}, indent=2) + '\n')
    print(child.pid)
