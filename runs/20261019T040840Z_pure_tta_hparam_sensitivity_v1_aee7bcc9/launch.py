"""Persistent single-GPU sensitivity runner with bounded wall time."""
import json
import os
import subprocess
import sys
import time
import traceback

from runtime import RUN, C, registry, save, sha, setup, verify_freeze


def main():
    setup()
    verify_freeze()
    preflight=json.loads((RUN/'artifacts/preflight.json').read_text())
    assert preflight['passed'] and preflight['config_sha256']==sha(RUN/'config.json')
    registry('running','已通过source-only预检；GPU0顺序执行63个新适应单元，18个历史控制只读复用。')
    start=time.monotonic()
    for seed in C['seeds']:
        remaining=C['budget']['max_wall_seconds']-(time.monotonic()-start)
        if remaining<=0:raise TimeoutError('matrix wall budget exhausted')
        env=dict(os.environ,TEMPORAL_SEED=str(seed),
                 CUDA_VISIBLE_DEVICES=str(C['budget']['physical_gpu']),
                 OMP_NUM_THREADS=str(C['budget']['threads']))
        with (RUN/'logs'/f'seed_{seed}.log').open('a') as log:
            subprocess.run([sys.executable,'-u',str(RUN/'worker.py')],env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
    remaining=C['budget']['max_wall_seconds']-(time.monotonic()-start)
    subprocess.run([sys.executable,str(RUN/'audit.py')],check=True,timeout=max(60,remaining))


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        save('launch_failure.json',{'type':type(exc).__name__,'message':str(exc),
                                    'traceback':traceback.format_exc()})
        registry('failed',f'敏感性批次未完成：{type(exc).__name__}: {exc}；部分产物保留。')
        raise
