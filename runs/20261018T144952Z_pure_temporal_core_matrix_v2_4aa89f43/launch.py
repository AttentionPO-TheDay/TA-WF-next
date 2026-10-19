"""Single persistent GPU worker; no polling or adaptive retuning."""
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
    preflight = json.loads((RUN / 'artifacts/preflight.json').read_text())
    assert preflight['passed'] and preflight['config_sha256'] == sha(RUN / 'config.json')
    registry('running', '纯净源域 TemporalDrift 五条件矩阵已启动；9个冻结引用，顺序执行36个新适应单元。')
    started = time.monotonic()
    for seed in C['seeds']:
        remaining = C['budget']['max_wall_seconds'] - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('matrix wall budget exhausted')
        env = dict(os.environ, TEMPORAL_SEED=str(seed),
                   CUDA_VISIBLE_DEVICES=str(C['budget']['physical_gpu']),
                   OMP_NUM_THREADS=str(C['budget']['threads']))
        with (RUN / 'logs' / f'seed_{seed}.log').open('a') as log:
            subprocess.run([sys.executable, '-u', str(RUN / 'runner.py')], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
    subprocess.run([sys.executable, str(RUN / 'finalize.py')], check=True,
                   timeout=max(60, C['budget']['max_wall_seconds'] - (time.monotonic() - started)))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('launch_failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                    'traceback': traceback.format_exc()})
        registry('failed', f'矩阵未通过完整审计：{type(exc).__name__}: {exc}；部分产物保留。')
        raise
