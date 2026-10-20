"""Single GPU process at a time; source preparation -> preflight -> matrix -> audit."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SOURCE_REFERENCE = ROOT / 'runs/20261018T130529Z_p2_token_alignment_a4be1af8'


def main():
    with (ROOT / '.paper2_gpu1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit_path = SOURCE_REFERENCE / 'artifacts/source_reference_audit.json'
        if not audit_path.exists():
            subprocess.run([sys.executable, '-u', str(SOURCE_REFERENCE / 'prepare_source_reference.py')], check=True)
        assert json.loads(audit_path.read_text())['passed']
        subprocess.run([sys.executable, '-u', str(RUN / 'preflight.py')], check=True)
        config = json.loads((RUN / 'config.json').read_text())
        # Preflight owns manifest writes. Workers only compare immutable
        # manifests, and write scores/progress/checkpoints under their seed.
        import torch
        for name in ('wtt_time', 'awf'):
            bundle = torch.load(RUN / 'artifacts' / f'{name}_prepared.pt', map_location='cpu', weights_only=False)
            manifest = {'mapping': bundle['mapping'], 'audit': bundle['audit'],
                        'source': {k: v for k, v in bundle['source'].items() if k != 'x'},
                        'targets': {date: {role: {k: v for k, v in value[role].items() if k != 'x'}
                                          for role in ('reference', 'query')} for date, value in bundle['targets'].items()}}
            manifest = json.loads(json.dumps(manifest, default=lambda x: x.tolist()))
            path = RUN / 'artifacts' / f'{name}_row_manifest.json'
            if path.exists():
                assert json.loads(path.read_text()) == manifest
            else:
                path.write_text(json.dumps(manifest, indent=2) + '\n')
            del bundle
        children, logs = [], []
        try:
            for seed in config['seeds']:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config['budget']['seed_gpu_map'][str(seed)]), P2_SEED=str(seed))
                out = (RUN / 'logs' / f'worker_{seed}.out').open('ab', buffering=0)
                logs.append(out)
                children.append(subprocess.Popen([sys.executable, '-u', str(RUN / 'runner.py')], env=env, stdout=out, stderr=subprocess.STDOUT))
            import time
            started = time.monotonic()
            while any(child.poll() is None for child in children):
                assert all(child.poll() in (None, 0) for child in children), 'worker failed; other workers stopped'
                if time.monotonic() - started > config['budget']['max_wall_seconds']:
                    raise TimeoutError('parallel worker wall budget exceeded')
                time.sleep(2)
            assert all(child.returncode == 0 for child in children)
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=30)
            for out in logs:
                out.close()
        scores = []
        for seed in config['seeds']:
            scores.extend(json.loads((RUN / 'artifacts' / f'scores_seed_{seed}.json').read_text()))
        assert len(scores) == config['budget']['total_cells']
        (RUN / 'artifacts/scores.json').write_text(json.dumps(scores, indent=2) + '\n')
        subprocess.run([sys.executable, '-u', str(RUN / 'audit.py')], check=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        (RUN / 'artifacts/controller_failure.json').write_text(json.dumps({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        subprocess.run([sys.executable, str(ROOT / 'scripts/experiment.py'), 'status', '--id', RUN.name,
                        '--state', 'failed', '--summary', f'发布流水线失败：{type(exc).__name__}；源准备/预检/执行阶段见日志。'], check=True)
        raise
