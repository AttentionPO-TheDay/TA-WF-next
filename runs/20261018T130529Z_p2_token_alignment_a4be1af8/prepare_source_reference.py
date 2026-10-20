"""Only the locked 18,553 source rows enter the Token reference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from model import MultiViewModel
from token_alignment import Moments

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def registry(state, message):
    subprocess.run([sys.executable, str(ROOT / 'scripts/experiment.py'), 'status', '--id', RUN.name,
                    '--state', state, '--summary', message], check=True)


def main():
    config = json.loads((RUN / 'config.json').read_text())
    assert config['status'] == 'frozen' and config['execution_stage'] == 'source_reference_only'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(config['budget']['physical_gpu'])
    for relative, expected in json.loads((RUN / 'artifacts/freeze.json').read_text()).items():
        assert sha(ROOT / relative) == expected, relative
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    base = ROOT / config['base_run']
    reports = json.loads((base / 'artifacts/reports.json').read_text())
    assert {r['seed'] for r in reports} == set(config['seeds'])
    assert reports == config['base_checkpoints'], 'source checkpoint reports changed'
    preflight = json.loads((base / 'artifacts/data_preflight.json').read_text())
    assert preflight['passed'] and preflight['source_rows'] == 18553
    assert sha(base / 'artifacts/prepared.pt') == preflight['sha256']['prepared']
    source = torch.load(base / 'artifacts/prepared.pt', map_location='cpu', weights_only=True)['source']
    assert len(source['rows']) == 18553
    registry('running', '仅源域Token参考构建启动；目标评分关闭。')
    start = time.monotonic()
    evidence = []
    for seed in config['seeds']:
        report = next(r for r in reports if r['seed'] == seed)
        checkpoint = base / 'checkpoints' / f'pure_source_wide_bn_{seed}.pth'
        assert sha(checkpoint) == report['checkpoint_sha256']
        model = MultiViewModel('wide_bn', 102).cuda().eval().requires_grad_(False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
        model.load_state_dict(state, strict=True)
        moments = Moments()
        prototype_sum = torch.zeros(102, 110, dtype=torch.float64)
        prototype_counts = torch.zeros(102, dtype=torch.long)
        embeddings = []
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            for first in range(0, 18553, config['batch_size']):
                if time.monotonic() - start > config['budget']['max_wall_seconds']:
                    raise TimeoutError('source reference budget exceeded')
                part = slice(first, first + config['batch_size'])
                tokens = model.generator(source['timestamps'][part].cuda(), source['tam'][part].cuda())['time']
                moments.add(tokens)
                pooled = tokens.mean(1).cpu()
                embeddings.append(pooled)
                labels = source['labels'][part]
                prototype_sum.index_add_(0, labels, pooled.double())
                prototype_counts.index_add_(0, labels, torch.ones_like(labels))
        assert all(torch.equal(value.detach().cpu(), before[key]) for key, value in model.state_dict().items())
        assert (prototype_counts > 0).all()
        output = {'moments': moments.finish(), 'class_prototypes': (prototype_sum / prototype_counts[:, None]).float(),
                  'class_counts': prototype_counts, 'source_embeddings': torch.cat(embeddings),
                  'source_rows': source['rows'], 'source_labels': source['labels'],
                  'checkpoint_sha256': report['checkpoint_sha256'], 'seed': seed,
                  'token_position': 'WideGenerator.time before time_projection', 'target_data_accessed': False}
        path = RUN / 'artifacts' / f'source_token_reference_{seed}.pt'
        torch.save(output, path)
        evidence.append({'seed': seed, 'rows': 18553, 'tokens': output['moments']['count'],
                         'checkpoint_sha256': report['checkpoint_sha256'], 'reference_sha256': sha(path),
                         'base_state_unchanged': True})
        print(json.dumps(evidence[-1]), flush=True)
        del model
        torch.cuda.empty_cache()
    save(RUN / 'artifacts/source_reference_audit.json', {'passed': True, 'seeds': evidence,
         'target_data_accessed': False, 'target_scored': False, 'elapsed_seconds': time.monotonic() - start})
    (RUN / 'RESULTS.md').write_text('# Token 对齐：源域准备结果\n\n三 seed 的纯净源 Token 统计量、102 类原型和逐行 pooled embedding 已完成；每 seed 仅使用锁定的 18,553 行，底座参数与 BN 缓冲区均未改变。\n\n目标域 A/B/C 消融仍待切片身份、标签映射及 reference/query 清单审计；本结果不包含 ΔF1 或对齐有效性的结论。目标评分未启动。\n')
    registry('planned', '三seed源Token参考已完成；目标A/B/C仍待合法切片清单，不含性能结论。')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(RUN / 'artifacts/failure.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        registry('failed', f'Token源参考准备失败：{type(exc).__name__}；产物保留。')
        raise
