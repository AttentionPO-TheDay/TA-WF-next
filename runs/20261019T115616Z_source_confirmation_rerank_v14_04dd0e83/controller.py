import os
import json
import time
import subprocess
import traceback
import fcntl
from pathlib import Path
import torch
from runtime import RUN, ROOT, C, save, sha, verify, status, selected, csvout

START = time.monotonic()


def budget():
    assert time.monotonic()-START <= C['budget']['wall_seconds']
    assert sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()) < C['budget']['output_gib']*1024**3


def freeze():
    manifest = {}
    def add(p, expected=None):
        p = Path(p)
        h = sha(p)
        if expected is not None:
            assert h == expected, str(p)
        manifest[str(p.relative_to(ROOT))] = h
    for key in ['v9', 'v10', 'v12', 'v13']:
        p = ROOT/C[key]
        delivery = json.loads((p/'artifacts/delivery_audit.json').read_text())
        assert delivery['passed']
        add(p/'artifacts/delivery_audit.json')
        if key == 'v13':
            for rel, h in delivery['outputs'].items():
                assert sha(p/rel) == h, rel
        if key in ['v9', 'v10']:
            for seed in C['seeds']:
                names = [f'checkpoints/{seed}_direction_iat.pt', f'artifacts/sourcevalid_{seed}_direction_iat.npz'] if key == 'v10' else [f'checkpoints/{seed}_{sl}_smooth_balance.pt' for sl in C['slices']]+[f'artifacts/valid_{seed}_{sl}_{arm}.npz' for sl in C['slices'] for arm in ['frozen', 'smooth_balance']]
                for name in names:
                    add(p/name, delivery['outputs'][name])
        if key == 'v12':
            for name in ['artifacts/source_calibration.json', 'artifacts/cells.json']:
                add(p/name, delivery['outputs'][name])
        if key == 'v13':
            for seed in C['seeds']:
                for sl in C['slices']:
                    name = f'artifacts/rows_query_{seed}_{sl}.npz'
                    add(p/name, delivery['outputs'][name])
    source = ROOT/C['source_run']
    frozen = json.loads((source/'artifacts/freeze.json').read_text())
    add(source/'artifacts/prepared.pt', frozen[str((source/'artifacts/prepared.pt').relative_to(ROOT))])
    reports = json.loads((source/'artifacts/reports.json').read_text())
    add(source/'artifacts/reports.json')
    for seed in C['seeds']:
        expected = next(r['checkpoint_sha256'] for r in reports if r['seed'] == seed)
        add(source/'checkpoints'/f'pure_source_wide_bn_{seed}.pth', expected)
    for key, names in [('v9', ['model.py', 'framework_base.py', 'methods.py', 'base_model.py']), ('v10', ['model.py']), ('v13', ['audit.py'])]:
        for name in names:
            add(ROOT/C[key]/name)
    for p in list(RUN.glob('*.py'))+[RUN/'config.json', RUN/'PLAN.md', RUN/'SOURCES.md', ROOT/'PROTOCOL.md', ROOT/'HANDOFF.md', ROOT/'configs/datasets.json']:
        add(p)
    ds = json.loads((ROOT/'configs/datasets.json').read_text())
    assert (Path(ds['data_root'])/ds['datasets']['proteus_temporal']['path']).is_dir()
    d = torch.load(source/'artifacts/prepared.pt', map_location='cpu', weights_only=True)
    assert len(d['source']['rows']) == 18553 and len(d['valid']['rows']) == 2160
    rows, selected_rows = [], {}
    for role in ['train', 'valid']:
        dd = d['source' if role == 'train' else 'valid']
        ix = selected(d, role)
        selected_rows[role] = dd['rows'][ix].tolist()
        for i in ix:
            rows.append(dict(role=role, row=int(dd['rows'][i]), label=int(dd['labels'][i])))
    save('source_rows.json', selected_rows)
    csvout('source_rows.csv', rows)
    add(RUN/'artifacts/source_rows.json')
    add(RUN/'artifacts/source_rows.csv')
    save('freeze.json', manifest)
    verify()
    save('reuse_audit.json', dict(passed=True, inputs=len(manifest), source_train_selected=len(selected_rows['train']), source_valid_selected=len(selected_rows['valid']), all_source_models_frozen=True, old_run_writes=False))


def parallel(phase):
    procs, handles = [], []
    try:
        for gpu, seed in zip(C['budget']['gpus'], C['seeds']):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), SOURCE_SEED=str(seed), CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
            h = (RUN/'logs'/f'{phase}_{seed}.log').open('w')
            handles.append(h)
            procs.append(subprocess.Popen([os.sys.executable, str(RUN/'worker.py'), phase], env=env, stdout=h, stderr=subprocess.STDOUT))
        save('active_processes.json', dict(phase=phase, pids=[p.pid for p in procs]))
        while any(p.poll() is None for p in procs):
            budget()
            if any(p.poll() not in [None, 0] for p in procs):
                raise RuntimeError(f'worker failure {phase}: {[p.poll() for p in procs]}')
            time.sleep(1)
        assert all(p.returncode == 0 for p in procs), (phase, [p.returncode for p in procs])
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=20)
        for h in handles:
            h.close()


def call(script, phase=None):
    budget()
    name = script if phase is None else phase
    with (RUN/'logs'/f'{name}.log').open('w') as f:
        cmd = [os.sys.executable, str(RUN/(script+'.py'))]+([phase] if phase else [])
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=max(1, C['budget']['wall_seconds']-(time.monotonic()-START)))


def deliver():
    verify()
    budget()
    elapsed = time.monotonic()-START
    save('active_processes.json', dict(phase='completed', pids=[]))
    save('resource_audit.json', dict(pipeline_seconds=elapsed, conservative_gpu_hours=3*elapsed/3600, new_backbone_training_steps=0, new_small_linear_models=12 if (RUN/'artifacts/source_fit_complete.json').exists() else 0, output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())))
    assert 3*elapsed/3600 <= C['budget']['gpu_hours']
    save('delivery_audit.json', dict(passed=True, outputs={str(p.relative_to(RUN)): sha(p) for p in RUN.rglob('*') if p.is_file() and not any(x in p.parts for x in ['logs', 'lgs', '__pycache__']) and p.name != 'delivery_audit.json'}))


def main():
    with (ROOT/'.source_confirmation_v14.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            freeze()
            status('running', '配置/源子集冻结；GPU0/1/2先核验全部源valid旧输出，不打开目标评分。')
            parallel('preflight')
            assert all(json.loads((RUN/'artifacts'/f'preflight_{s}.json').read_text())['passed'] for s in C['seeds'])
            status('running', '旧网络行为核验通过；生成固定源扰动和冻结候选，主网络不训练。')
            parallel('compute')
            parallel('audit')
            status('running', '30源缓存探针重载通过；只训练源R/D/W线性重排器，目标评分仍关闭。')
            try:
                call('rerank', 'fit')
            except subprocess.CalledProcessError:
                path = RUN/'artifacts/source_feasibility.json'
                if path.exists() and not json.loads(path.read_text())['passed']:
                    (RUN/'RESULTS.md').write_text('# 源域候选确认重排 v14\n\n固定源训练改写类型数量未达到每类25，按预注册停止；未打开目标评分，未增加扰动或训练预算。详见artifacts/source_type_counts.csv。\n')
                    save('registered_decisions.json', dict(primary_passed=False, reason='source_type_insufficient', query_opened=False, integrated=False))
                    deliver()
                    status('completed', '源训练类型不足，按冻结规则停止；未打开目标评分，无反馈追加。')
                    return
                raise
            status('running', '12个源线性重排器和源阈值已封存；开始两片终端评分，目标无新前向。')
            call('rerank', 'evaluate')
            call('rerank', 'audit')
            call('report')
            audits = [json.loads((RUN/'artifacts'/f'reload_audit_{s}.json').read_text()) for s in C['seeds']]
            assert all(a['passed'] for a in audits) and sum(a['cache_states'] for a in audits) == 30
            assert json.loads((RUN/'artifacts/rerank_reload_audit.json').read_text())['passed']
            save('postrun_audit.json', dict(passed=True, source_cache_probe_states=30, probe_rows_per_state=256, full_sourcevalid_old_model_states=12, small_linear_models_reloaded=12, target_metric_states=24, target_labels_not_fit=True))
            deliver()
            decision = json.loads((RUN/'artifacts/registered_decisions.json').read_text())
            status('completed', f'30源缓存探针/12线性模型重载/24目标评分核验通过；主判据={decision["primary_passed"]}。候选不变，未部署。')
            print(json.dumps(decision, ensure_ascii=False), flush=True)
        except Exception as e:
            save('controller_failure.json', dict(type=type(e).__name__, error=str(e), traceback=traceback.format_exc()))
            save('active_processes.json', dict(phase='failed', pids=[]))
            status('failed', f'执行失败：{type(e).__name__}；日志和旧产物保留。')
            raise


if __name__ == '__main__':
    main()
