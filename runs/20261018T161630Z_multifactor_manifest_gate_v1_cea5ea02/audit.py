"""Build factor row manifests from audited identity artifacts; no X or model reads."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / 'config.json').read_text())
INPUT_HASHES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def input_file(path):
    INPUT_HASHES[str(path.relative_to(ROOT))] = sha(path)
    return path


def member(path, name):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name + '.npy')
        with archive.open(info) as stream:
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        with archive.open(info) as stream:
            value = np.load(stream, allow_pickle=False)
    INPUT_HASHES[str(path) + '::' + name + '.npy'] = h.hexdigest()
    return value


def registry(state, summary):
    subprocess.run([sys.executable, str(ROOT / 'scripts/experiment.py'), 'status',
                    '--id', RUN.name, '--state', state, '--summary', summary], check=True)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    start = time.monotonic()
    assert C['status'] == 'frozen' and C['id'] == RUN.name
    assert not C['performance_scoring'] and not C['adaptation'] and not C['new_training']
    freeze = json.loads((RUN / 'artifacts/source_freeze.json').read_text())
    assert all(sha(RUN / key) == value for key, value in freeze.items())
    datasets = json.loads(input_file(ROOT / C['dataset_config']).read_text())
    data_root = Path(datasets['data_root'])
    assert data_root.is_dir()
    p0 = ROOT / C['p0_run'] / 'artifacts'
    index = json.loads(input_file(p0 / 'core_file_index.json').read_text())
    index_by_path = {entry['path']: (i, entry) for i, entry in enumerate(index)}
    assert len(index) == 30 and len(index_by_path) == 30
    group_path = input_file(p0 / 'core_row_group_ids.npz')
    with np.load(group_path) as archive:
        groups = {entry['path']: archive[f'raw10000_{i}'].copy() for i, entry in enumerate(index)}
    assert all(len(groups[e['path']]) == e['rows'] for e in index)
    source_path = input_file(ROOT / C['source_manifest'])
    source_rows, source_groups, source_labels = set(), set(), set()
    with source_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            assert row['file'] == 'TemporalDrift/train.npz'
            i = int(row['row'])
            assert i not in source_rows and int(row['raw_group_id']) == groups[row['file']][i]
            source_rows.add(i)
            source_groups.add(int(row['raw_group_id']))
            source_labels.add(int(row['label']))
    assert len(source_rows) == len(source_groups) == 18553 and len(source_labels) == 102
    source_audit = json.loads(input_file(ROOT / C['source_run'] / 'artifacts/weight_audit.json').read_text())
    assert source_audit['passed']
    registry('running', '只读构建多因素逐行身份、源域隔离及可评价性门槛清单；不评分。')

    anchor_groups = set()
    for version in C['version_views']:
        for role in ('train', 'valid'):
            anchor_groups.update(map(int, groups[f'VersionDrift/{version}/{role}.npz']))
    files = []
    for version in C['version_views']:
        files.append(('VersionDrift', f'VersionDrift/{version}/drift.npz', version))
    for geo in C['network_views']:
        files.append(('NetworkDrift', f'NetworkDrift/{geo}.npz', geo))
    files.append(('BehaviorDrift', C['behavior_file'], 'subpage'))
    target_group_counts = Counter(int(g) for _, path, _ in files for g in groups[path])
    assert len(files) == 10
    out_path = RUN / 'artifacts/factor_rows.csv'
    fields = ['family','file','slice','row','label','raw_group_id','in_pure_source','cross_target_reuse_count',
              'attributed_version','version_status','version_anchor_reuse','exact_url_sha256','exact_url_copies']
    summary = {}
    with out_path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for family, relative, slice_name in files:
            group = groups[relative]
            physical = data_root / relative
            stat = physical.stat()
            entry = index_by_path[relative][1]
            assert stat.st_size == entry['source_stat']['size'] and stat.st_mtime_ns == entry['source_stat']['mtime_ns']
            if family == 'VersionDrift':
                manifest = ROOT / C['version_identity_run'] / 'artifacts' / f'version_{slice_name}_identity.csv'
                identity_rows = list(csv.DictReader(input_file(manifest).open(newline='')))
                assert len(identity_rows) == len(group)
                labels = np.array([int(x['label']) for x in identity_rows], dtype=np.int64)
                urls = None
            else:
                labels = member(physical, 'y').astype(np.int64)
                assert len(labels) == len(group)
                urls = member(physical, 'url') if family == 'BehaviorDrift' else None
            assert labels.min() >= 0 and labels.max() < 102
            url_hashes = None
            if urls is not None:
                assert len(urls) == len(group)
                url_hashes = [hashlib.sha256(str(url).encode('utf-8')).hexdigest() for url in urls]
                counts = Counter(url_hashes)
                by_url_labels = defaultdict(set)
                for u, y in zip(url_hashes, labels):
                    by_url_labels[u].add(int(y))
                assert len(counts) == 17739 and sum(v - 1 for v in counts.values()) == 2661
                assert all(len(x) == 1 for x in by_url_labels.values())
            status_counts = Counter()
            attributed = Counter()
            overlap = 0
            anchor_reuse = 0
            target_reuse = 0
            for i, (g, y) in enumerate(zip(group, labels)):
                g = int(g)
                if family == 'VersionDrift':
                    identity = identity_rows[i]
                    assert int(identity['row']) == i and int(identity['label']) == int(y)
                    assert identity['status'] in ('unique','unmatched','ambiguous')
                    version = identity['anchor_versions'] if identity['status'] == 'unique' else ''
                    assert int(identity['label_conflict']) == 0
                    status = identity['status']
                    if version:
                        attributed[version] += 1
                else:
                    version, status = '', 'not_applicable'
                source_hit = g in source_groups
                anchor_hit = family == 'VersionDrift' and g in anchor_groups
                reuse = target_group_counts[g]
                u = url_hashes[i] if url_hashes is not None else ''
                writer.writerow({'family':family,'file':relative,'slice':slice_name,'row':i,'label':int(y),
                    'raw_group_id':g,'in_pure_source':int(source_hit),'cross_target_reuse_count':reuse,
                    'attributed_version':version,'version_status':status,'version_anchor_reuse':int(anchor_hit),
                    'exact_url_sha256':u,'exact_url_copies':counts[u] if u else ''})
                overlap += source_hit
                anchor_reuse += anchor_hit
                target_reuse += reuse > 1
                status_counts[status] += 1
            summary[relative] = {'family':family,'slice':slice_name,'rows':len(group),'classes':len(set(map(int,labels))),
                'source_exact_overlap':overlap,'rows_reused_across_target_views':target_reuse,
                'version_status':dict(status_counts),'attributed_version':dict(attributed),
                'version_anchor_reuse_rows':anchor_reuse,
                'unique_exact_urls':len(counts) if urls is not None else None,
                'exact_url_duplicate_copies':sum(v-1 for v in counts.values()) if urls is not None else None}
            assert overlap == 0, relative
            if family == 'VersionDrift':
                assert status_counts['unmatched'] == 10017 and status_counts['ambiguous'] == 0
                assert anchor_reuse == status_counts['unique']
            if time.monotonic() - start > C['budget']['max_wall_seconds']:
                raise TimeoutError('audit budget exceeded')
    gate = {'pure_source_exact_isolation': True,
            'version_independent_attributed_query': False,
            'network_geographic_file_identity': True,
            'network_quantitative_loss_latency_provenance': False,
            'behavior_exact_url_consistency': True,
            'behavior_homepage_site_mapping': False,
            'official_cross_family_site_mapping': False,
            'reference_query_protocol_frozen': False,
            'multifactor_performance_unlocked': False}
    report = {'passed': True, 'kind': 'structural_candidate_manifest_not_performance',
              'source_rows':len(source_rows), 'source_classes':len(source_labels),
              'target_files':summary,'gate':gate,'input_sha256':INPUT_HASHES,
              'manifest_sha256':sha(out_path),'elapsed_seconds':time.monotonic()-start}
    write_json(RUN / 'artifacts/audit.json', report)
    lines = ['# 多因素身份门槛审计结果', '',
        'CPU 只读完成。逐行清单在 `artifacts/factor_rows.csv`；无训练、适应、预测或性能评分。纯净源域 18,553 行与本轮10个目标文件均无完整原始 trace 精确重叠；这不证明 session 独立或网站语义一致。', '',
        '| 切片 | 行数 | 102类覆盖 | 源精确重叠 | 可归因/备注 |', '|---|---:|---:|---:|---|']
    for path, row in summary.items():
        note = (f"唯一版本 {row['version_status'].get('unique',0)}，未知 {row['version_status'].get('unmatched',0)}；唯一归因行均复用train/valid锚点" if row['family']=='VersionDrift' else
                f"URL {row['unique_exact_urls']} 个，副本 {row['exact_url_duplicate_copies']}" if row['family']=='BehaviorDrift' else '仅地域文件名，丢包/延迟数值未证实')
        lines.append(f"| {path} | {row['rows']} | {row['classes']} | {row['source_exact_overlap']} | {note} |")
    lines += ['', '裁决：多因素目标评分**仍未解锁**。Version 可归因行是版本训练/验证锚点重用，未知行不能命名版本；Network 缺定量物理条件与站点身份；Behavior URL 一致性不能替代 homepage↔subpage 网站映射。后续可依据此清单设计已观察数据上的诊断子集，但须另行冻结标签映射和 reference/query 权限，不能称独立盲测。',
              '', '输入与清单 SHA256、逐文件计数及门槛布尔值见 `artifacts/audit.json`。']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    assert out_path.stat().st_size <= C['budget']['max_output_gib'] * 1024**3
    registry('completed', '10个多因素目标文件逐行清单及源域精确隔离审计完成；身份/切分门槛仍未通过，目标评分关闭。')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write_json(RUN / 'artifacts/failure.json', {'type':type(exc).__name__,'message':str(exc),
            'traceback':traceback.format_exc()})
        registry('failed', f'逐行身份门槛审计失败：{type(exc).__name__}: {exc}；不解锁目标评分。')
        raise
