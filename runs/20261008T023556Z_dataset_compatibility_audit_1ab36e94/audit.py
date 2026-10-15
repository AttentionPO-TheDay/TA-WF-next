#!/usr/bin/env python3
import ast
import hashlib
import json
import os
import struct
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(__file__).resolve().parent
DATA = Path('/mnt/data2/ren/datasets')

CANDIDATES = {
    'TemporalDrift': {'path': 'TemporalDrift', 'modality': 'signed_relative_timestamp; direction is derivable; TAM exists', 'factor': 'time/content', 'role': 'development_observed'},
    'VersionDrift': {'path': 'VersionDrift', 'modality': 'Proteus-family X; verify timestamp semantics', 'factor': 'Tor version', 'role': 'development_candidate_requires_audit'},
    'NetworkDrift': {'path': 'NetworkDrift', 'modality': 'Proteus-family X; verify timestamp semantics', 'factor': 'network/location', 'role': 'development_candidate_requires_audit'},
    'BehaviorDrift': {'path': 'BehaviorDrift', 'modality': 'Proteus-family X; verify timestamp semantics', 'factor': 'browsing behavior', 'role': 'development_candidate_requires_audit'},
    'Defense': {'path': 'Defense', 'modality': 'Proteus-family X; defense condition', 'factor': 'traffic defense', 'role': 'separate_cross_defense_evaluation'},
    'OpenWorld': {'path': 'OpenWorld', 'modality': 'Proteus-family X; open-world setting', 'factor': 'unknown/background traffic', 'role': 'open_world_evaluation'},
    'AWF': {'path': 'awf_concept_drift_200w', 'modality': 'packet direction int16', 'factor': 'concept/date', 'role': 'external_audit_required'},
    'Shusterman': {'path': 'shusterman_concept_drift_v2', 'modality': 'signed packet size in HDF5; inspect internally', 'factor': 'date/browser/network', 'role': 'historical_development_exposure'},
    'WTT-Time': {'path': 'processed/wtt_time/exp_26dd67e087824c2f', 'modality': 'signed packet size and order', 'factor': 'time/network', 'role': 'frozen_protocol_external_evaluation'},
    'PCAP': {'path': 'PCAP', 'modality': 'raw pcap/pcapng', 'factor': 'cross-application traffic', 'role': 'unlabeled_pretraining_only_provenance_pending'},
}

def npy_header(data):
    if len(data) < 10 or data[:6] != b'\x93NUMPY':
        return {'error': 'not_npy'}
    ver = tuple(data[6:8])
    if ver == (1, 0):
        if len(data) < 10: return {'error': 'short_v1_header'}
        n = struct.unpack('<H', data[8:10])[0]; off = 10
    else:
        if len(data) < 12: return {'error': 'short_v2_header'}
        n = struct.unpack('<I', data[8:12])[0]; off = 12
    raw = data[off:off+n]
    try:
        h = ast.literal_eval(raw.decode('latin1'))
        return {'descr': h.get('descr'), 'shape': list(h.get('shape', ())), 'fortran_order': bool(h.get('fortran_order', False)), 'version': list(ver)}
    except Exception as e:
        return {'error': f'header_parse:{type(e).__name__}'}

def npz_schema(path):
    out = {'file': str(path), 'size_bytes': path.stat().st_size, 'arrays': {}}
    try:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith('.npy'):
                    with z.open(name) as f:
                        buf = f.read(16384)
                    out['arrays'][name[:-4]] = npy_header(buf)
            out['zip_members'] = len(z.namelist())
    except Exception as e:
        out['error'] = f'{type(e).__name__}:{e}'
    return out

def quick_hash(path, limit=2*1024*1024):
    # Hash only small files; large data are represented by size/path to keep this audit bounded.
    if path.stat().st_size > limit:
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()

def inventory(root):
    files = []
    ext = Counter()
    total = 0
    for p in root.rglob('*'):
        if p.is_file():
            st = p.stat(); total += st.st_size; ext[p.suffix.lower() or '<none>'] += 1
            files.append({'path': str(p.relative_to(root)), 'size_bytes': st.st_size, 'suffix': p.suffix.lower(), 'small_sha256': quick_hash(p)})
    return {'file_count': len(files), 'total_bytes': total, 'extensions': dict(sorted(ext.items())), 'files': files}

def dedup_report(items):
    by_hash = defaultdict(list)
    for ds, inv in items.items():
        for f in inv['files']:
            if f['small_sha256']:
                by_hash[f['small_sha256']].append((ds, f['path']))
    return {'small_file_hash_groups': [v for v in by_hash.values() if len({x[0] for x in v}) > 1], 'scope': 'only files <=2MiB; no sample-level duplicate claim'}

def main():
    started = time.time(); reports = {}; errors = []
    for name, meta in CANDIDATES.items():
        root = DATA / meta['path']
        rec = {'dataset': name, **meta, 'absolute_path': str(root), 'exists': root.is_dir()}
        if not root.is_dir():
            rec['error'] = 'missing_directory'; errors.append((name, rec['error'])); reports[name] = rec; continue
        try:
            inv = inventory(root); rec['inventory'] = {k:v for k,v in inv.items() if k != 'files'}
            rec['files'] = inv['files']
            # NPZ structural signatures, capped per dataset only for repeated cached artifacts.
            npzs = [Path(root / f['path']) for f in inv['files'] if f['suffix'] == '.npz']
            schemas = [npz_schema(p) for p in npzs]
            sigs = Counter(json.dumps(s.get('arrays', {}), sort_keys=True) for s in schemas if 'arrays' in s)
            rec['npz_file_count'] = len(npzs)
            rec['npz_schema_signature_count'] = len(sigs)
            rec['npz_schema_signatures'] = [{'count': n, 'arrays': json.loads(sig)} for sig, n in sigs.most_common(20)]
            rec['sample_npz_reports'] = schemas[:8]
            rec['hdf5_file_count'] = sum(1 for f in inv['files'] if f['suffix'] in ('.h5', '.hdf5'))
            rec['role_path_counts'] = dict(Counter(Path(f['path']).parts[-1] for f in inv['files'] if Path(f['path']).suffix.lower() in ('.npz', '.h5', '.hdf5')))
            reports[name] = rec
        except Exception as e:
            rec['error'] = f'{type(e).__name__}:{e}'; errors.append((name, rec['error'])); reports[name] = rec
    (RUN/'artifacts'/'dataset_inventory.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
    # Candidate compatibility is an explicit, auditable recommendation, not a performance result.
    compatibility = {
      'strict_common_direction_pool': {'datasets': ['TemporalDrift', 'VersionDrift', 'NetworkDrift', 'BehaviorDrift'], 'status': 'conditional', 'reason': 'NPZ schema appears Proteus-family X/y; must verify X timestamp semantics and cross-directory label mapping before training.'},
      'direction_external_extension': {'datasets': ['AWF'], 'status': 'conditional', 'reason': 'direction-only int16 data; requires direction extraction and class/identity audit; do not merge raw X with timestamp X.'},
      'signed_size_pool': {'datasets': ['WTT-Time', 'Shusterman'], 'status': 'conditional', 'reason': 'both expose signed packet-size/order families, but container schemas, site IDs and protocol roles differ.'},
      'separate_evaluation_only': {'datasets': ['OpenWorld', 'Defense'], 'status': 'recommended', 'reason': 'open-world and defense settings change the task or threat model; preserve separate evaluation unless a new protocol is frozen.'},
      'exclude_from_first_supervised_pool': {'datasets': ['PCAP'], 'status': 'recommended', 'reason': 'raw cross-application PCAP, unlabeled and provenance/license pending; not a confirmed closed-world WF source.'},
      'model_frontend_rule': 'Never feed timestamp magnitudes, packet-size magnitudes and direction integers through one untyped numeric channel. Use a common direction-only view or dataset-specific frontends followed by a shared token projection.'
    }
    (RUN/'artifacts'/'compatibility_report.json').write_text(json.dumps(compatibility, ensure_ascii=False, indent=2) + '\n')
    (RUN/'artifacts'/'duplicate_boundary.json').write_text(json.dumps(dedup_report(reports), ensure_ascii=False, indent=2) + '\n')
    summary = ['# 数据集兼容性审计结果', '', f'- 完成时间（UTC）：{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}', f'- 候选目录：{len(reports)}；缺失/错误：{len(errors)}', '- 设备：CPU；训练、TTA、外部未来评分：均未执行。', '', '## 初步筛选', '', '- 核心 Proteus-family NPZ 候选：TemporalDrift、VersionDrift、NetworkDrift、BehaviorDrift；需先验证 X 的时间戳语义和跨目录标签体系。', '- AWF 只能通过方向子表示接入，不能和 timestamp 数值直接拼接。', '- WTT-Time 与 Shusterman 可作为有符号包大小候选，但需独立前端/分类头和 HDF5 内部 schema 审计。', '- Defense、OpenWorld 保留为独立任务评价；PCAP 暂不进入第一版监督联合训练。', '', '## 限制', '', '- 本轮使用标准库读取容器和 NPZ header；未加载数组到模型。', '- HDF5 内部 dataset/label 结构尚未解析；尚未声称跨数据集网站标签相同。', '- 小文件 hash 只用于边界提示，未完成样本级重复检测。', '', '详细产物见 `dataset_inventory.json`、`compatibility_report.json`、`duplicate_boundary.json`。']
    (RUN/'artifacts'/'summary.md').write_text('\n'.join(summary)+'\n')
    (RUN/'RESULTS.md').write_text('\n'.join(summary)+'\n')
    print(json.dumps({'datasets': len(reports), 'errors': errors, 'seconds': round(time.time()-started, 2)}, ensure_ascii=False))

if __name__ == '__main__': main()
