"""Select compact experiment evidence; never run experiments or stage files."""
import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/evidence_manifest.json'
POLICY = 'compact-evidence-v1'
JSON_NAMES = {
    'aggregate.json', 'audit.json', 'config.json', 'contrasts.json',
    'failure.json', 'freeze.json', 'historical_audit.json', 'input_audit.json',
    'input_seal.json', 'integrity.json', 'integrity_check.json', 'manifest.json',
    'metrics.json', 'output_seal.json', 'pipeline_result.json',
    'postrun_audit.json', 'preflight.json', 'preflight_cpu.json',
    'preflight_gpu.json', 'pretraining_seal.json', 'provenance.json',
    'report.json', 'resume_audit.json', 'scores.json', 'seal.json',
    'split_manifest.json', 'summary.json', 'verification.json', 'verdict.json',
}
CSV_NAMES = {'metrics.csv', 'metrics_long.csv', 'aggregate_metrics.csv'}
SNAPSHOT_PARTS = ('invalid', 'superseded', 'repair_history', 'pre_revision',
                  'pre_write_repair')
VENDOR_PARTS = {'upstream', 'wflib_copy', 'ares', 'proteus_rf', 'rf',
                'varcnn', 'keras'}
RAW_FIELDS = {'x', 'y', 'y_true', 'y_pred', 'labels', 'true_labels',
              'pred', 'preds', 'predictions', 'probabilities', 'probs',
              'logits', 'features', 'packets', 'traces', 'urls'}
SECRET = re.compile(
    r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'
    r'|\bgh[pousr]_[A-Za-z0-9]{30,}\b'
    r'|\bgithub_pat_[A-Za-z0-9_]{50,}\b'
    r'|\bAKIA[0-9A-Z]{16}\b|\bsk-[A-Za-z0-9_-]{32,}\b'
    r'|https?://[^\s/:@]+:[^\s/@]+@'
    r'|[\'"](?:api_key|access_token|refresh_token|password|client_secret)'
    r'[\'"]\s*[:=]\s*[\'"][^\'"\s]{8,}[\'"]', re.IGNORECASE)


def category(path):
    parts = path.parts
    if not (len(parts) >= 4 and parts[0] == 'runs' and parts[2] == 'artifacts'):
        if not (len(parts) == 3 and parts[0] == 'runs' and path.name in CSV_NAMES):
            return None
    if any(part in VENDOR_PARTS for part in parts[3:]):
        return None
    snapshot = any(marker in part for part in parts[3:]
                   for marker in SNAPSHOT_PARTS)
    if path.suffix == '.json' and path.name in JSON_NAMES:
        return 'historical_record' if snapshot else 'result_or_audit'
    if path.suffix == '.csv' and path.name in CSV_NAMES:
        return 'aggregate_table'
    if path.suffix in {'.py', '.md'} and snapshot:
        return 'historical_source_or_note'
    if path.suffix == '.md' and path.name in {
            'summary.md', 'attempt_v1_summary.md', 'PLAN_before_eval_v4.md'}:
        return 'historical_source_or_note'
    return None


def has_sample_payload(node):
    if isinstance(node, dict):
        return any((key.lower() in RAW_FIELDS and isinstance(value, (list, dict))
                    and bool(value)) or has_sample_payload(value)
                   for key, value in node.items())
    if isinstance(node, list):
        return len(node) > 2000 or any(has_sample_payload(item) for item in node)
    return False


def inspect(path, data):
    limit = 4 * 1024 * 1024 if path.name == 'scores.json' else 512 * 1024
    if len(data) > limit:
        return 'size_limit'
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return 'non_utf8'
    if SECRET.search(text):
        return 'credential_pattern'
    if path.suffix == '.json':
        try:
            obj = json.loads(text)
        except ValueError:
            return 'invalid_json'
        if not isinstance(obj, (dict, list)):
            return 'non_structured_json'
        if has_sample_payload(obj):
            return 'sample_payload_or_large_array'
    if path.suffix == '.csv':
        rows = list(csv.reader(io.StringIO(text)))
        if not rows or len(rows) > 2001:
            return 'csv_row_limit'
        if set(field.lower() for field in rows[0]) & RAW_FIELDS:
            return 'sample_columns'
        if not any('accuracy' in field.lower() or 'f1' in field.lower()
                   or 'metric' in field.lower() for field in rows[0]):
            return 'not_metric_table'
    return None


def candidates():
    for run in sorted((ROOT / 'runs').iterdir()):
        if run.is_symlink() or not run.is_dir() or not (run / 'PLAN.md').is_file():
            continue
        artifacts = run / 'artifacts'
        if artifacts.is_dir() and not artifacts.is_symlink():
            for base, dirs, names in os.walk(artifacts, followlinks=False):
                dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
                for name in sorted(names):
                    path = Path(base) / name
                    if not path.is_symlink():
                        yield path.relative_to(ROOT)
        for name in sorted(CSV_NAMES):
            path = run / name
            if path.is_file() and not path.is_symlink():
                yield path.relative_to(ROOT)


def select():
    selected, excluded = [], []
    for path in candidates():
        kind = category(path)
        if kind is None:
            continue
        data = (ROOT / path).read_bytes()
        reason = inspect(path, data)
        if reason:
            excluded.append({'path': path.as_posix(), 'reason': reason})
        else:
            selected.append({'path': path.as_posix(), 'bytes': len(data),
                             'sha256': hashlib.sha256(data).hexdigest(),
                             'category': kind})
    return sorted(selected, key=lambda row: row['path']), excluded


def verify(manifest):
    if manifest['policy'] != POLICY:
        raise ValueError('Unknown selection policy')
    seen = set()
    for row in manifest['files']:
        path = Path(row['path'])
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'runs':
            raise ValueError('Manifest path outside runs')
        if row['path'] in seen or category(path) != row['category']:
            raise ValueError('Duplicate path or category mismatch')
        seen.add(row['path'])
        target = ROOT / path
        if not target.resolve().is_relative_to(ROOT) or target.is_symlink():
            raise ValueError('Evidence path resolves outside workspace or is a link')
        data = target.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Evidence changed: ' + row['path'])
        reason = inspect(path, data)
        if reason:
            raise ValueError(reason + ': ' + row['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write-manifest', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        verify(manifest)
        print(f"Verified {len(manifest['files'])} evidence files; hashes and filters match.")
        return
    files, excluded = select()
    print(json.dumps({'selected_files': len(files),
                      'selected_bytes': sum(row['bytes'] for row in files),
                      'categories': dict(collections.Counter(row['category'] for row in files)),
                      'excluded_candidates': dict(collections.Counter(row['reason'] for row in excluded))},
                     ensure_ascii=False, indent=2))
    if args.write_manifest:
        manifest = {'schema_version': 1, 'policy': POLICY,
                    'generated_utc': datetime.now(timezone.utc).isoformat(),
                    'source_head': subprocess.check_output(
                        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'files': files, 'excluded_candidates': excluded}
        verify(manifest)
        MANIFEST.parent.mkdir(exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
