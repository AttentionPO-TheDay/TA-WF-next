"""Small standard-library experiment registry; never starts training."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['id', 'name', 'status', 'question', 'summary']


def payload_root(root):
    path = root / 'configs/storage.json'
    if not path.exists():
        return None
    config = json.loads(path.read_text())
    destination = Path(config['run_payload_root'])
    if config.get('schema_version') != 1 or not destination.is_absolute() or not destination.is_dir():
        raise SystemExit('Storage config requires schema_version=1 and an existing absolute run_payload_root directory.')
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    new = sub.add_parser('new')
    new.add_argument('--name', required=True)
    new.add_argument('--question', required=True)
    new.add_argument('--resume-id', help='Register an unregistered directory left by an interrupted creation.')
    status = sub.add_parser('status')
    status.add_argument('--id', required=True)
    status.add_argument('--state', required=True, choices=['planned', 'running', 'completed', 'failed', 'stopped'])
    status.add_argument('--summary', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'check':
        config = json.loads((root / 'configs/datasets.json').read_text())
        for name, item in config['datasets'].items():
            path = Path(config['data_root']) / item['path']
            if not path.is_dir():
                raise SystemExit(f'Missing: {path}')
            print(f'{name}: OK')
        with (root / 'EXPERIMENTS.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len({r['id'] for r in rows}) == len(rows)
        assert all((root / 'runs' / r['id'] / 'PLAN.md').is_file() for r in rows)
        destination = payload_root(root)
        if destination is not None:
            print(f'New experiment payloads: {destination}')
        print(f'{len(rows)} experiments; no arrays, labels, or checkpoints loaded.')
        return
    with (root / '.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = root / 'EXPERIMENTS.csv'
        with index.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        if args.command == 'new':
            if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,47}', args.name):
                raise SystemExit('Use a short lowercase name, digits, underscores or hyphens.')
            run_id = args.resume_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + args.name + '_' + uuid.uuid4().hex[:8]
            run = root / 'runs' / run_id
            if args.resume_id:
                if not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z_' + re.escape(args.name) + r'_[0-9a-f]{8}', run_id):
                    raise SystemExit('resume-id does not match the requested experiment name')
                if any(r['id'] == run_id for r in rows):
                    raise SystemExit('Experiment already registered')
                if not run.is_dir() or run.is_symlink():
                    raise SystemExit('Interrupted experiment directory missing or symlinked')
                if not all((run / p).exists() for p in ['PLAN.md', 'RESULTS.md', 'config.json', 'logs', 'checkpoints', 'artifacts']):
                    raise SystemExit('Restore the interrupted run files before registering')
                if json.loads((run / 'config.json').read_text()).get('id') != run_id:
                    raise SystemExit('Config ID must match resume-id')
            else:
                destination = payload_root(root)
                run.mkdir(parents=True, exist_ok=False)
                # Keep logs in a physical directory named ``lgs`` and expose
                # the stable ``logs`` path as a relative symlink.  Some
                # shared workspaces mount/overlay ``logs`` and reject a
                # direct mkdir, while the protocol requires both paths to be
                # available to run-local code.
                (run / 'lgs').mkdir()
                (run / 'logs').symlink_to('lgs', target_is_directory=True)
                for folder in ['checkpoints', 'artifacts']:
                    if destination is None:
                        (run / folder).mkdir()
                    else:
                        target = destination / run_id / folder
                        target.mkdir(parents=True, exist_ok=False)
                        (run / folder).symlink_to(target, target_is_directory=True)
                (run / 'PLAN.md').write_text('# ' + run_id + '\n\n问题：' + args.question + '\n\n状态：planned；配置尚未冻结。\n\n待填写：数据及代码版本、标签权限、对照、指标、seeds、选模规则、预算、停止条件。\n')
                (run / 'config.json').write_text(json.dumps({'id': run_id, 'status': 'draft'}, indent=2) + '\n')
                (run / 'RESULTS.md').write_text('# 实验结果\n\n尚未执行。\n')
            rows.append(dict(zip(FIELDS, [run_id, args.name, 'planned', args.question, ''])))
            print(run)
        else:
            matches = [r for r in rows if r['id'] == args.id]
            if len(matches) != 1:
                raise SystemExit('Unknown or ambiguous experiment ID')
            matches[0].update(status=args.state, summary=args.summary)
        temporary = index.with_suffix('.csv.tmp')
        with temporary.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(index)


if __name__ == '__main__':
    main()
