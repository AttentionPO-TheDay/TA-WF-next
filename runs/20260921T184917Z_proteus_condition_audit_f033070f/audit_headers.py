"""Read NPZ headers only; no arrays, models or training code loaded."""
import ast
import hashlib
import json
import struct
import zipfile
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
config = json.loads((ROOT / 'configs/datasets.json').read_text())
data_root = Path(config['data_root'])
prior_path = Path('/home/rbf/TA-WF/outputs/proteus_six_dataset_protocol_audit_exp_d839f7ce217f488a/evidence/local_npz_audit.json')
prior = json.loads(prior_path.read_text())
groups = ['TemporalDrift', 'VersionDrift', 'NetworkDrift', 'BehaviorDrift', 'OpenWorld', 'Defense']
files, errors = {}, []
for group in groups:
    for path in sorted((data_root / group).rglob('*.npz')):
        if path.name.startswith('tam_') or 'smoke' in path.name:
            continue
        rel = str(path.relative_to(data_root))
        members = {}
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                assert info.filename.endswith('.npy')
                with archive.open(info) as handle:
                    assert handle.read(6) == b'\x93NUMPY'
                    major, minor = handle.read(2)
                    assert major in (1, 2, 3)
                    width = 2 if major == 1 else 4
                    length = struct.unpack('<H' if width == 2 else '<I', handle.read(width))[0]
                    assert length <= 65536
                    header = ast.literal_eval(handle.read(length).decode('utf8' if major == 3 else 'latin1').strip())
                members[info.filename] = dict(shape=list(header['shape']), dtype=header['descr'],
                    fortran_order=header['fortran_order'], zip_compression=info.compress_type,
                    zip_crc32=f'{info.CRC:08x}', member_bytes=info.file_size)
        old = prior['files'].get(rel)
        same = old is not None and old['members'] == members and old['file_bytes'] == path.stat().st_size
        if not same:
            errors.append(f'Prior archive metadata mismatch: {rel}')
        files[rel] = dict(file_bytes=path.stat().st_size, members=members,
            prior_archive_metadata_match=same, inherited_labels=old.get('labels') if same else None)
result = dict(scope='ZIP directory and NPY headers only; arrays not requested',
    caveat='Stored CRC/shape/size match is not fresh payload verification. Label statistics inherited, not reverified.',
    prior_path=str(prior_path), prior_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
    files=files, file_count=len(files), errors=errors,
    access=dict(array_values_loaded=0, labels_loaded=0, training=0, scoring=0, external_wtt_awf_opened=False))
with (RUN / 'artifacts/header_audit.json').open('x') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
print(json.dumps(dict(file_count=len(files), errors=errors)))
if errors:
    raise SystemExit(1)
