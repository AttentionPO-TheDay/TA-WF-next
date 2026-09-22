"""Read only X headers and selected rows; no labels, predictions or checkpoints."""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import zipfile
import numpy as np
from ta_wf_next.burst_tokens import encode_directions

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
DATA=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
def mmap_x(path):
    with zipfile.ZipFile(path) as z:
        i=z.getinfo('X.npy'); assert i.compress_type==0
    with path.open('rb') as f:
        f.seek(i.header_offset); h=struct.unpack('<IHHHHHIIIHH',f.read(30)); f.seek(h[-2]+h[-1],1)
        assert f.read(6)==b'\x93NUMPY'
        major,minor=f.read(2); assert major in (1,2)
        n=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
        header=ast.literal_eval(f.read(n).decode('latin1').strip()); offset=f.tell()
    assert header['descr']=='<f8' and not header['fortran_order']
    return np.memmap(path,mode='r',dtype='<f8',offset=offset,shape=header['shape'])

results={}
for name in ['VersionDrift/048/train.npz','NetworkDrift/train.npz','BehaviorDrift/subpage.npz']:
    x=mmap_x(DATA/name)
    indices=np.linspace(0,len(x)-1,128,dtype=int)
    rows=[]
    for i in indices:
        row=x[i,:5000]
        encoding=encode_directions(row,budget=5000,input_kind='signed_timestamp')
        expected=tuple(np.sign(row[row!=0]).astype(int))
        assert encoding.decode()==expected
        assert encoding==encode_directions(np.sign(row),budget=5000)
        coarse=encoding.coarse()
        assert len(coarse)==len(encoding.runs)
        for exact,token in zip(encoding.runs,coarse):
            assert 2**token.log2_count_bin<=exact.count<2**(token.log2_count_bin+1)
            assert exact.direction==token.direction
        rows.append(dict(row=int(i),observations=encoding.observed_count,tokens=len(coarse),end_reason=encoding.end_reason))
    results[name]=dict(checked_rows=len(rows),roundtrip_errors=0,interface_parity_errors=0,rows=rows)
demo=encode_directions([1]*4+[-1]*6+[1]*2+[-1]*3)
output=dict(files=results,example=dict(exact=[asdict(r) for r in demo.runs],coarse=[asdict(t) for t in demo.coarse()]),
    sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/ta_wf_next/burst_tokens.py',ROOT/'tests/test_burst_tokens.py',Path(__file__)]},
    training=0,labels_read=False,time_features=False,claim='interface and correctness only; no temporal stability or classification evidence')
with (RUN/'artifacts/smoke.json').open('x') as f: json.dump(output,f,indent=2)
print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in results.items()},indent=2))
