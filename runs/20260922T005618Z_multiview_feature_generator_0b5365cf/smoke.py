"""Read-only real-data interface smoke; no y/URL/model/optimizer."""
import ast, json, struct, zipfile
from pathlib import Path
import numpy as np
from ta_wf_next.traffic_views import generate_views

RUN=Path(__file__).resolve().parent; ROOT=RUN.parents[1]
DATA=Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
def mmap_x(path):
    with zipfile.ZipFile(path) as z: i=z.getinfo('X.npy'); assert i.compress_type==0
    with path.open('rb') as f:
        f.seek(i.header_offset); h=struct.unpack('<IHHHHHIIIHH',f.read(30)); f.seek(h[-2]+h[-1],1); assert f.read(6)==b'\x93NUMPY'
        major,_=f.read(2); n=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
        hdr=ast.literal_eval(f.read(n).decode('latin1').strip()); off=f.tell()
    assert hdr['descr']=='<f8' and not hdr['fortran_order']
    return np.memmap(path,mode='r',dtype='<f8',offset=off,shape=hdr['shape'])

results={}
for rel in ['VersionDrift/048/train.npz','NetworkDrift/train.npz','BehaviorDrift/subpage.npz']:
    x=mmap_x(DATA/rel); ids=np.linspace(0,len(x)-1,128,dtype=int); rows=[]
    for i in ids:
        row=x[i,:5000]
        v=generate_views(row,input_kind='signed_timestamp',budget=5000,enable_timing=True)
        d=generate_views(np.sign(row),input_kind='direction',budget=5000)
        assert v.packet_direction==d.packet_direction and v.runs==d.runs
        assert v.select('packet_direction','exact_runs','direction_windows')
        assert v.timing is not None and v.size is None
        rows.append(dict(row=int(i),observed=v.runs.observed_count,runs=len(v.runs.runs),
                         invalid_intervals=v.timing.invalid_same_direction_intervals,
                         invalid_spans=v.timing.invalid_run_spans))
    results[rel]=dict(rows_checked=len(rows),rows=rows)
out=dict(files=results,labels_read=False,urls_read=False,training=0,scoring=0,
         time_channel='diagnostic only; cross-sign intervals not generated; missing/invalid same-sign intervals are None')
(RUN/'artifacts/real_smoke.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:{'rows_checked':v['rows_checked'],'invalid_intervals':sum(r['invalid_intervals'] for r in v['rows'])} for k,v in results.items()}))
