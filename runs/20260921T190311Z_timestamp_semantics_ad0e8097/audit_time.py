import ast
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from collections import Counter
import numpy as np

RUN=Path(__file__).resolve().parent
DATA=Path(json.loads((RUN.parents[1]/'configs/datasets.json').read_text())['data_root'])
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
    return np.memmap(path,mode='r',dtype='<f8',shape=header['shape'],offset=offset)

paths=[f'VersionDrift/{v}/{r}.npz' for v in ['045','046','047','048'] for r in ['train','valid']]
paths+=['VersionDrift/048/drift.npz']+[f'NetworkDrift/{r}.npz' for r in ['train','valid','SG','JP','USA','DE','UK']]+['BehaviorDrift/subpage.npz']
output={}
for path in paths:
    x=mmap_x(DATA/path); c=Counter(); samples=[]; examples=[]; maximum=0.; frontier_max=0.
    sample_ids=set(np.linspace(0,len(x)-1,min(128,len(x)),dtype=int).tolist())
    for j,row in enumerate(x):
        c['rows']+=1
        if not np.isfinite(row).all(): c['nonfinite_rows']+=1; continue
        nz=np.flatnonzero(row!=0)
        if not len(nz): c['empty_rows']+=1; continue
        if len(nz)!=nz[-1]+1: c['internal_or_leading_zero_rows']+=1
        r=row[nz]; t=np.abs(r); signs=np.sign(r); d=np.diff(t); neg=d<0; change=signs[1:]!=signs[:-1]
        c['valid_values']+=len(r); c['edges']+=len(d); c['negative_edges']+=int(neg.sum()); c['negative_rows']+=int(neg.any())
        c['same_direction_edges']+=int((~change).sum()); c['cross_direction_edges']+=int(change.sum())
        c['negative_same_direction_edges']+=int((neg&~change).sum()); c['negative_cross_direction_edges']+=int((neg&change).sum())
        c['zero_edges']+=int((d==0).sum())
        for threshold in [1e-9,1e-6,1e-5,1e-4,1e-3,1e-2,.1,1.,10.]:
            c['negative_gt_'+str(threshold)]+=int((d < -threshold).sum())
        for sign in [-1,1]:
            sub=np.diff(t[signs==sign]); c[f'direction_{sign}_negative_edges']+=int((sub<0).sum()); c[f'direction_{sign}_negative_rows']+=int((sub<0).any())
        if neg.any():
            magnitude=float(-d.min()); maximum=max(maximum,magnitude)
            if len(examples)<8:
                k=int(np.argmin(d)); examples.append(dict(row=j,index=int(nz[k]),signed_values=r[max(0,k-2):k+4].tolist(),delta=float(d[k]),direction_switch=bool(change[k])))
        frontier_max=max(frontier_max,float(np.max(np.maximum.accumulate(t)-t)))
        starts=np.r_[0,np.flatnonzero(change)+1]; ends=np.r_[starts[1:]-1,len(r)-1]
        durations=t[ends]-t[starts]
        c['runs']+=len(starts); c['negative_duration_runs']+=int((durations<0).sum()); c['zero_duration_runs']+=int((durations==0).sum())
        if j in sample_ids:
            order=np.argsort(t,kind='stable'); sorted_signs=signs[order]
            rounding={str(dec):int((np.diff(np.round(t,dec))<0).sum()) for dec in [3,4,5,6]}
            samples.append(dict(row=j,negative_edges=int(neg.sum()),rounding_negative_edges=rounding,
                sign_positions_changed=int((sorted_signs!=signs).sum()),
                run_count_before=len(starts),run_count_after=int(np.count_nonzero(np.diff(sorted_signs)))+1))
    output[path]=dict(counts=dict(c),max_negative_seconds=maximum,max_frontier_backtrack_seconds=frontier_max,examples=examples,samples=samples)
    print(path,json.dumps(dict(rows=c['rows'],negative_rows=c['negative_rows'],negative_edges=c['negative_edges'],same=c['negative_same_direction_edges'],cross=c['negative_cross_direction_edges'],maximum=maximum)),flush=True)
result=dict(files=output,access=dict(labels_read=False,training=0,data_modified=False),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),notes=['Statistics count file views, not unique traces','Zero padding excluded before differencing; internal zeros explicitly counted','Sorting/rounding only sensitivity measurements, not corrections'])
with (RUN/'artifacts/time_audit.json').open('x') as f: json.dump(result,f,indent=2)
print('COMPLETE',flush=True)
