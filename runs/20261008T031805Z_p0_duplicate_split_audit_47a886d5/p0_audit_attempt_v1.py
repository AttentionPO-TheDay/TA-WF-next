#!/usr/bin/env python3
import ast, hashlib, json, struct, time, zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import numpy as np
import h5py
RUN=Path(__file__).resolve().parent
BASE=Path('/mnt/data2/ren/datasets')
CORE={'TemporalDrift':BASE/'TemporalDrift','VersionDrift':BASE/'VersionDrift','NetworkDrift':BASE/'NetworkDrift','BehaviorDrift':BASE/'BehaviorDrift'}
H5ROOT=BASE/'shusterman_concept_drift_v2'
def text_arr(ds): return [str(x) for x in ds.asstr()[...].tolist()]
def canonical_numeric_bytes(a):
    a=np.asarray(a)
    if not np.all(np.isfinite(a)): return None
    if np.issubdtype(a.dtype,np.floating):
        if not np.all(a==np.trunc(a)) or np.any(a < -2147483648) or np.any(a > 2147483647): return None
        out=a.astype('<i4',copy=False)
    else:
        out=a.astype(a.dtype.newbyteorder('<'),copy=False)
    return out.tobytes()
def digest_row(a):
    b=canonical_numeric_bytes(a); return None if b is None else hashlib.sha256(b).hexdigest()
def h5_audit():
    seen={}; groups=defaultdict(list); collisions=Counter(); recs=[]; total=0
    for path in sorted(H5ROOT.rglob('*.h5')):
        rel=str(path.relative_to(BASE)); comp=path.parent.parent.name; date=path.parent.name; split=path.stem
        with h5py.File(path,'r') as f:
            keys=sorted(f.keys()); sites=text_arr(f['site_name']); v=f['vector']; zero=cap=invalid=internal_zero=0; local=Counter()
            for st in range(0,len(v),256):
                a=np.asarray(v[st:st+256]); nz=a!=0; lengths=np.where(nz.any(1),a.shape[1]-np.argmax(nz[:,::-1],axis=1),0); zero+=int((lengths==0).sum()); cap+=int((lengths==a.shape[1]).sum()); internal_zero+=int(np.any((~nz)&(np.arange(a.shape[1])[None,:]<lengths[:,None]),axis=1).sum())
                for j,row in enumerate(a):
                    idx=st+j; d=digest_row(row)
                    if d is None: invalid+=1; continue
                    total+=1; local[d]+=1; occ={'file':rel,'row':idx,'site':sites[idx],'component':comp,'date':date,'split':split,'length':int(lengths[j]),'all_zero':bool(lengths[j]==0)}
                    if d in seen:
                        prev=seen[d]; cat='within_file' if prev['file']==rel else ('cross_component' if prev['component']!=comp else ('cross_date' if prev['date']!=date else 'same_date_train_test')); collisions[cat]+=1
                        if prev['site']!=occ['site']: collisions['conflicting_site_rows']+=1
                        if len(groups[d])<20: groups[d].append(prev) if not groups[d] else None; groups[d].append(occ)
                    else: seen[d]=occ
            recs.append({'file':rel,'component':comp,'date':date,'split':split,'keys':keys,'rows':len(v),'vector_shape':list(v.shape),'vector_dtype':str(v.dtype),'site_count':len(set(sites)),'all_zero_rows':zero,'at_length_cap':cap,'invalid_rows':invalid,'internal_zero_rows':internal_zero,'within_file_duplicate_rows':sum(n-1 for n in local.values() if n>1)})
    p0=[]
    for d,occ in groups.items():
        vals=list({(x['file'],x['row']):x for x in occ}.values()); cats=set()
        for i,a in enumerate(vals):
            for b in vals[i+1:]:
                cats.add('cross_component' if a['component']!=b['component'] else ('cross_date' if a['date']!=b['date'] else ('same_date_train_test' if a['split']!=b['split'] else 'within_file')))
        if cats & {'same_date_train_test','cross_component','cross_date'}: p0.append({'digest':d,'categories':sorted(cats),'occurrences':vals[:20],'occurrence_count':len(vals),'all_zero':all(x['all_zero'] for x in vals),'lengths':sorted(set(x['length'] for x in vals))})
    return {'records':recs,'collisions':dict(collisions),'p0_duplicate_groups':p0,'canonical_rows_hashed':total,'files':len(recs)}
def header(stream):
    lead=stream.read(10)
    if lead[:6]!=b'\x93NUMPY': raise ValueError('not npy')
    ver=tuple(lead[6:8])
    if ver==(1,0):
        n=struct.unpack('<H',lead[8:10])[0]
    else:
        lead += stream.read(2)
        n=struct.unpack('<I',lead[8:12])[0]
    return ast.literal_eval(stream.read(n).decode('latin1').strip())
def mmap_x(path):
    with zipfile.ZipFile(path) as z:
        info=z.getinfo('X.npy')
        if info.compress_type != 0: raise ValueError('X.npy is compressed')
        with path.open('rb') as f:
            f.seek(info.header_offset); local=f.read(30)
            sig,ver,flag,method,tm,dt,crc,csize,usize,fnlen,xlen=struct.unpack('<4s5H3I2H',local)
            data_offset=info.header_offset+30+fnlen+xlen
            f.seek(data_offset); pre=f.read(12)
            if pre[:6] != b'\x93NUMPY': raise ValueError('bad X.npy header')
            npver=tuple(pre[6:8])
            if npver==(1,0): hlen=struct.unpack('<H',pre[8:10])[0]; hoff=data_offset+10
            else: hlen=struct.unpack('<I',pre[8:12])[0]; hoff=data_offset+12
            meta=ast.literal_eval(f.read(hlen).decode('latin1').strip())
            body=hoff+hlen
        if meta['fortran_order']: raise ValueError('fortran array')
        return np.memmap(path,mode='r',dtype=np.dtype(meta['descr']),offset=body,shape=tuple(meta['shape']))
def label_array(z,key): return np.load(BytesIO(z.read(key+'.npy')),allow_pickle=True).reshape(-1)
def npz_audit():
    seen={}; groups=defaultdict(list); counts=Counter(); files=[]; errors=[]; total=0
    for ds,root in CORE.items():
        for path in sorted(p for p in root.rglob('*.npz') if not p.name.startswith('tam_')):
            rel=str(path.relative_to(BASE)); role=path.stem; nrows=0; zeros=invalid=label_conf=0; local=Counter()
            try:
                with zipfile.ZipFile(path) as z:
                    names=z.namelist(); y=label_array(z,'y')
                    x=mmap_x(path); shape=x.shape; descr=str(x.dtype); rows,width=shape; nrows=rows
                    if len(y)!=rows or len(shape)!=2: raise ValueError(f'shape/labels {shape}/{len(y)}')
                    for st in range(0,rows,128):
                            a=x[st:st+128]
                            for j,row in enumerate(a):
                                idx=st+j; total+=1; d=digest_row(row); lab=str(y[idx].item() if hasattr(y[idx],'item') else y[idx]); occ={'dataset':ds,'file':rel,'row':idx,'role':role,'label':lab}
                                if d is None: invalid+=1; continue
                                local[d]+=1; zeros+=int(np.all(row==0))
                                if d in seen:
                                    prev=seen[d]; cat='same_file' if prev['file']==rel else ('cross_dataset' if prev['dataset']!=ds else ('cross_role' if prev['role']!=role else 'same_role')); counts[cat]+=1
                                    if prev['label']!=lab: counts['conflicting_labels']+=1; label_conf+=1
                                    if len(groups[d])<20:
                                        if not groups[d]: groups[d].append(prev)
                                        groups[d].append(occ)
                                else: seen[d]=occ
                files.append({'dataset':ds,'file':rel,'role':role,'rows':nrows,'label_count':len(set(str(x.item() if hasattr(x,'item') else x) for x in y.tolist())),'shape':[nrows,width],'dtype':descr,'all_zero_rows':zeros,'invalid_rows':invalid,'within_file_duplicate_rows':sum(n-1 for n in local.values() if n>1),'conflicting_duplicate_labels':label_conf})
            except Exception as e: errors.append({'dataset':ds,'file':rel,'error':f'{type(e).__name__}:{e}'})
    p0=[]
    for d,occ in groups.items():
        vals=list({(x['file'],x['row']):x for x in occ}.values())
        if len(vals)>1: p0.append({'digest':d,'occurrences':vals,'categories':sorted(set('cross_dataset' if a['dataset']!=b['dataset'] else ('cross_role' if a['role']!=b['role'] else 'same_file') for i,a in enumerate(vals) for b in vals[i+1:])),'labels':sorted(set(x['label'] for x in vals)),'label_conflict':len({x['label'] for x in vals})>1})
    return {'files':files,'errors':errors,'collisions':dict(counts),'p0_duplicate_groups':p0,'rows_hashed':total,'file_count':len(files)}
def main():
    t=time.time(); h=h5_audit(); n=npz_audit(); out={'scope':'P0 CPU read-only full exact duplicate and split audit; no training/TTA/future scoring','hdf5':h,'npz':n,'elapsed_seconds':round(time.time()-t,2)}
    (RUN/'artifacts'/'p0_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines=['# P0 重复与 split 泄漏审计','',f"- Shusterman HDF5：{h['files']} 文件，规范化行 hash {h['canonical_rows_hashed']} 条",f"- Proteus-family NPZ：{n['file_count']} 文件，行 hash {n['rows_hashed']} 条",f"- HDF5 重复计数：{h['collisions']}",f"- NPZ 重复计数：{n['collisions']}",f"- 读取错误：HDF5 0；NPZ {len(n['errors'])}",'- 全程 CPU 只读；未训练、未适配、未访问外部未来测试。','', '## 裁决边界','', '- 只有明确列出的重复才进入 P0 处理候选；不自动删除或改写共享数据。', '- 重复向量的标签冲突需要按 occurrence 审核；在隔离完成前不建立联合训练 manifest。', '- 该 run 不创建新的 split，也不根据 query 标签选模。', '', '完整 occurrence 记录见 `artifacts/p0_audit.json`。']
    (RUN/'artifacts'/'summary.md').write_text('\n'.join(lines)+'\n'); (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n'); print(json.dumps({'hdf5_collisions':h['collisions'],'npz_collisions':n['collisions'],'errors':len(n['errors']),'seconds':out['elapsed_seconds']},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
