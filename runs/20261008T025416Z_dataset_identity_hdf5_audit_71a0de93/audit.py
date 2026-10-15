#!/usr/bin/env python3
import ast, hashlib, json, struct, time, zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import numpy as np
import h5py

RUN = Path(__file__).resolve().parent
BASE = Path('/mnt/data2/ren/datasets')
H5ROOT = BASE / 'shusterman_concept_drift_v2'
NPZROOTS = {
 'TemporalDrift': BASE/'TemporalDrift', 'VersionDrift': BASE/'VersionDrift',
 'NetworkDrift': BASE/'NetworkDrift', 'BehaviorDrift': BASE/'BehaviorDrift',
 'Defense': BASE/'Defense', 'OpenWorld': BASE/'OpenWorld', 'AWF': BASE/'awf_concept_drift_200w'
}

def scalar(x):
    if isinstance(x, bytes):
        return x.decode('utf-8', 'replace')
    if hasattr(x, 'item'):
        return scalar(x.item())
    return str(x)

def h5_text_array(ds):
    a = ds.asstr()[...]
    return [str(x) for x in a.tolist()]

def canonical_hash(row):
    a = np.asarray(row)
    valid = np.all(np.isfinite(a) & (a == np.trunc(a)) & (a >= -2147483648) & (a <= 2147483647))
    if not valid: return None
    return hashlib.sha256(a.astype('<i4', copy=False).tobytes()).digest()

def audit_hdf5():
    records=[]; first_seen={}; collisions=Counter(); examples=[]; duplicate_within=Counter(); all_hash_count=0
    for path in sorted(H5ROOT.rglob('*.h5')):
        rel=str(path.relative_to(BASE)); comp=path.parent.parent.name; date=path.parent.name; split=path.stem
        with h5py.File(path, 'r') as f:
            keys=sorted(f.keys())
            required={'site_name','start_date','tag','vector','version'}
            sites=h5_text_array(f['site_name']); dates=h5_text_array(f['start_date']); tags=h5_text_array(f['tag']); versions=h5_text_array(f['version'])
            v=f['vector']; local=set(); per_site=defaultdict(set); clean_site=defaultdict(set); zero=cap=invalid=internal_zero=0; lo=float('inf'); hi=float('-inf')
            for start in range(0,len(v),512):
                a=np.asarray(v[start:start+512])
                lo=min(lo,float(np.nanmin(a))); hi=max(hi,float(np.nanmax(a)))
                valid=np.all(np.isfinite(a) & (a==np.trunc(a)) & (a>=-2147483648) & (a<=2147483647),axis=1)
                invalid += int((~valid).sum())
                nz=a!=0; lengths=np.where(nz.any(1), a.shape[1]-np.argmax(nz[:,::-1],axis=1),0)
                zero += int((lengths==0).sum()); cap += int((lengths==a.shape[1]).sum())
                internal_zero += int(np.any((~nz) & (np.arange(a.shape[1])[None,:] < lengths[:,None]),axis=1).sum())
                for j,row in enumerate(a):
                    d=canonical_hash(row)
                    if d is None: continue
                    local.add(d); site=sites[start+j]; per_site[site].add(d)
                    if lengths[j] > 0: clean_site[site].add(d)
                    all_hash_count += 1
                    if d in first_seen:
                        psite, prel, pcomp, pdate, psplit = first_seen[d]
                        if prel == rel: cat='within_file'
                        elif pcomp != comp: cat='cross_component'
                        elif pdate != date: cat='cross_date'
                        else: cat='same_date_train_test'
                        collisions[cat]+=1
                        if psite != site: collisions['conflicting_site_rows'] += 1
                        if len(examples)<40 and cat!='within_file': examples.append({'category':cat,'first':prel,'current':rel,'first_site':psite,'current_site':site})
                    else: first_seen[d]=(site,rel,comp,date,split)
            duplicate_within[rel]=len(v)-len(local)
            records.append({'path':rel,'component':comp,'date':date,'split':split,'keys':keys,'required_keys_present':required.issubset(keys),'rows':len(v),'vector_shape':list(v.shape),'vector_dtype':str(v.dtype),'field_schema':{k:{'shape':list(f[k].shape),'dtype':str(f[k].dtype)} for k in keys},'site_count':len(set(sites)),'sites':sorted(set(sites)),'date_values':sorted(set(dates)),'tag_values':sorted(set(tags)),'version_values':sorted(set(versions)),'rows_per_site':dict(sorted(Counter(sites).items())),'unique_vectors_per_site':{s:len(per_site[s]) for s in sorted(per_site)},'nonzero_unique_vectors_per_site':{s:len(clean_site[s]) for s in sorted(per_site)},'all_zero_rows':zero,'at_length_cap':cap,'invalid_rows':invalid,'internal_zero_rows':internal_zero,'value_min':lo,'value_max':hi,'within_file_duplicate_rows':duplicate_within[rel]})
    return {'records':records,'collisions':dict(collisions),'collision_examples':examples,'canonical_rows_hashed':all_hash_count}

def npy_header(data):
    if data[:6] != b'\x93NUMPY': raise ValueError('not npy')
    ver=tuple(data[6:8]); off=8
    if ver==(1,0): n=struct.unpack('<H',data[8:10])[0]; off=10
    else: n=struct.unpack('<I',data[8:12])[0]; off=12
    h=ast.literal_eval(data[off:off+n].decode('latin1'))
    return h,off+n

def npz_label_summary(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist(); label_name='y.npy' if 'y.npy' in names else ('labels.npy' if 'labels.npy' in names else None)
        result={'path':str(path),'members':names,'label_key':label_name}
        if label_name is None: return result
        raw=z.read(label_name); h,off=npy_header(raw); result['label_schema']={'descr':h.get('descr'),'shape':list(h.get('shape',())),'fortran_order':h.get('fortran_order')}
        try:
            vals=np.load(BytesIO(raw),allow_pickle=True)
            flat=[scalar(x) for x in np.asarray(vals).reshape(-1).tolist()]
            c=Counter(flat); result['label_count']=len(flat); result['unique_label_count']=len(c); result['labels']=sorted(c); result['label_counts']=dict(sorted(c.items()))
        except Exception as e: result['label_read_error']=f'{type(e).__name__}:{e}'
        return result

def sampled_row_hashes(path, limit=512):
    # Load only small AWF files for deterministic sample checks; large compressed NPZ arrays are not fully decoded here.
    try:
        with np.load(path, allow_pickle=False) as z:
            key='data' if 'data' in z.files else ('X' if 'X' in z.files else None)
            if key is None: return {'path':str(path),'status':'no_data_key'}
            a=z[key]; n=a.shape[0]; idx=sorted(set(list(range(min(limit//2,n))) + list(range(max(0,n-limit//2),n))))
            hs=[hashlib.sha256(np.asarray(a[i]).tobytes()).hexdigest() for i in idx]
            return {'path':str(path),'status':'sampled','rows':int(n),'sample_count':len(hs),'sample_hashes':hs}
    except Exception as e: return {'path':str(path),'status':'error','error':f'{type(e).__name__}:{e}'}

def main():
    t=time.time(); h5=audit_hdf5(); summaries={}; samples=[]; all_labels={}
    for name,root in NPZROOTS.items():
        files=sorted(root.rglob('*.npz')); rows=[]; label_union=set(); errors=[]
        for p in files:
            try:
                r=npz_label_summary(p); rows.append(r)
                label_union.update(r.get('labels',[]))
                if name=='AWF' or p.stat().st_size < 300*1024*1024: samples.append(sampled_row_hashes(p))
            except Exception as e: errors.append({'path':str(p),'error':f'{type(e).__name__}:{e}'})
        summaries[name]={'file_count':len(files),'files':rows,'raw_label_union':sorted(label_union),'errors':errors}
        all_labels[name]=label_union
    h5sites=set(s for r in h5['records'] for s in r['sites'])
    label_overlap={name:sorted(h5sites & vals) for name,vals in all_labels.items() if vals and all(isinstance(x,str) for x in vals)}
    cross_label_raw={a:{b:sorted(all_labels[a]&all_labels[b]) for b in all_labels if b>a and all_labels[a] and all_labels[b]} for a in all_labels}
    duplicate_sample_groups=defaultdict(list)
    for s in samples:
        for h in s.get('sample_hashes',[]): duplicate_sample_groups[h].append(s['path'])
    repeated={h:paths for h,paths in duplicate_sample_groups.items() if len(set(paths))>1}
    out={'scope':'CPU read-only identity/schema/duplicate audit; no training/TTA/future scoring','hdf5':h5,'npz':summaries,'raw_label_intersection':cross_label_raw,'hdf5_string_site_overlap_with_npz_labels':label_overlap,'npz_sample_hash_overlap':{'groups':list(repeated.values())[:100],'group_count':len(repeated),'scope':'only deterministic sampled rows; not full cross-container proof'},'elapsed_seconds':round(time.time()-t,2)}
    (RUN/'artifacts'/'hdf5_identity_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines=['# 第二轮数据身份与重复审计结果','',f"- HDF5 文件：{len(h5['records'])}；规范化行 hash：{h5['canonical_rows_hashed']}",f"- HDF5 重复计数：{h5['collisions']}",f"- NPZ 数据集：{len(summaries)}；AWF/小文件执行确定性抽样 hash。",'- 未训练、未适配、未访问外部未来测试。','', '## 解释边界','', '- HDF5 内部字段和 Shusterman 的网站/日期/浏览器/网络元数据已解析。', '- 原始标签集合交集只表示数值/字符串碰撞提示，不能证明不同数据集的网站身份相同。', '- NPZ 大文件未做全量样本 hash；抽样相同不等于确认重复，未发现也不等于证明无重复。', '- 只有通过人工或来源 manifest 建立跨数据集身份映射后，才能训练共享分类头。', '', '详见 `artifacts/hdf5_identity_audit.json`。']
    (RUN/'artifacts'/'summary.md').write_text('\n'.join(lines)+'\n'); (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n'); print(json.dumps({'hdf5_files':len(h5['records']),'collisions':h5['collisions'],'seconds':out['elapsed_seconds']},ensure_ascii=False))
if __name__=='__main__': main()
