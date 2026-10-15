"""Independent P0 audit. Only numpy/h5py borrowed from the archived environment.
NPZ memmap header logic explicitly adapted from local 20260921 identity audit;
no imports from archived training code. No models or performance evaluation.
"""
import ast, csv, hashlib, json, os, struct, time, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from io import BytesIO
from contextlib import ExitStack
import numpy as np
import h5py
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
CFG=json.loads((ROOT/'configs/datasets.json').read_text())
BASE=Path(CFG['data_root']); OUT=RUN/'artifacts'
VIEWS=('raw10000','float32_first5000','direction_first5000')
START=time.monotonic()

def deadline():
    if time.monotonic()-START > 1800: raise TimeoutError('1800s audit budget')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def mmap_x(path):
    with zipfile.ZipFile(path) as z:
        info=z.getinfo('X.npy')
        if info.compress_type != zipfile.ZIP_STORED: raise ValueError('expected uncompressed X.npy')
    with path.open('rb') as f:
        f.seek(info.header_offset)
        h=struct.unpack('<IHHHHHIIIHH',f.read(30))
        assert h[0]==0x04034b50
        f.seek(h[-2]+h[-1],1)
        assert f.read(6)==b'\x93NUMPY'
        major,minor=f.read(2)
        assert major in (1,2)
        n=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
        meta=ast.literal_eval(f.read(n).decode('latin1').strip()); offset=f.tell()
    assert meta['descr']=='<f8' and not meta['fortran_order'] and len(meta['shape'])==2 and meta['shape'][1]==10000
    return np.memmap(path,mode='r',dtype='<f8',offset=offset,shape=meta['shape'])

def npz_labels(path):
    with zipfile.ZipFile(path) as z: y=np.load(BytesIO(z.read('y.npy')),allow_pickle=False)
    assert y.ndim==1 and np.isfinite(y).all() and np.equal(y,np.trunc(y)).all()
    return y.astype(np.int64)

def frozen_stat(path):
    s=path.stat(); return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino}

def core_audit():
    keys=['proteus_temporal','proteus_version','proteus_network','proteus_behavior']
    paths=[p for key in keys for p in sorted((BASE/CFG['datasets'][key]['path']).rglob('*.npz')) if not p.name.startswith('tam_')]
    arrays=[]; labels=[]; meta=[]; ids={v:[] for v in VIEWS}; buckets={v:{} for v in VIEWS}; reps={v:[] for v in VIEWS}
    def viewbytes(v,row):
        if v=='raw10000': return row.tobytes()
        if v=='float32_first5000': return row[:5000].astype('<f4').tobytes()
        return np.sign(row[:5000]).astype('i1').tobytes()
    def register(v,raw,fi,ri):
        digest=hashlib.sha256(raw).digest()
        for cid in buckets[v].get(digest,()):
            p,j=reps[v][cid]
            # Byte verification on each SHA256 match, preserving exact equality semantics.
            if viewbytes(v,arrays[p][j])==raw: return cid
        cid=len(reps[v]); reps[v].append((fi,ri)); buckets[v].setdefault(digest,[]).append(cid); return cid
    for fi,p in enumerate(paths):
        deadline(); before=frozen_stat(p); x=mmap_x(p); y=npz_labels(p); assert len(y)==len(x)
        arrays.append(x); labels.append(y); rec={'path':str(p.relative_to(BASE)),'dataset':p.relative_to(BASE).parts[0],'role':p.stem,'rows':len(x),'source_stat':before,'all_zero_rows':0,'nonfinite_rows':0,'internal_zero_rows':0,'float32_overflow_rows':0,'classes':int(len(np.unique(y)))}
        vi={v:np.full(len(x),-1,dtype=np.int64) for v in VIEWS}
        for start in range(0,len(x),256):
            deadline(); a=x[start:start+256]; finite=np.isfinite(a).all(1); zero=np.all(a==0,axis=1); nz=a!=0; lengths=np.where(nz.any(1),a.shape[1]-np.argmax(nz[:,::-1],axis=1),0)
            rec['all_zero_rows']+=int(zero.sum());rec['nonfinite_rows']+=int((~finite).sum()); rec['internal_zero_rows']+=int(np.any((~nz)&(np.arange(a.shape[1])[None,:]<lengths[:,None]),axis=1).sum())
            with np.errstate(over='ignore'): prefix=a[:,:5000].astype('<f4')
            finite32=np.isfinite(prefix).all(1);rec['float32_overflow_rows']+=int((finite & ~finite32).sum())
            for j,row in enumerate(a):
                if not finite[j]: continue
                ri=start+j
                for v in VIEWS:
                    if v=='float32_first5000' and not finite32[j]: continue
                    vi[v][ri]=register(v,viewbytes(v,row),fi,ri)
        for v in VIEWS: ids[v].append(vi[v])
        assert frozen_stat(p)==before,'input changed during scan'
        meta.append(rec); print('CORE',fi+1,len(paths),rec['path'],rec['rows'],flush=True)
    file_index={m['path']:i for i,m in enumerate(meta)}
    results={}; group_records={}; quarantine=defaultdict(set)
    for v in VIEWS:
        counts=Counter(int(c) for values in ids[v] for c in values if c>=0)
        groups=defaultdict(list)
        for fi,values in enumerate(ids[v]):
            for ri,c in enumerate(values):
                if c>=0 and counts[int(c)]>1: groups[int(c)].append((fi,ri,int(labels[fi][ri])))
        pair_counts=Counter(); crossgroups=[]; exclusion_counts=Counter()
        with (OUT/f'core_{v}_duplicate_members.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['group_id','file','row','label','original_role','action','reason'])
            for cid,occ in sorted(groups.items()):
                fis=set(i for i,r,l in occ); labs=set(l for i,r,l in occ); ds=set(meta[i]['dataset'] for i in fis); roles=set(meta[i]['role'] for i in fis)
                if len(fis)>1:
                    for ai,a in enumerate(sorted(fis)):
                        for b in sorted(fis)[ai+1:]: pair_counts[(a,b)]+=1
                conflict=len(labs)>1
                # Same raw numeric IDs across datasets are not assumed identical site identities.
                reason_conflict='raw_label_disagreement_mapping_unresolved' if len(ds)>1 else 'within_dataset_label_conflict'
                protected=any(meta[i]['role']!='train' for i,r,l in occ); kept_train=False
                for fi,ri,lab in occ:
                    action='group_constraint_only'; reason='same_observed_input_group'
                    if conflict:
                        action='quarantine_candidate';reason=reason_conflict
                    elif meta[fi]['role']=='train':
                        if protected: action='exclude_from_future_joint_training';reason='matches_original_nontrain_role'
                        elif kept_train: action='exclude_from_future_joint_training';reason='duplicate_train_copy'
                        else: action='canonical_train_candidate';reason='first_deterministic_train_copy';kept_train=True
                    w.writerow([cid,meta[fi]['path'],ri,lab,meta[fi]['role'],action,reason]);exclusion_counts[action]+=1
                    if action in ('quarantine_candidate','exclude_from_future_joint_training'):quarantine[(fi,ri)].add(v+':'+reason)
                crossgroups.append({'group_id':cid,'members':len(occ),'files':sorted(meta[i]['path'] for i in fis),'labels':sorted(labs),'label_disagreement':conflict})
        results[v]={'unique_inputs':len(counts),'duplicate_groups':len(groups),'duplicate_member_rows':sum(len(g) for g in groups.values()),'duplicate_copies_beyond_first':sum(len(g)-1 for g in groups.values()),'label_disagreement_groups':sum(g['label_disagreement'] for g in crossgroups),'pairs':[{'a':meta[a]['path'],'b':meta[b]['path'],'shared_unique_inputs':n,'different_dataset':meta[a]['dataset']!=meta[b]['dataset'],'different_role':meta[a]['role']!=meta[b]['role']} for (a,b),n in sorted(pair_counts.items())],'action_counts':dict(exclusion_counts)}
        group_records[v]=crossgroups
        for fi,rec in enumerate(meta):
            vals=ids[v][fi]; rec[v+'_duplicate_copies']=int(sum(c>=0 for c in vals)-len(set(int(c) for c in vals if c>=0)))
    with (OUT/'core_quarantine_candidates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['file','row','label','original_role','reasons','applied'])
        for (fi,ri),reasons in sorted(quarantine.items()):w.writerow([meta[fi]['path'],ri,int(labels[fi][ri]),meta[fi]['role'],'|'.join(sorted(reasons)),False])
    # Every row is addressed using original file/row, retaining all original split boundaries.
    np.savez_compressed(OUT/'core_row_group_ids.npz',**{f'{v}_{i}':a for v,lists in ids.items() for i,a in enumerate(lists)})
    (OUT/'core_file_index.json').write_text(json.dumps(meta,indent=2)+'\n')
    (OUT/'core_duplicate_groups.json').write_text(json.dumps(group_records,indent=2)+'\n')
    assert all(frozen_stat(paths[i])==rec['source_stat'] for i,rec in enumerate(meta))
    return {'files':len(meta),'rows':sum(m['rows'] for m in meta),'views':results,'records':meta,'quarantine_candidate_rows':len(quarantine),'full_scan_errors':0,'byte_verified_matches':True,'limits':['exact observed input equality does not establish capture/session independence','TAM binning can create additional collisions not measured here','label mapping across dataset families remains unverified','candidate exclusions are not applied to existing splits/checkpoints']}

def h5_audit():
    # Integer-semantic canonicalization, identical for float/int storage.
    paths=sorted((BASE/CFG['datasets']['shusterman']['path']).rglob('*.h5'))
    seen={}; reps=[]; ids=[]; sites=[]; records=[]; totals=Counter()
    for fi,p in enumerate(paths):
        deadline(); before=frozen_stat(p)
        with h5py.File(p,'r') as f:
            labs=[str(x) for x in f['site_name'].asstr()[...]]; v=f['vector']; cids=np.full(len(v),-1,dtype=np.int64); lengths_all=np.zeros(len(v),np.int32); invalid=[]
            for start in range(0,len(v),256):
                deadline(); a=v[start:start+256]; valid=np.all(np.isfinite(a)&(a==np.trunc(a))&(a>=-2147483648)&(a<=2147483647),axis=1)
                nz=a!=0; lengths=np.where(nz.any(1),a.shape[1]-np.argmax(nz[:,::-1],axis=1),0); lengths_all[start:start+len(a)]=lengths
                for j,row in enumerate(a):
                    ri=start+j
                    if not valid[j]: invalid.append(ri);continue
                    raw=row.astype('<i4').tobytes();digest=hashlib.sha256(raw).digest()
                    cid=seen.get(digest)
                    if cid is None: cid=len(reps);seen[digest]=cid;reps.append((fi,ri))
                    cids[ri]=cid
            rec={'file':str(p.relative_to(BASE)),'component':p.parent.parent.name,'date_directory':p.parent.name,'role':p.stem,'rows':len(v),'source_stat':before,'invalid_rows':invalid,'all_zero_rows':int((lengths_all==0).sum()),'capped_rows':int((lengths_all==v.shape[1]).sum()),'vector_dtype':str(v.dtype)}
            ids.append(cids);sites.append(labs);rec['_lengths']=lengths_all;records.append(rec)
        assert frozen_stat(p)==before;print('HDF5',fi+1,len(paths),rec['file'],flush=True)
    counts=Counter(int(c) for a in ids for c in a if c>=0);groups=defaultdict(list)
    for fi,a in enumerate(ids):
        for ri,c in enumerate(a):
            if c>=0 and counts[int(c)]>1:groups[int(c)].append((fi,ri))
    quarantine={}; group_report=[];pairs=Counter(); labels_conflict=[]
    with ExitStack() as stack:
        handles={}
        def bytes_at(fi,ri):
            if fi not in handles: handles[fi]=stack.enter_context(h5py.File(paths[fi],'r'))
            return handles[fi]['vector'][ri].astype('<i4').tobytes()
        for cid,occ in sorted(groups.items()):
            a,b=reps[cid];raw=bytes_at(a,b)
            assert all(bytes_at(fi,ri)==raw for fi,ri in occ),'hash collision'
            labels=set(sites[fi][ri] for fi,ri in occ); nonzero=any(records[fi]['_lengths'][ri]>0 for fi,ri in occ);conflict=len(labels)>1
            fis=sorted(set(fi for fi,ri in occ))
            if nonzero:
                for ai,a in enumerate(fis):
                    for b in fis[ai+1:]:pairs[(a,b)]+=1
            if conflict:labels_conflict.append({'group_id':cid,'all_zero':not nonzero,'members':len(occ),'site_count':len(labels),'sites':sorted(labels)})
            main_occ=[(fi,ri) for fi,ri in occ if records[fi]['component']!='static_webpages_network']
            protected=any(records[fi]['role']=='test' for fi,ri in main_occ);first_train=False
            for fi,ri in occ:
                rec=records[fi];reason=None
                if rec['_lengths'][ri]==0:reason='all_zero_unusable_trace'
                elif conflict:reason='nonzero_exact_vector_conflicting_site_labels'
                elif rec['component']=='static_webpages_network':reason='auxiliary_component_not_in_main_pool'
                elif rec['role']=='train' and protected:reason='nonzero_matches_test_trace'
                elif rec['role']=='train':
                    if first_train:reason='duplicate_main_training_copy'
                    first_train=True
                if reason:quarantine[(fi,ri)]=reason
            group_report.append({'group_id':cid,'members':len(occ),'all_zero':not nonzero,'label_conflict':conflict,'sites':sorted(labels),'files':[records[fi]['file'] for fi in fis]})
    # All zero/invalid and all auxiliary rows are explicitly excluded in the candidate main-pool manifest.
    with (OUT/'shusterman_quarantine_candidates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['file','row','site','original_role','group_id','reason','applied'])
        for fi,rec in enumerate(records):
            for ri,c in enumerate(ids[fi]):
                reason=quarantine.get((fi,ri))
                if rec['_lengths'][ri]==0:reason='all_zero_unusable_trace'
                elif c<0:reason='invalid_noninteger_or_nonfinite_or_overflow'
                elif rec['component']=='static_webpages_network':reason='auxiliary_component_not_in_main_pool'
                if reason:quarantine[(fi,ri)]=reason;w.writerow([rec['file'],ri,sites[fi][ri],rec['role'],int(c),reason,False])
    with (OUT/'shusterman_duplicate_members.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['group_id','file','row','site','length','component','date_directory','original_role'])
        for c,occ in sorted(groups.items()):
            for fi,ri in occ:w.writerow([c,records[fi]['file'],ri,sites[fi][ri],int(records[fi]['_lengths'][ri]),records[fi]['component'],records[fi]['date_directory'],records[fi]['role']])
    for rec in records: rec.pop('_lengths')
    np.savez_compressed(OUT/'shusterman_row_group_ids.npz',**{f'file_{i}':a for i,a in enumerate(ids)})
    (OUT/'shusterman_file_index.json').write_text(json.dumps(records,indent=2)+'\n')
    (OUT/'shusterman_duplicate_groups.json').write_text(json.dumps(group_report,indent=2)+'\n')
    nonzero_pairs=[{'a':records[a]['file'],'b':records[b]['file'],'unique_nonzero_vectors':n} for (a,b),n in sorted(pairs.items())]
    zero_groups=[g for g in group_report if g['all_zero']];nonzero_conf=[g for g in labels_conflict if not g['all_zero']]
    # All pair comparisons, not only comparisons against the first global occurrence.
    test_train=[p for p in nonzero_pairs if Path(p['a']).parent==Path(p['b']).parent and Path(p['a']).stem!=Path(p['b']).stem]
    assert all(frozen_stat(paths[i])==r['source_stat'] for i,r in enumerate(records))
    return {'files':len(records),'rows':sum(r['rows'] for r in records),'all_zero_rows':sum(r['all_zero_rows'] for r in records),'invalid_rows':sum(len(r['invalid_rows']) for r in records),'duplicate_groups':len(groups),'duplicate_copies_beyond_first':sum(len(g)-1 for g in groups.values()),'zero_duplicate_groups':zero_groups,'label_conflict_groups':labels_conflict,'nonzero_label_conflict_groups':len(nonzero_conf),'same_date_train_test_nonzero_pairs':test_train,'nonzero_cross_file_pairs':nonzero_pairs,'candidate_exclusions_by_reason':dict(Counter(quarantine.values())),'byte_verified_matches':True,'sources_unchanged_by_stat':True,'limits':['exact observed vectors do not identify visits/sessions','exclusion candidates do not rewrite frozen WTT-Time manifests','component/date directories do not establish exact causal factor isolation']}

def verify():
    # Real memmap parity in fixed rows plus synthetic hash and collision-policy checks.
    p=BASE/CFG['datasets']['proteus_network']['path']/'SG.npz';a=mmap_x(p)
    with np.load(p,allow_pickle=False) as z: b=z['X']
    assert np.array_equal(a[[0,len(a)-1]],b[[0,len(b)-1]])
    assert np.array_equal(np.sign(np.array([-1.2,0,3.4])).astype('i1'),[-1,0,1])
    assert np.array([1,-2,0],dtype=np.float64).astype('<i4').tobytes()==np.array([1,-2,0],dtype=np.int64).astype('<i4').tobytes()
    return {'memmap_np_load_parity':True,'signed_noninteger_timestamp_direction':True,'int_float_hdf5_canonical_parity':True}

def main():
    assert json.loads((RUN/'config.json').read_text())['status']=='frozen'
    checks=verify();h=h5_audit();n=core_audit();report={'scope':'structural integrity only; no performance evaluation or new split','checks':checks,'shusterman':h,'core_npz':n,'elapsed_seconds':time.monotonic()-START}
    (OUT/'p0_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINISHED',json.dumps({'h5_rows':h['rows'],'core_rows':n['rows'],'core_files':n['files'],'seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
