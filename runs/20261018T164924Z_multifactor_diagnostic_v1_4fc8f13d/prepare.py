"""Label-independent selection, effective-input isolation and explicit diagnostic manifests."""
import csv
import hashlib
import json
from collections import defaultdict,Counter
from pathlib import Path
import time
from urllib.parse import urlsplit
import numpy as np
import torch
from runtime import RUN,ROOT,C,sha,save,source

def tam(x):
    out=np.empty((len(x),2,1800),np.float32)
    for i,r in enumerate(x):
        nz=r[r!=0];bins=np.minimum(np.floor(np.abs(nz).astype(np.float64)*1799/80),1799).astype(np.int64)
        out[i]=np.bincount(bins+1800*(nz<0),minlength=3600).reshape(2,1800)
    return out

def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def order(key):return hashlib.sha256(('diagnostic-v1|'+str(key)).encode()).hexdigest()

def main():
    started=time.monotonic()
    ds=json.loads((ROOT/C['dataset_config']).read_text()); data=Path(ds['data_root'])
    gate=ROOT/C['identity_run']/'artifacts'
    old=json.loads((gate/'audit.json').read_text());assert old['passed'] and sha(gate/'factor_rows.csv')==old['manifest_sha256']
    evidence=list(csv.DictReader((ROOT/C['label_evidence_run']/'artifacts/cross_family_label_evidence.csv').open()))
    for f in ('VersionDrift','NetworkDrift','BehaviorDrift'):
        e=[r for r in evidence if r['family_a']=='TemporalDrift' and r['family_b']==f]
        assert len(e)==102 and {int(r['label_a']) for r in e}==set(range(102))
        assert all(r['label_a']==r['label_b'] and int(r['shared_unique_raw_inputs'])>0 for r in e)
    records=defaultdict(list); seen_raw=set(); unknown=set()
    with (gate/'factor_rows.csv').open() as f:
        for r in csv.DictReader(f):
            g=int(r['raw_group_id']);assert r['in_pure_source']=='0'
            if r['family']=='VersionDrift':
                if r['version_status']!='unique':unknown.add(g);continue
                if g in seen_raw:continue
                seen_raw.add(g);name='version_'+r['attributed_version']
            elif r['family']=='NetworkDrift':name='network_'+r['slice']
            else:name='behavior_subpage'
            records[name].append({'file':r['file'],'row':int(r['row']),'group':g,'label':int(r['label'])})
    assert len(unknown)==10017 and len(seen_raw)==90145
    pure=torch.load(ROOT/C['source_run']/'artifacts/prepared.pt',map_location='cpu',weights_only=True)
    assert np.array_equal(tam(pure['source']['timestamps'][:128].numpy()),pure['source']['tam'][:128].numpy())
    protected={digest(r) for role in ('source','valid') for r in pure[role]['tam'].numpy()}
    used=set(protected); counts={}; manifest=[]; input_hashes={}
    # Read each physical NPZ once, extract only selected first5000; do not persist giant raw files.
    extracted={}; file_records=defaultdict(list)
    for name,rows in records.items():
        for r in rows:file_records[r['file']].append((name,r))
    index=json.loads((ROOT/C['p0_run']/'artifacts/core_file_index.json').read_text()); expected={r['path']:r for r in index}
    for file,pairs in sorted(file_records.items()):
        path=data/file;st=path.stat();e=expected[file]['source_stat']
        assert st.st_size==e['size'] and st.st_mtime_ns==e['mtime_ns'],file
        input_hashes[file]={'sha256':sha(path),'size':st.st_size,'mtime_ns':st.st_mtime_ns}
        with np.load(path,allow_pickle=False) as archive:
            x=archive['X'];y=archive['y'];urls=archive['url'] if file.startswith('BehaviorDrift/') else None
            for name,r in pairs:
                i=r['row'];assert y[i]==r['label']
                r['x']=np.array(x[i,:5000],dtype=np.float32,copy=True)
                r['url']=str(urls[i]) if urls is not None else ''
        del x,y
        print('loaded',file,len(pairs),flush=True)
    for name in C['slices']:
        rows=records[name];raw_n=len(rows)
        if name=='behavior_subpage':
            unique={}
            for r in sorted(rows,key=lambda r:(r['group'],r['row'])):unique.setdefault(r['url'],r)
            assert len(unique)==17739
            rows=list(unique.values())
        rows=sorted(rows,key=lambda r:(r['group'],r['file'],r['row']))
        x=np.stack([r['x'] for r in rows]);assert np.isfinite(x).all() and (x!=0).any(1).all()
        t=tam(x);keep=[];stats=Counter()
        for i,r in enumerate(rows):
            h=digest(t[i]);r['tam_hash']=h
            if h in used:stats['protected_or_target_effective_duplicate']+=1;continue
            used.add(h);keep.append(i)
        rows=[rows[i] for i in keep];x=x[keep];t=t[keep]
        perm=sorted(range(len(rows)),key=lambda i:order(rows[i]['url'] if name=='behavior_subpage' else rows[i]['group']))
        assert len(perm)>1024+128,name
        s,q=np.array(perm[:1024]),np.array(perm[1024:])
        d={}
        for role,ix in [('reference',s),('query',q)]:
            d[role]={'timestamps':torch.from_numpy(x[ix]),'tam':torch.from_numpy(t[ix]),
                     'rows':torch.tensor([rows[i]['group'] for i in ix]),
                     'hosts':[urlsplit(rows[i]['url']).hostname.lower().rstrip('.').removeprefix('www.') if rows[i]['url'] else '' for i in ix]}
            if role=='query':d[role]['labels']=torch.tensor([rows[i]['label'] for i in ix])
            for i in ix:
                r=rows[i];manifest.append({k:r[k] for k in ('file','row','group','label','url','tam_hash')}|{'slice':name,'role':role,'hostname':urlsplit(r['url']).hostname or '' if r['url'] else ''})
        assert set(d['reference']['rows'].tolist()).isdisjoint(d['query']['rows'].tolist())
        assert all(h not in protected for h in (rows[i]['tam_hash'] for i in perm))
        path=RUN/'artifacts'/f'data_{name}.pt';torch.save(d,path)
        counts[name]={'raw_attributed_rows':raw_n,'after_url_representative':len(keep)+sum(stats.values()),'retained':len(rows),'reference':len(s),'query':len(q),'query_classes':len(set(d['query']['labels'].tolist())),'exclusions':dict(stats),'cache_sha256':sha(path)}
        print(name,counts[name],flush=True)
        for r in records[name]:r.pop('x',None)
    path=RUN/'artifacts/diagnostic_rows.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['slice','role','file','row','group','label','url','hostname','tam_hash']);w.writeheader();w.writerows(manifest)
    assert len({r['tam_hash'] for r in manifest})==len(manifest)
    save('data_audit.json',{'passed':True,'evaluation_role':'diagnostic','unknown_version_unique_excluded':len(unknown),'attributed_version_unique_before_filters':len(seen_raw),'slices':counts,'physical_inputs':input_hashes,'manifest_sha256':sha(path),'target_reference_labels_in_cache':False,'elapsed_seconds':time.monotonic()-started})
    save('diagnostic_gate.json',{'unlocked':True,'scope':'observed_diagnostic_only','official_site_map_verified':False,'content_supported_map_assumed':True,'source_and_selection_valid_effective_overlap':0,'reference_query_effective_overlap':0,'query_for_adaptation':False,'independent_confirmation':False})

if __name__=='__main__':main()
