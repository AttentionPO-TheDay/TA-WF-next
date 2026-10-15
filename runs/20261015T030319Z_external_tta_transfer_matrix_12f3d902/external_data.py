import hashlib, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import h5py

DATA=Path('/mnt/data2/ren/datasets')
W=DATA/'processed/wtt_time/exp_26dd67e087824c2f'
AWF=DATA/'awf_concept_drift_200w'

class H5Rows:
    def __init__(self, records):
        self.records=records
        self.files={}
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]; p=Path(r['path'])
        key=str(p)
        if key not in self.files: self.files[key]=h5py.File(p,'r')
        x=np.asarray(self.files[key]['vector'][r['row']],dtype=np.float32)
        if len(x)<5000: x=np.pad(x,(0,5000-len(x)))
        else: x=x[:5000]
        return x.astype(np.float32,copy=False), int(r['label']), int(r['uid'])
    def close(self):
        for f in self.files.values(): f.close()
        self.files={}

class ArrayRows:
    def __init__(self,x,y,uids): self.x=x; self.y=y; self.uids=uids
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.x[i].astype(np.float32,copy=False), int(self.y[i]), int(self.uids[i])


def _stable(records):
    return sorted(records,key=lambda r:hashlib.sha256(str(r['uid']).encode()).hexdigest())


def load_wtt_index():
    label_map=json.loads((W/'label_map.json').read_text()); site_to_id={x['site_name']:x['global_class_id'] for x in label_map['sites']}
    out=[]; uid=0
    with (W/'sample_index.jsonl').open() as f:
        for line in f:
            x=json.loads(line)
            if x['condition_name'] not in ('dynamic_browser_version_network','static_browser_version_network'): continue
            rel=x['source_file'].split('shusterman_concept_drift_v2/',1)[1]
            out.append({'path':str(DATA/'shusterman_concept_drift_v2'/rel),'row':int(x['row_index']),'label':int(site_to_id[x['site_name']]),'date':x['collection_date'],'condition':x['condition_name'],'role':x['protocol_role'],'phase':x['protocol_phase'],'uid':uid,'sample_id':x['sample_id']}); uid+=1
    return out, len(site_to_id)


def _balanced(records, per_class):
    groups=defaultdict(list)
    for r in records: groups[r['label']].append(r)
    assert groups and min(map(len,groups.values()))>=per_class
    out=[]
    for c in sorted(groups): out.extend(sorted(groups[c],key=lambda r:(r['date'],r['condition'],r['row']))[:per_class])
    return out


def wtt_dataset():
    records,nclasses=load_wtt_index()
    source=_balanced([r for r in records if r['role']=='historical_train'],100)
    by_date={}
    dates=sorted({r['date'] for r in records if r['role'] in ('development_reference_candidate','development_query','future_reference_candidate','future_query')})
    for date in dates:
        ref=_stable([r for r in records if r['date']==date and r['role'] in ('development_reference_candidate','future_reference_candidate')])[:1024]
        query=_stable([r for r in records if r['date']==date and r['role'] in ('development_query','future_query')])
        assert len(ref)==1024 and query
        by_date[date]={'reference':H5Rows(ref),'query':H5Rows(query),'phase':'development' if date<'2022-01-16' else 'future','query_rows':len(query)}
    return H5Rows(source),nclasses,by_date


def load_awf_file(name, mapping=None):
    z=np.load(AWF/name,allow_pickle=True); x=z['data']; labels=z['labels'].astype(str)
    classes=sorted(set(labels)) if mapping is None else sorted(mapping)
    mapping={c:i for i,c in enumerate(classes)} if mapping is None else mapping
    y=np.array([mapping.get(a,-1) for a in labels],dtype=np.int64)
    u=np.arange(len(y),dtype=np.int64); return x,y,u,len(mapping),mapping


def awf_dataset():
    names=['tor_time_test10d_200w_100tr.npz','tor_time_test2w_200w_100tr.npz','tor_time_test3d_200w_100tr.npz','tor_time_test4w_200w_100tr.npz','tor_time_test6w_200w_100tr.npz']
    src_labels=set(np.load(AWF/'tor_200w_2500tr.npz',allow_pickle=True)['labels'].astype(str).tolist())
    common=set(src_labels)
    for name in names: common &= set(np.load(AWF/name,allow_pickle=True)['labels'].astype(str).tolist())
    assert len(common)>=180
    mapping={c:i for i,c in enumerate(sorted(common))}; nclasses=len(mapping)
    sx,sy,su,_,_=load_awf_file('tor_200w_2500tr.npz',mapping)
    keep=np.array([v in common for v in np.load(AWF/'tor_200w_2500tr.npz',allow_pickle=True)['labels'].astype(str)])
    source_idx=np.concatenate([np.flatnonzero(keep & (sy==c))[:100] for c in range(nclasses)])
    source=ArrayRows(sx[source_idx],sy[source_idx],su[source_idx])
    dates={}
    for name in names:
        tx,ty,tu,_,_=load_awf_file(name,mapping); raw_labels=np.load(AWF/name,allow_pickle=True)['labels'].astype(str); keepq=np.array([v in common for v in raw_labels]); valid=np.flatnonzero(keepq); order=valid[np.argsort([hashlib.sha256(f'{name}|{int(i)}'.encode()).hexdigest() for i in valid])]
        support=order[:1024]; query=order[1024:]
        dates[name]={'reference':ArrayRows(tx[support],ty[support],tu[support]),'query':ArrayRows(tx[query],ty[query],tu[query]),'phase':'diagnostic','query_rows':len(query)}
    return source,nclasses,dates
