"""Explicit read-only audit. Standard NPY memmap, no donor code imports."""
import ast
import csv
import hashlib
import io
import json
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
DATA = Path(json.loads((ROOT/'configs/datasets.json').read_text())['data_root'])
OUT = RUN/'artifacts'

def mmap_x(path):
    with zipfile.ZipFile(path) as z:
        i = z.getinfo('X.npy')
        assert i.compress_type == 0
    with path.open('rb') as f:
        f.seek(i.header_offset)
        h = struct.unpack('<IHHHHHIIIHH', f.read(30))
        f.seek(h[-2]+h[-1], 1)
        assert f.read(6) == b'\x93NUMPY'
        major, minor = f.read(2)
        assert major in (1, 2)
        n = struct.unpack('<H' if major == 1 else '<I', f.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(f.read(n).decode('latin1').strip())
        offset = f.tell()
    assert header['descr'] == '<f8' and not header['fortran_order']
    assert header['shape'][1] == 10000
    return np.memmap(path, mode='r', dtype='<f8', offset=offset, shape=header['shape'])

def small(path, key):
    with zipfile.ZipFile(path) as z:
        return np.load(io.BytesIO(z.read(key+'.npy')), allow_pickle=False)

paths = [f'VersionDrift/{v}/{role}.npz' for v in ['045','046','047','048'] for role in ['train','valid','drift']]
paths += [f'NetworkDrift/{r}.npz' for r in ['train','valid','SG','JP','USA','DE','UK']]
paths += [f'BehaviorDrift/{r}.npz' for r in ['train','valid','test','subpage']]
arrays, labels, ids, sign_ids = {}, {}, {}, {}
representatives, sign_representatives = {}, {}
buckets, sign_buckets = defaultdict(list), defaultdict(list)
summaries = {}

def content_id(raw, buckets, reps, view):
    digest = hashlib.sha256(raw).digest()
    for cid in buckets[digest]:
        p, j = reps[cid]
        candidate = arrays[p][j].tobytes() if view == 'raw' else np.sign(arrays[p][j,:5000]).astype(np.int8).tobytes()
        if candidate == raw:
            return cid
    cid = len(reps)
    buckets[digest].append(cid)
    return cid

for p in paths:
    path = DATA/p
    x = arrays[p] = mmap_x(path)
    y = small(path, 'y')
    assert np.isfinite(y).all() and np.equal(y, y.astype(np.int64)).all() and len(y)==len(x)
    labels[p] = y.astype(np.int64)
    result, sr = [], []
    for j, row in enumerate(x):
        cid = content_id(row.tobytes(), buckets, representatives, 'raw')
        representatives.setdefault(cid, (p,j))
        result.append(cid)
        if not p.startswith('Version'):
            raw = np.sign(row[:5000]).astype(np.int8).tobytes()
            sid = content_id(raw, sign_buckets, sign_representatives, 'sign')
            sign_representatives.setdefault(sid, (p,j))
            sr.append(sid)
    ids[p] = result
    if sr:
        sign_ids[p] = sr
    sample = np.asarray(x[np.linspace(0,len(x)-1,min(32,len(x)),dtype=int)])
    zero_internal = sum(bool(np.any(row[np.flatnonzero(row==0)[0]:] != 0)) for row in sample if np.any(row==0))
    summaries[p] = dict(rows=len(x), classes=len(np.unique(y)), duplicate_copies=len(x)-len(set(result)),
        sign5000_duplicate_copies=len(x)-len(set(sr)) if sr else None,
        sample_only=dict(rows=len(sample), nonfinite=int(np.count_nonzero(~np.isfinite(sample))),
            internal_zero_rows=zero_internal,
            decreasing_abs_time_rows=sum(bool(np.any(np.diff(np.abs(r[r!=0]))<0)) for r in sample)))
    print(p, summaries[p]['rows'], 'duplicates', summaries[p]['duplicate_copies'], flush=True)

anchor_versions = defaultdict(set)
anchor_labels = defaultdict(set)
for p in paths:
    if p.startswith('Version') and not p.endswith('/drift.npz'):
        for cid,y in zip(ids[p],labels[p]):
            anchor_versions[cid].add(p.split('/')[1])
            anchor_labels[cid].add(int(y))
version = {}
for p in paths:
    if not p.startswith('Version') or not p.endswith('/drift.npz'):
        continue
    counts = Counter()
    with (OUT/('version_'+p.split('/')[1]+'_identity.csv')).open('x', newline='') as f:
        w=csv.writer(f); w.writerow(['row','label','content_id','anchor_versions','status','label_conflict'])
        for j,(cid,y) in enumerate(zip(ids[p],labels[p])):
            versions=sorted(anchor_versions[cid])
            status='unique' if len(versions)==1 else 'ambiguous' if versions else 'unmatched'
            conflict=bool(anchor_labels[cid] and anchor_labels[cid] != {int(y)})
            counts[status]+=1; counts['label_conflict']+=int(conflict)
            if len(versions)==1: counts['matched_'+versions[0]]+=1
            w.writerow([j,int(y),cid,'|'.join(versions),status,int(conflict)])
    version[p]=dict(counts)

def pair_table(mapping, name):
    rows=[]
    sets={p:set(v) for p,v in mapping.items()}
    for a in mapping:
        for b in mapping:
            if a>=b: continue
            overlap=sets[a]&sets[b]
            if not overlap: continue
            rows.append(dict(a=a,b=b,unique_shared=len(overlap),a_rows=sum(c in overlap for c in mapping[a]),b_rows=sum(c in overlap for c in mapping[b])))
    (OUT/name).write_text(json.dumps(rows,indent=2))
    return rows

pair_table(ids,'exact_overlap.json')
pair_table(sign_ids,'sign5000_overlap.json')
conflicts={}
for scope,mapping in [('raw',ids),('sign5000',sign_ids)]:
    # Version and Network/Behavior have no proven cross-family label mapping.
    for family in ['Version','NetworkBehavior']:
        groups=defaultdict(set)
        for p,values in mapping.items():
            if p.startswith('Version') != (family=='Version'): continue
            for cid,y in zip(values,labels[p]): groups[cid].add(int(y))
        conflicts[scope+'_'+family]=sum(len(v)>1 for v in groups.values())
urls=small(DATA/'BehaviorDrift/subpage.npz','url')
url_counts=Counter(urls.tolist()); url_labels=defaultdict(set)
for u,y in zip(urls,labels['BehaviorDrift/subpage.npz']): url_labels[str(u)].add(int(y))
url_summary=dict(rows=len(urls),unique=len(url_counts),duplicate_copies=len(urls)-len(url_counts),
    repeated_url_groups=sum(n>1 for n in url_counts.values()),max_multiplicity=max(url_counts.values()),
    cross_label_urls=sum(len(v)>1 for v in url_labels.values()),
    source_url_overlap='unverifiable: homepage files lack URL fields')
summary=dict(files=summaries,version=version,label_conflict_content_groups=conflicts,url=url_summary,
    version_anchor_unique_content=len([c for c,v in anchor_versions.items() if v]),
    version_anchor_multiversion_content=sum(len(v)>1 for v in anchor_versions.values()),
    limits=['exact bytes only, not session independence','sign check only first 5000 for Network/Behavior','quality checks sampled 32 rows/file','no source/target split created','version identity inferred from release views, not independent capture metadata'],
    access=dict(explicit_files=paths,labels_used_for='audit only',training=0,scoring=0))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print('COMPLETE',json.dumps(dict(version=version,url=url_summary,conflicts=conflicts)),flush=True)
