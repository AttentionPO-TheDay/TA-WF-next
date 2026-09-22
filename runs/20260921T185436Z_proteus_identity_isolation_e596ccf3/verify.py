"""Artifact consistency checks; does not reopen datasets."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

run=Path(__file__).resolve().parent
a=run/'artifacts'
s=json.loads((a/'summary.json').read_text())
assert len(s['files']) == 23
checked=0
for p, counts in s['version'].items():
    actual=Counter()
    with (a/('version_'+p.split('/')[1]+'_identity.csv')).open() as f:
        for i,r in enumerate(csv.DictReader(f)):
            assert int(r['row'])==i
            v=r['anchor_versions'].split('|') if r['anchor_versions'] else []
            assert r['status']==('unique' if len(v)==1 else 'ambiguous' if v else 'unmatched')
            actual[r['status']]+=1
            actual['label_conflict']+=int(r['label_conflict'])
            if len(v)==1: actual['matched_'+v[0]]+=1
            checked+=1
    assert dict(actual)==counts
    assert sum(actual[k] for k in ['unique','ambiguous','unmatched'])==s['files'][p]['rows']
for filename in ['exact_overlap.json','sign5000_overlap.json']:
    for r in json.loads((a/filename).read_text()):
        assert 0<r['unique_shared']<=r['a_rows']<=s['files'][r['a']]['rows']
        assert r['unique_shared']<=r['b_rows']<=s['files'][r['b']]['rows']
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.iterdir()) if p.is_file() and p.name!='integrity.json'}
result=dict(errors=0,manifest_rows_checked=checked,files=23,checks='artifact consistency only, not independent payload rerun',sha256=hashes)
(a/'integrity.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
