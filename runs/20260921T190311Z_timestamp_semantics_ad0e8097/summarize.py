import hashlib
import json
from collections import Counter
from pathlib import Path

run=Path(__file__).resolve().parent
data=json.loads((run/'artifacts/time_audit.json').read_text())
total=Counter(); samples=[]
for v in data['files'].values():
    c=v['counts']; total.update(c); samples.extend(v['samples'])
    assert c['negative_edges']==c['negative_same_direction_edges']+c['negative_cross_direction_edges']
    assert c['edges']==c['same_direction_edges']+c['cross_direction_edges']
    assert c['negative_rows']<=c['rows']
assert len(data['files'])==17
repo=Path('/tmp/proteus_audit.HOKVvy/Adaptive-WF-Attack-4cdab4163bf3de7036a2498dac533c888b97664d')
sources={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [repo/'README.md',repo/'wflib_copy/WFlib/tools/data_processor.py',repo/'exp/dataset_process/gen_tam.py']}
summary=dict(counts=dict(total),row_negative_fraction=total['negative_rows']/total['rows'],
    edge_negative_fraction=total['negative_edges']/total['edges'],
    negative_at_most_100us_fraction=1-total['negative_gt_0.0001']/total['negative_edges'],
    negative_at_most_1ms_fraction=1-total['negative_gt_0.001']/total['negative_edges'],
    sampled_rows=len(samples),sorting_changes_direction_rows=sum(r['sign_positions_changed']>0 for r in samples),
    sorting_changes_run_count_rows=sum(r['run_count_before']!=r['run_count_after'] for r in samples),
    sample_negative_edges=sum(r['negative_edges'] for r in samples),
    sample_rounded_negative_edges={d:sum(r['rounding_negative_edges'][d] for r in samples) for d in ['3','4','5','6']},
    max_negative_seconds=max(v['max_negative_seconds'] for v in data['files'].values()),
    evidence_sources_sha256=sources,integrity_errors=0,
    caveat='view-weighted counts, not independent traces; artifact arithmetic checks only')
(run/'artifacts/summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
