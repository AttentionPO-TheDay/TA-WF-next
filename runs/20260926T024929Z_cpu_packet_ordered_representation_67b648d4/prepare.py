"""Read only the fixed historical source/valid selection; never open future files."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from ta_wf_next.hierarchical_transformer import HierarchicalViewTransformer
from ta_wf_next.packet_patches import packet_patch_inputs
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views, GeneratorTokenBatch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(batch):
    return {name: getattr(batch, name) for name in ('packet','packet_mask','runs','runs_mask','windows','windows_mask')} | {'spans': dict(batch.spans)}


def unpack(values):
    return GeneratorTokenBatch(**values)


def prepare():
    config = json.loads((RUN/'config.json').read_text())
    manifest_path = ROOT/config['sampling_manifest']
    assert sha(manifest_path) == config['sampling_manifest_sha256']
    sampling = json.loads(manifest_path.read_text())['sampling']
    paths = json.loads((ROOT/'configs/datasets.json').read_text())
    data_root = Path(paths['data_root'])
    temporal_root = data_root / paths['datasets']['proteus_temporal']['path']
    previous_path = ROOT/config['historical_prepared']
    assert sha(previous_path) == config['historical_prepared_sha256']
    prior = torch.load(previous_path, map_location='cpu', weights_only=False)
    data = {}; report = {'roles': {}, 'future_access': False, 'gpu_used': False}
    hashes = {}
    for role, filename, count in [('source','train.npz',2040),('valid','valid.npz',510)]:
        entry = sampling[role]
        path = data_root/entry['path']
        assert path.resolve() == (temporal_root/filename).resolve()
        assert len(entry['rows']) == count and len(set(entry['rows'])) == count
        with np.load(path, allow_pickle=False) as archive:
            x = archive['X'][entry['rows'], :5000].astype(np.float32)
            y = archive['y'][entry['rows']].astype(np.int64)
        assert x.shape == (count,5000) and np.isfinite(x).all()
        directions = np.sign(x).astype(np.int8)
        assert np.all(np.bincount(y,minlength=102) == (20 if role=='source' else 5))
        assert np.array_equal(y, prior[role]['labels'].numpy())
        # Independent regeneration catches stale prepared features / row mismatches.
        regenerated = batch_from_views([generate_views(row, input_kind='direction', budget=5000,
                                      window_sizes=(50,250)) for row in directions], packet_patch=50,max_runs=128)
        old = prior[role]['values']
        for name in ('packet','packet_mask','runs','runs_mask','windows','windows_mask'):
            assert torch.equal(getattr(regenerated,name), old[name]), (role,name)
        for name in ('packet','runs','windows'):
            assert torch.equal(regenerated.spans[name],old['spans_'+name])
        direction_tensor=torch.from_numpy(directions)
        for condition in config['conditions']:
            candidate = packet_patch_inputs(regenerated,direction_tensor,condition=condition)
            assert candidate.packet.shape == (count,100,102)
        data[role] = {'original':pack(regenerated),'directions':direction_tensor,
                      'labels':torch.from_numpy(y),'rows':torch.tensor(entry['rows'])}
        hashes[role] = {hashlib.sha256(row.tobytes()).hexdigest() for row in directions}
        report['roles'][role] = {'count':count,'raw_path':str(path),'selected_direction_sha256':hashlib.sha256(directions.tobytes()).hexdigest(),
                                 'labels_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
                                 'regenerated_original_matches_history':True}
        print('prepared',role,count,flush=True)
    assert len(hashes['source'])==2040 and not hashes['source'] & hashes['valid']
    report['source_valid_direction_overlap']=0
    torch.save(data,RUN/'artifacts/prepared.pt')
    report['prepared_sha256']=sha(RUN/'artifacts/prepared.pt')
    (RUN/'artifacts/input_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def benchmark():
    """Synthetic-only timing, no optimizer updates, labels or validation scores."""
    torch.manual_seed(919)
    directions=(torch.randint(0,2,(64,5000))*2-1).float()
    views=[generate_views(row.tolist(),input_kind='direction',budget=5000,window_sizes=(50,250)) for row in directions]
    original=batch_from_views(views,packet_patch=50,max_runs=128)
    output=[]
    for threads in (1,2,4):
        torch.set_num_threads(threads)
        for condition in ('summary','ordered'):
            batch=packet_patch_inputs(original,directions,condition=condition)
            model=HierarchicalViewTransformer(packet_dim=102)
            durations=[]
            for step in range(7):
                model.zero_grad(set_to_none=True)
                started=time.perf_counter()
                logits=model(batch)
                logits.square().mean().backward()
                if step>=2: durations.append(time.perf_counter()-started)
            row={'threads':threads,'condition':condition,'median_train_step_seconds':float(np.median(durations)),
                 'parameters':sum(p.numel() for p in model.parameters())}
            output.append(row); print(row,flush=True)
    (RUN/'artifacts/benchmark.json').write_text(json.dumps({'synthetic_only':True,'no_optimizer_steps':True,'rows':output},indent=2)+'\n')

if __name__=='__main__':
    if sys.argv[1:] == ['benchmark']: benchmark()
    elif sys.argv[1:] == ['prepare']: prepare()
    else: raise SystemExit('use benchmark or prepare')
