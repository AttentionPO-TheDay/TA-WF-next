"""Synthetic ridge equivalence and read-only input/anchor structural checks."""
from diagnose import *
from ta_wf_next.packet_patches import packet_patch_inputs


def main():
    torch.set_num_threads(1)
    rng=np.random.default_rng(7);xs=rng.normal(size=(53,12));xv=rng.normal(size=(17,12));xs[:,2]=2;xv[:,2]=2
    y=np.arange(53)%7
    mu,scale,fits=ridge_fit(xs,xv,y,[1.,.1,10.])
    z=(xs-mu)/scale;v=(xv-mu)/scale;target=np.eye(102)[y];prior=target.mean(0)
    for lam,w,b,sp,vp in fits:
        direct=np.linalg.solve(z.T@z+len(y)*lam*np.eye(12),z.T@(target-prior))
        assert np.allclose(w,direct,atol=1e-12,rtol=1e-10)
        assert np.array_equal(vp,(v@direct+prior).argmax(1))
        assert scale[2]==1 and np.array_equal(b,prior)
    c=json.loads((RUN/'config.json').read_text())
    for path,digest in c['anchor_sha256'].items():assert sha(ROOT/path)==digest
    assert sha(ROOT/c['prepared_input'])==c['prepared_input_sha256']
    data=torch.load(ROOT/c['prepared_input'],map_location='cpu',weights_only=False)
    sampling=json.loads((ROOT/c['sampling_manifest']).read_text())['sampling']
    mapping=json.loads((ROOT/'configs/datasets.json').read_text());root=Path(mapping['data_root']);temporal=root/mapping['datasets']['proteus_temporal']['path']
    hashes={}
    for role,count,per_class,filename in [('source',2040,20,'train.npz'),('valid',510,5,'valid.npz')]:
        entry=data[role];assert entry['rows'].tolist()==sampling[role]['rows']
        assert (root/sampling[role]['path']).resolve()==(temporal/filename).resolve()
        assert len(entry['directions'])==count and np.all(np.bincount(entry['labels'].numpy(),minlength=102)==per_class)
        packet_patch_inputs(GeneratorTokenBatch(**entry['original']),entry['directions'],condition='ordered')
        hashes[role]={hashlib.sha256(row.tobytes()).hexdigest() for row in entry['directions'].numpy()}
    assert not hashes['source']&hashes['valid'] and len(hashes['source'])==2040
    assert json.loads((ROOT/c['prior_run']/'artifacts/integrity.json').read_text())['complete']
    write_json(RUN/'artifacts/preflight.json',{'synthetic_primal_dual_equivalence':True,'constant_feature_handling':True,
               'source_valid_overlap':0,'rows_labels_paths_and_anchors_verified':True,'prior_experiment_integrity_complete':True,'performance_scored':False})
    print('preflight PASS')
if __name__=='__main__':main()
