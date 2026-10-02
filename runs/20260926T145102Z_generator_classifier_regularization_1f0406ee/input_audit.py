from common import *
from ta_wf_next.packet_patches import packet_patch_inputs

def audit():
    config=config_read(False);data=load_data(config)
    assert sha(ROOT/config['sampling_manifest'])==config['sampling_manifest_sha256']
    sampling=json.loads((ROOT/config['sampling_manifest']).read_text())['sampling']
    mapping=json.loads((ROOT/'configs/datasets.json').read_text())
    data_root=Path(mapping['data_root']);temporal=data_root/mapping['datasets']['proteus_temporal']['path']
    original_audit=json.loads((ROOT/config['prepared_input']).with_name('input_audit.json').read_text())
    hashes={};result={'roles':{},'raw_files_opened':False,'future_access':False,'gpu_used':False}
    for role,expected,count in [('source','train.npz',2040),('valid','valid.npz',510)]:
        entry=data[role];sample=sampling[role]
        assert (data_root/sample['path']).resolve()==(temporal/expected).resolve()
        assert entry['rows'].tolist()==sample['rows'] and len(entry['rows'])==count
        assert np.all(np.bincount(entry['labels'].numpy(),minlength=102)==(20 if role=='source' else 5))
        original=GeneratorTokenBatch(**entry['original'])
        packet_patch_inputs(original,entry['directions'],condition='ordered')
        digest=hashlib.sha256(entry['directions'].numpy().tobytes()).hexdigest()
        assert digest==original_audit['roles'][role]['selected_direction_sha256']
        assert hashlib.sha256(entry['labels'].numpy().tobytes()).hexdigest()==original_audit['roles'][role]['labels_sha256']
        hashes[role]={hashlib.sha256(row.tobytes()).hexdigest() for row in entry['directions'].numpy()}
        result['roles'][role]={'count':count,'directions_sha256':digest,'historical_arrays_and_rows_match':True}
    assert len(hashes['source'])==2040 and not hashes['source']&hashes['valid']
    result.update(source_valid_direction_overlap=0,prepared_input_sha256=config['prepared_input_sha256'])
    atomic_json(RUN/'artifacts/input_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":audit()
