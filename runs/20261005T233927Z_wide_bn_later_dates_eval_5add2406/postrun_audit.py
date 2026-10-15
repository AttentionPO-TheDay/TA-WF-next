"""Independent sklearn metric and full checkpoint replay audit, no selection."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from evaluate import R,C,sha,save,verify,setup,load_model,predict
import json

def main():
    setup();n=verify();fr=json.loads((R/'artifacts/score_freeze.json').read_text())
    assert sha(R/'artifacts/data_manifest.json')==fr['manifest_sha256']
    for role,h in fr['prepared'].items():assert sha(R/f'artifacts/{role}_prepared.pt')==h
    reports=json.loads((R/'artifacts/scores.json').read_text());expected=len(C['seeds'])*len(C['roles']);assert len(reports)==expected
    assert {(q['seed'],q['role']) for q in reports}=={(seed,role) for seed in C['seeds'] for role in C['roles']}
    valid=torch.load(R/'artifacts/source_valid_prepared.pt',map_location='cpu',weights_only=False)['valid'];checks=[]
    for seed in C['seeds']:
        m,step=load_model(seed)
        for role in C['roles']:
            e=valid if role=='valid' else torch.load(R/f'artifacts/{role}_prepared.pt',map_location='cpu',weights_only=True)
            z=predict(m,e)
            with np.load(R/f'artifacts/predictions_{seed}_{role}.npz') as f:
                y=f['labels'];p=f['predictions'];assert np.array_equal(y,e['labels'].numpy());assert np.array_equal(f['rows'],e['rows'].numpy())
                assert np.array_equal(z.argmax(1),p);assert np.array_equal(f['logits'].argmax(1),p);assert np.allclose(z,f['logits'],rtol=1e-6,atol=1e-6)
            if role=='valid':
                with np.load(R/f'artifacts/reference/{seed}_logits_best.npz') as f:assert np.array_equal(p,f['valid'].argmax(1))
            record=next(q for q in reports if q['seed']==seed and q['role']==role)
            assert abs(record['accuracy']-accuracy_score(y,p))<1e-12
            assert abs(record['macro_f1']-f1_score(y,p,labels=np.arange(102),average='macro',zero_division=0))<1e-12
            np.save(R/f'artifacts/confusion_{seed}_{role}.npy',confusion_matrix(y,p,labels=np.arange(102)))
            checks.append({'seed':seed,'role':role,'rows':len(y),'prediction_replay_exact':True,'independent_metrics':True,'model_state_unchanged':True})
        del m
    save('artifacts/postrun_audit.json',{'passed':True,'freeze_files_verified':n,'metric_groups':expected,'checkpoint_replays':len(checks),'checks':checks})
    assert len(checks)==expected
    print(f'AUDIT PASSED: {expected} independent metrics and {len(checks)} full checkpoint replays',flush=True)
if __name__=='__main__':main()
