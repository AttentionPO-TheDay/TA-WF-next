import json, numpy as np, torch
import port_base as B
import runner as Q
R,ROOT,C,DEVICE=B.R,B.ROOT,B.C,B.DEVICE
Q.setup(); scores=json.loads((R/'artifacts/scores.json').read_text()); hist=[x for x in scores if x['origin']=='historical_reference']; new=[x for x in scores if x['origin']=='new']
assert len(scores)==81 and len(hist)==72 and len(new)==9 and {x['condition'] for x in new}=={'tent_shot_span'}
assert len(list((R/'checkpoints').glob('*.pt')))==9 and len(list((R/'artifacts').glob('update_*.json')))==9
for path,digest in json.loads((R/'artifacts/freeze.json').read_text()).items(): assert B.sha(ROOT/path)==digest,path
for row in scores:
    with np.load(R/'artifacts'/f"pred_{row['seed']}_{row['role']}_{row['condition']}.npz") as z:
        y=z['labels']; pred=z['predictions']; logits=z['logits']; rows=z['rows']
        assert len(y)==row['rows'] and logits.shape==(len(y),102) and np.array_equal(pred,logits.argmax(1)) and np.isfinite(logits).all()
        assert abs(B.metric(y,pred)['accuracy']-row['accuracy'])<1e-12 and abs(B.metric(y,pred)['macro_f1']-row['macro_f1'])<1e-12
        with np.load(R/'artifacts'/f"pred_{row['seed']}_{row['role']}_frozen.npz") as f:
            base=f['predictions']; assert np.array_equal(rows,f['rows']) and np.array_equal(y,f['labels'])
        correct=base==y; wrong=~correct
        expected={'negative_adaptation_rate':float((correct&(pred!=y)).sum()/max(1,correct.sum())),'recovery_rate':float((wrong&(pred==y)).sum()/max(1,wrong.sum())),'changed_rate':float((base!=pred).mean())}
        for k,v in expected.items(): assert abs(v-row[k])<1e-12
for row in new:
    assert row['candidate_updates']==800 and row['trainable_parameter_count']==32384 and row['batch_statistics'] is False
    assert row['bn_forward_batches']==1600 and row['bn_num_batches_delta']=={}
    gate=np.load(R/'artifacts'/f"gate_{row['seed']}_{row['role']}_{row['condition']}.npz")
    assert gate['selected'].shape==(800,128) and gate['agreement'].shape==(800,128) and gate['entropy_nats'].shape==(800,128)
    model,_=B.load_model(row['seed']); base={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; params,names=Q.configure_shot_span_model(model); assert sum(p.numel() for p in params)==32384
    assert not any('final_norm' in n or 'classifier' in n for n in names)
    ck=torch.load(R/'checkpoints'/f"{row['seed']}_{row['role']}_{row['condition']}.pt",map_location='cpu',weights_only=True); missing,unexpected=model.load_state_dict(ck['state_dict'],strict=True); assert not missing and not unexpected
    allowed=set(names)
    for key,val in ck['state_dict'].items():
        if key.endswith(('running_mean','running_var','num_batches_tracked')): assert torch.equal(val,base[key]),key
        elif key in model.state_dict() and key not in allowed: assert torch.equal(val,base[key]),key
    model.eval(); target=B.load_prepared(C['prepared'][row['role']]);
    with np.load(R/'artifacts'/f"split_{row['role']}.npz") as split: q=split['query_indices'][:128]
    rel=B.predict(model,target['timestamps'][q],target['tam'][q],128)
    with np.load(R/'artifacts'/f"pred_{row['seed']}_{row['role']}_{row['condition']}.npz") as p: assert np.allclose(rel,p['logits'][:128],atol=1e-5,rtol=1e-5)
    del model; torch.cuda.empty_cache()
B.save_json('audit.json',{'passed':True,'total_rows':81,'historical_rows':72,'new_tasks':9,'metrics_recomputed':81,'checkpoints':9,'checkpoint_reloads':9,'trainable_parameter_count':32384,'bn_buffers_unchanged':True,'query_statistics':False,'gate_masks_checked':9})
print('audit passed')
