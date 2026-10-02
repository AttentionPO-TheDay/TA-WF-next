from pathlib import Path
import json,hashlib,time,sys,importlib.util
import numpy as np
import torch
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def hashes(x):return [hashlib.sha256(a.tobytes()).hexdigest() for a in x]
c=json.loads((R/'config.json').read_text());assert c['status']=='frozen';start=time.monotonic();torch.set_num_threads(4)
for p,h in c['input_hashes'].items():assert sha(ROOT/p)==h,p
old=ROOT/c['prior_run'];prior=torch.load(ROOT/c['prepared_input'],map_location='cpu',weights_only=False);sr=prior['source150']['rows'].numpy();vr=prior['valid']['rows'].numpy();y=prior['valid']['labels'].numpy();d=prior['valid']['directions'].numpy();length=(d!=0).sum(1)
mapping=json.loads((ROOT/'configs/datasets.json').read_text());base=Path(mapping['data_root'])/mapping['datasets']['proteus_temporal']['path']
with np.load(base/'train.npz',allow_pickle=False) as f:
 raw=f['X'];full_length=np.count_nonzero(raw,axis=1);assert np.isfinite(raw).all();direction=np.sign(raw[:,:5000].astype(np.float32)).astype(np.int8);sy=f['y'].astype(np.int64);del raw
with np.load(base/'valid.npz',allow_pickle=False) as f:
 raw=f['X'];vl=np.count_nonzero(raw,axis=1)[vr];vd=np.sign(raw[:,:5000].astype(np.float32)).astype(np.int8);assert np.array_equal(f['y'][vr],y);del raw
assert np.array_equal(direction[sr],prior['source150']['directions'].numpy()) and np.array_equal(vd[vr],d)
hs=hashes(direction);hv=set(hashes(vd));groups={}
for i,h in enumerate(hs):groups.setdefault(h,[]).append(i)
conflict={h for h,ix in groups.items() if len(set(sy[ix]))>1};obs=direction!=0;struct=obs.any(1)&~(((~obs).cumsum(1)>0)&obs).any(1)
seen=set();eligible=[];oldset=set(sr.tolist());excluded={'overlap':0,'conflict':0,'structure':0,'duplicate':0}
for i in list(sr)+[i for i in range(len(sy)) if i not in oldset]:
 h=hs[i];reason='overlap' if h in hv else ('conflict' if h in conflict else ('structure' if not struct[i] else ('duplicate' if h in seen else None)))
 if reason:excluded[reason]+=1;assert i not in oldset;continue
 seen.add(h);eligible.append(int(i))
counts=np.bincount(sy[eligible],minlength=102);assert oldset<=set(eligible)
manifest={'source150_rows':sr.tolist(),'all_eligible_rows':eligible,'valid_rows':vr.tolist(),'per_class_counts':counts.tolist(),'excluded':excluded,'raw_hashes':{n:sha(base/n) for n in ['train.npz','valid.npz']}}
save(R/'artifacts/manifest.json',manifest)
torch.save({'source150':{k:prior['source150'][k] for k in ['directions','labels','rows']},'source_all':{'directions':torch.from_numpy(direction[eligible]),'labels':torch.from_numpy(sy[eligible]),'rows':torch.tensor(eligible)},'valid':{k:prior['valid'][k] for k in ['directions','labels','rows']}},R/'artifacts/prepared.pt')
# Load only explicitly identified current-project checkpoints for diagnostic inference.
spec=importlib.util.spec_from_file_location('audit_packet_model',old/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
results={};pred_by_kind={}
for kind in ['mlp','transformer']:
 preds=[];probs=[]
 for seed in c['seeds']:
  model=mod.PacketModel(head=kind);ck=torch.load(old/'checkpoints'/f'{kind}_{seed}_best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval();out=[]
  with torch.inference_mode():
   for i in range(0,len(y),32):out.append(model(torch.from_numpy(d[i:i+32]).float()).softmax(1).numpy())
  p=np.concatenate(out);original=np.load(old/'artifacts'/f'{kind}_{seed}'/'predictions_best.npz')['valid'];assert np.array_equal(p.argmax(1),original),'CPU diagnostic parity'
  np.save(R/'artifacts'/f'{kind}_{seed}_probabilities.npy',p);preds.append(original);probs.append(p)
 pred=np.stack(preds);err=pred!=y;allwrong=err.all(0);anywrong=err.any(0);conf=np.stack(probs).max(2);cm=np.zeros((102,102),int)
 for pr in pred:np.add.at(cm,(y,pr),1)
 np.fill_diagonal(cm,0);pairs=sorted([(int(cm[a,b]),int(a),int(b)) for a,b in zip(*np.nonzero(cm))],reverse=True)
 bins=[]
 for lo,hi in [(1,1000),(1001,2500),(2501,4999),(5000,5000)]:
  mask=(length>=lo)&(length<=hi);bins.append({'length_range':[lo,hi],'samples':int(mask.sum()),'mean_error_rate':float(err[:,mask].mean()) if mask.any() else None,'all_seed_wrong':int(allwrong[mask].sum())})
 results[kind]={'mean_accuracy':float((~err).mean()),'all_seed_wrong':int(allwrong.sum()),'any_seed_wrong':int(anywrong.sum()),'all_seed_correct':int((~anywrong).sum()),'pairwise_error_jaccard':[float((err[a]&err[b]).sum()/(err[a]|err[b]).sum()) for a,b in [(0,1),(0,2),(1,2)]],'wrong_confidence_ge_09':int(((conf>=.9)&err).sum()),'wrong_predictions':int(err.sum()),'length_bins':bins,'top20_confusions':[{'count_across_seeds':n,'true_class':a,'predicted_class':b} for n,a,b in pairs[:20]],'top20_confusion_error_fraction':float(sum(z[0] for z in pairs[:20])/err.sum()),'majority_vote_accuracy':float((np.apply_along_axis(lambda a:np.bincount(a,minlength=102).argmax(),0,pred)==y).mean()),'all_wrong_rows':vr[allwrong].tolist()};pred_by_kind[kind]=err
summary={'source_total':len(sy),'eligible_total':len(eligible),'additional_over150':len(eligible)-len(sr),'per_class_min':int(counts.min()),'per_class_median':float(np.median(counts)),'per_class_max':int(counts.max()),'balanced_max_per_class':int(counts.min()),'excluded':excluded,'source_stored_width':int(direction.shape[1]),'selected_valid_longer_than5000':int((vl>5000).sum()),'selected_valid_at5000':int((length==5000).sum()),'both_models_all_seeds_wrong':int((pred_by_kind['mlp'].all(0)&pred_by_kind['transformer'].all(0)).sum()),'models':results,'elapsed_seconds':time.monotonic()-start,'future_access':False}
save(R/'artifacts/summary.json',summary)
text=['# 源期错误与数据量审计','',f"可用训练样本{len(eligible)}，比150条/类增加{len(eligible)-len(sr)}；每类min/median/max={counts.min()}/{np.median(counts)}/{counts.max()}。保持类均衡最多每类{counts.min()}条，不可承诺扩到300/500条。",'',f"两个模型全部六seed都错的验证样本：{summary['both_models_all_seeds_wrong']}/510。仅描述已观察valid错误，不是独立确认。",'',json.dumps(summary,ensure_ascii=False,indent=2),'','已核验6份checkpoint在CPU上重算与历史GPU预测逐条一致。保存置信度仅用于本轮诊断，不用于适应或梯度。source/valid完全方向重复已排除，source150嵌套保留；允许数据用途未扩展至未来。']
(R/'RESULTS.md').write_text('\n'.join(text)+'\n');print(json.dumps(summary,ensure_ascii=False))
