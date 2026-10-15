import json, hashlib, os, glob
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import h5py
ROOT=Path('/home/rbf/TA-WF-next'); DATA=Path('/mnt/data2/ren/datasets')
W=DATA/'processed/wtt_time/exp_26dd67e087824c2f'
AWF=DATA/'awf_concept_drift_200w'
run=ROOT/'runs/20261015T023855Z_external_transfer_audit_e405d268'

def digest(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

label_map=json.loads((W/'label_map.json').read_text())
site_to_id={x['site_name']:x['global_class_id'] for x in label_map['sites']}
split=json.loads((W/'split_manifest.json').read_text())
roles={phase:info['dates'] for phase,info in split['phases'].items()}
# Stream the canonical sample index to audit role counts and label/date consistency.
role_counts=Counter(); date_counts=Counter(); role_labels=defaultdict(set); source_files=set(); row_count=0; sample_ids=set(); duplicate_ids=0
with (W/'sample_index.jsonl').open() as f:
 for line in f:
  x=json.loads(line); row_count+=1
  role=x['protocol_phase'] + ':' + x['protocol_role']; role_counts[role]+=1; date_counts[(x['protocol_phase'],x['collection_date'])]+=1
  source_files.add(x['source_file'])
  assert x['site_name'] in site_to_id and int(x['global_class_id'])==site_to_id[x['site_name']]
  assert x['collection_date'] in sum(roles.values(),[])
  if x['sample_id'] in sample_ids: duplicate_ids+=1
  sample_ids.add(x['sample_id']); role_labels[role].add(int(x['global_class_id']))
# Check every main HDF5 container schema without loading vectors into memory.
h5_schema=[]
for condition_name in ('dynamic_browser_version_network','static_browser_version_network'):
 for date in sorted(set(sum(roles.values(),[]))):
  date_dir=date.replace('-','_')
  for split_name in ('train','test'):
   p=DATA/'shusterman_concept_drift_v2'/condition_name/date_dir/(split_name+'.h5')
   assert p.is_file(), p
   with h5py.File(p,'r') as h:
    assert set(h.keys())=={'site_name','start_date','tag','vector','version'}
    assert h['vector'].ndim==2 and h['vector'].shape[1]==3000 and str(h['vector'].dtype) in {'int32','float64'}
    n=h['vector'].shape[0]; assert len(h['site_name'])==n and len(h['tag'])==n
    h5_schema.append({'condition':condition_name,'date':date,'split':split_name,'rows':int(n),'shape':list(h['vector'].shape),'dtype':str(h['vector'].dtype)})
# AWF containers and label consistency.
awf=[]
for p in sorted(AWF.glob('*.npz')):
 z=np.load(p,allow_pickle=True); x=z['data']; y=z['labels']; vals,c=np.unique(y,return_counts=True)
 assert x.ndim==2 and x.shape[1]==5000 and x.dtype==np.int16 and len(y)==len(x)
 awf.append({'file':p.name,'rows':int(len(y)),'shape':list(x.shape),'dtype':str(x.dtype),'classes':int(len(vals)),'min_class_count':int(c.min()),'max_class_count':int(c.max()),'labels_sha256':hashlib.sha256(np.asarray(sorted(map(str,vals)),dtype='U').tobytes()).hexdigest()})
# Other-factor NPZ headers only: prior identity audit found shared content, so not eligible as confirmation training roles here.
other={}
for name in ('VersionDrift','NetworkDrift','BehaviorDrift'):
 files=sorted((DATA/name).rglob('*.npz')); rows=[]
 for p in files:
  z=np.load(p,allow_pickle=False); rows.append({'file':str(p.relative_to(DATA)),'keys':z.files,'shapes':{k:list(z[k].shape) for k in z.files},'dtypes':{k:str(z[k].dtype) for k in z.files}})
 other[name]={'files':len(files),'entries':rows}
result={'passed':True,'wtt':{'experiment_id':json.loads((W/'audit_report.json').read_text())['experiment_id'],'label_count':len(site_to_id),'row_count_index':row_count,'duplicate_sample_ids':duplicate_ids,'role_counts':dict(role_counts),'date_counts':{f'{a}|{b}':v for (a,b),v in date_counts.items()},'role_label_counts':{k:len(v) for k,v in role_labels.items()},'source_file_count':len(source_files),'h5_schema':h5_schema,'protocol_roles':roles,'input':'signed_packet_size_3000; direction-only adapter required'},'awf':awf,'other_factors':other,'data_mutation':False,'training':False,'tta':False,'query_scoring':False}
(run/'artifacts/audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
plan='''# 外部迁移数据审计\n\nWTT-Time 与 AWF 的结构、角色、输入模态和标签边界审计。该 run 不训练、不适应、不读取 query 标签做评分。\n\n## 结论\n\nWTT-Time 具备 100 类统一标签、15 个历史训练阶段、5 个开发阶段和 7 个 future 阶段；原始向量为 3000 长有符号包大小，必须使用独立方向序列前端，不能复用 TemporalDrift 时间戳语义。AWF 具备 200 类、5000 长有符号方向序列，训练容器与 5 个相对时间 query 容器分离。Version/Network/Behavior 的既有身份审计显示跨角色共享内容，暂不把它们作为未经新角色冻结的独立确认训练集。\n\n下一 run 将分别训练 WTT-Time/AWF 自有类别模型，固定方向前端和五个适应条件：frozen、tent_bn、tent_entropy_gate（tau=0.20/0.40）、Proteus-default、no_pseudo_fixed。\n'''
(run/'PLAN.md').write_text(plan)
(run/'RESULTS.md').write_text('# 外部迁移数据审计结果\n\n结构与角色审计完成；详情见 artifacts/audit.json。未训练、未适应、未进行性能评分。\n')
print(json.dumps({'wtt_rows':row_count,'wtt_h5_files':len(h5_schema),'awf_files':len(awf),'duplicate_sample_ids':duplicate_ids},ensure_ascii=False))
