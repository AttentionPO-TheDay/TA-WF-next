"""Pre-score identity, permissions and gradient-boundary checks; no real-data fitting."""
from common import *
from torch.nn import functional as F

def main():
 c=json.loads((RUN/'config.json').read_text());assert c['status']=='draft'
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for p,h in c['anchor_sha256'].items():assert sha(ROOT/p)==h,p
 assert sha(ROOT/c['prepared_input'])==c['prepared_input_sha256']
 assert sha(ROOT/c['sampling_manifest'])==c['sampling_manifest_sha256']
 datasets=json.loads((ROOT/'configs/datasets.json').read_text());assert (Path(datasets['data_root'])/datasets['datasets']['proteus_temporal']['path']).is_dir()
 for path in [ROOT/c['diagnostic_run']/'artifacts/integrity.json',ROOT/c['prior_run']/'artifacts/integrity.json']:
  integrity=json.loads(path.read_text());assert not integrity.get('errors')
 data=torch.load(ROOT/c['prepared_input'],map_location='cpu',weights_only=False);assert set(data)=={'source','valid'}
 for role,n,k in [('source',2040,20),('valid',510,5)]:
  e=data[role];assert len(e['labels'])==n and e['directions'].shape==(n,5000)
  assert torch.equal(torch.bincount(e['labels'],minlength=102),torch.full((102,),k))
 digests=lambda e:{hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in e['directions']}
 assert not digests(data['source'])&digests(data['valid'])
 init={}
 for seed in c['seeds']:
  a=fresh(seed+c['classifier_seed_offset']);b=fresh(seed+c['classifier_seed_offset'])
  assert state_hash(a)==state_hash(b);assert sum(p.numel() for p in a.parameters())==113514
  init[str(seed)]=state_hash(a)
 # Synthetic full-length masked input; online/cache logits and source-only updates.
 torch.manual_seed(73);old=GeneratorClassifier(pool='attention').eval();before=state_hash(old.generator)
 d=torch.randint(0,2,(2,5000))*2-1;d[1,4100:]=0
 base=GeneratorTokenBatch(torch.randn(2,100,2),torch.ones(2,100,dtype=torch.bool),torch.randn(2,128,4),torch.ones(2,128,dtype=torch.bool),torch.randn(2,120,4),torch.ones(2,120,dtype=torch.bool),{k:torch.zeros(2,n,2,dtype=torch.long) for k,n in [('packet',100),('runs',128),('windows',120)]})
 with torch.inference_mode():
  t=old.generator(d,d!=0,base.packet);online=old(base,d,d!=0);cached=old.classifier(replace(base,packet=t));assert torch.equal(online,cached)
 token=t.clone();assert not token.requires_grad and not token.is_inference()
 classifier=fresh(101729);initial=state_hash(classifier);optimizer=torch.optim.AdamW(classifier.parameters(),lr=.001)
 classifier.train();F.cross_entropy(classifier(replace(base,packet=token)),torch.tensor([0,1])).backward();optimizer.step()
 assert state_hash(classifier)!=initial and state_hash(old.generator)==before and all(p.grad is None for p in old.generator.parameters())
 write_json(RUN/'artifacts/preflight.json',{'complete':True,'paired_initial_sha256':init,'classifier_parameters':113514,'source_valid_overlap':0,'synthetic_cache_online_parity':True,'generator_gradient_isolation':True,'data_counts_verified':True,'no_real_training':True,'errors':[]})
 print('PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
