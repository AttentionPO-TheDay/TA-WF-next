from worker import *
import subprocess,sys
for f in ['test_varcnn.py','test_keras_compat.py']:subprocess.run([sys.executable,str(R/f)],check=True)
c=json.loads((R/'config.json').read_text());assert c['status']=='draft';torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
raw=torch.load(R/'artifacts/native_prepared.pt',map_location='cpu',weights_only=False);reports={}
for kind in c['conditions']:
 cfg=c[kind];torch.cuda.reset_peak_memory_stats();torch.manual_seed(171);m=model(kind);feat='direction' if kind=='varcnn' else 'tam';x=raw['source'][feat][:cfg['batch_size']].cuda();y=raw['source']['labels'][:cfg['batch_size']].cuda()
 z=m(x);loss=F.cross_entropy(z,y);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 m.eval()
 with torch.inference_mode():
  z=m(x[:cfg['eval_batch_size']]);assert torch.isfinite(z).all()
 reports[kind]={'peak_bytes':torch.cuda.max_memory_allocated(),'parameters':sum(p.numel() for p in m.parameters()),'batch_size':cfg['batch_size'],'forward_backward':'passed'}
 del m,x,y,z,loss;torch.cuda.empty_cache()
free,total=torch.cuda.mem_get_info();estimate=sum(v['peak_bytes']+2*1024**3 for v in reports.values());assert estimate<free*.85,(estimate,free)
save(R/'artifacts/preflight.json',{'models':reports,'free_bytes':free,'estimated_parallel_with_reserve':estimate,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'training_steps':0,'valid_scored':False});print(reports)
