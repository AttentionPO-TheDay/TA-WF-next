from worker import *
import subprocess
c=json.loads((RUN/'config.json').read_text())
subprocess.run([sys.executable,str(RUN/'test_model.py')],check=True)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
raw=torch.load(ROOT/c['prepared_input'],weights_only=False,map_location='cpu')
for r,n in [('source150',150),('valid',5)]:
 assert torch.equal(torch.bincount(raw[r]['labels']),torch.full((102,),n));assert set(raw[r]['directions'].unique().tolist())<={-1.,0.,1.}
hashrows=lambda x:[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in x]
s=hashrows(raw['source150']['directions']);v=hashrows(raw['valid']['directions']);assert not set(s)&set(v)
savej(RUN/'artifacts/input_audit.json',{'source_count':15300,'valid_count':510,'overlap':0,'source_rows':raw['source150']['rows'].tolist(),'valid_rows':raw['valid']['rows'].tolist(),'direction_hashes':{'source':s,'valid':v}})
report={}
for k in c['conditions']:
 torch.cuda.reset_peak_memory_stats();torch.manual_seed(171);m=new_model(k);x=raw['source150']['directions'][:64].float().cuda();y=raw['source150']['labels'][:64].cuda()
 loss=F.cross_entropy(m(x),y);loss.backward();assert torch.isfinite(loss);assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 m.eval()
 with torch.inference_mode():z=m(raw['source150']['directions'][:128].float().cuda());assert torch.isfinite(z).all()
 report[k]={'peak_bytes':torch.cuda.max_memory_allocated(),'parameters':sum(p.numel() for p in m.parameters())}
 del m,x,y,loss,z;torch.cuda.empty_cache()
free,total=torch.cuda.mem_get_info();estimate=2*(max(x['peak_bytes'] for x in report.values())+2*1024**3)
assert estimate<free*.85,(estimate,free)
savej(RUN/'artifacts/preflight.json',{'models':report,'free_bytes':free,'two_workers_conservative_estimate':estimate,'optimizer_steps':0,'valid_scored':False,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
print(report)
