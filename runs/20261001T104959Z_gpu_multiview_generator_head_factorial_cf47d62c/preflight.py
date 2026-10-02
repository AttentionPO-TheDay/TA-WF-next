"""Source-only GPU interface/timing checks; no optimizer, no valid scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import gc, json, time
import torch
from model import MultiViewModel, parameter_counts

R = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
cache = torch.load(R/'artifacts/prepared.pt', map_location='cpu', weights_only=False)
source = cache['source']
data_bytes = sum(cache[r][k].numel()*cache[r][k].element_size()
                 for r in ('source','valid') for k in ('timestamps','tam','labels'))
x, t, y = [source[k][:64].cuda() for k in ('timestamps','tam','labels')]
del cache, source
records = {}
reference = None
reference_masks = None
for mode in ('fixed','learned'):
    for head in ('mlp','transformer'):
        torch.manual_seed(21729)
        m = MultiViewModel(mode,head).cuda()
        shared = {k:v.cpu().clone() for k,v in m.state_dict().items() if not k.startswith('head.')}
        if reference is None:
            reference = shared
        else:
            assert all(torch.equal(v,reference[k]) for k,v in shared.items())
        with torch.inference_mode():
            tokens = m.generator(x,t)
            masks = {k:tokens[k].cpu() for k in ('packet_mask','time_mask')}
            assert tokens['packet'].shape == (64,100,100)
            assert tokens['time'].shape == (64,120,110)
            if reference_masks is None:
                reference_masks = masks
            else:
                assert all(torch.equal(v,reference_masks[k]) for k,v in masks.items())
        torch.cuda.reset_peak_memory_stats()
        # Warm up and time real source forward/backward without updating parameters.
        for _ in range(3):
            m.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(m(x,t),y).backward()
        torch.cuda.synchronize()
        start = time.monotonic()
        for _ in range(12):
            m.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(m(x,t),y)
            assert torch.isfinite(loss)
            loss.backward()
        torch.cuda.synchronize()
        seconds = (time.monotonic()-start)/12
        gradients = {k:None if p.grad is None else float(p.grad.norm())
                     for k,p in m.generator.named_parameters()}
        if mode == 'fixed':
            assert all(v is None for v in gradients.values())
        else:
            assert all(v is not None and v>0 for v in gradients.values())
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        m.eval()
        with torch.inference_mode():
            assert torch.isfinite(m(x,t)).all()
        records[mode+'_'+head] = {**parameter_counts(m),'step_seconds':seconds,
                               'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
                               'generator_gradient_norms':gradients}
        del m, shared, tokens, loss
        gc.collect()
        torch.cuda.empty_cache()
free,total = torch.cuda.mem_get_info()
estimated = 2*(max(v['peak_cuda_bytes'] for v in records.values())+data_bytes+2*1024**3)
assert estimated < .85*free, (estimated,free)
result = {'checks':'passed','source_rows':64,'optimizer_steps':0,'valid_scored':False,
          'future_access':False,'shared_initialization_equal':True,'generator_masks_equal':True,
          'records':records,'resident_cache_bytes_per_worker':data_bytes,
          'two_worker_memory_estimate_bytes':estimated,'free_cuda_bytes':free,'total_cuda_bytes':total}
(R/'artifacts/preflight_gpu.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
