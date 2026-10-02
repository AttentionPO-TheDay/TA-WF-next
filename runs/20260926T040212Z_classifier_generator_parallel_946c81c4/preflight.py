"""Source/valid structural audit and synthetic-only throughput/restart checks."""
from __future__ import annotations
import argparse
import random
import runpy
import statistics
import subprocess
import time
from common import *
from ta_wf_next.packet_patches import packet_patch_inputs
from ta_wf_next.traffic_views import generate_views
from ta_wf_next.transformer_proto import batch_from_views


def synthetic():
    rng=torch.Generator().manual_seed(919)
    x=(torch.randint(0,2,(64,5000),generator=rng)*2-1).to(torch.int8)
    # Include partial/padded samples while keeping shapes fixed.
    x[::3,3701:]=0
    views=[generate_views(row.tolist(),input_kind='direction',budget=5000,window_sizes=(50,250)) for row in x]
    return Batch(batch_from_views(views,packet_patch=50,max_runs=128),x,x!=0)


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


def restart_check():
    config=config_read(False)
    test=runpy.run_path(str(ROOT/'tests/test_learnable_generator.py'))
    original,x,valid=test['example']()
    def step(model,optimizer):
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(original,x,valid),torch.tensor([0,1,2]))
        loss.backward();optimizer.step()
    torch.manual_seed(1729);np.random.seed(1729);random.seed(1729)
    model=test['make']();opt=make_optimizer(model,config,'B_local_attention_joint')
    step(model,opt)
    path=RUN/'artifacts/synthetic_restart.pt'
    atomic_torch(path,{'model':model.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),
                      'numpy_rng':np.random.get_state(),'python_rng':random.getstate()})
    step(model,opt);expected=state_hash(model);nr=np.random.rand();pr=random.random()
    restored=test['make']();ropt=make_optimizer(restored,config,'B_local_attention_joint')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    restored.load_state_dict(saved['model']);ropt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
    np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);step(restored,ropt)
    assert state_hash(restored)==expected and np.random.rand()==nr and random.random()==pr
    atomic_json(RUN/'artifacts/restart_check.json',{'exact_parameter_recovery':True,'optimizer_and_three_rngs_restored':True,'synthetic_only':True})


def bench_worker(name,tag,barrier):
    config=config_read(False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(919)
    batch=synthetic();model=make_model(config,name)
    def step():
        model.train();model.zero_grad(set_to_none=True);batch.logits(model).square().mean().backward()
    step();step()
    if barrier:
        (RUN/'artifacts'/f'ready_{tag}').write_text('ready')
        deadline=time.monotonic()+90
        while not (RUN/'artifacts/benchmark_go').exists():
            if time.monotonic()>deadline:raise TimeoutError('benchmark barrier')
            time.sleep(.05)
    started=time.time();times=[]
    for _ in range(6):
        tic=time.perf_counter();step();times.append(time.perf_counter()-tic)
    result={'condition':name,'tag':tag,'threads':2,'median_step_seconds':statistics.median(times),'times':times,
            'started_unix':started,'ended_unix':time.time(),'parameters':parameter_counts(model),'synthetic_only':True,'optimizer_steps':0}
    atomic_json(RUN/'artifacts'/f'benchmark_{tag}.json',result)
    print(json.dumps(result),flush=True)


def benchmark():
    config=config_read(False);names=[c['id'] for c in config['conditions']]
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for name in names:
        tag='single_'+name
        with (RUN/'logs'/f'benchmark_{tag}.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--name',name,'--tag',tag],env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    workers=[]
    assignments=names+names[:2]
    assert not (RUN/'artifacts/benchmark_go').exists(), 'do not rerun timing into existing artifacts'
    for i,name in enumerate(assignments):
        tag=f'parallel_{i}_{name}';log=(RUN/'logs'/f'benchmark_{tag}.log').open('w')
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--name',name,'--tag',tag,'--barrier'],env=environment,stdout=log,stderr=subprocess.STDOUT)
        workers.append((tag,p,log))
    deadline=time.monotonic()+90
    while not all((RUN/'artifacts'/f'ready_{tag}').exists() for tag,_,_ in workers):
        if time.monotonic()>deadline or any(p.poll() is not None for _,p,_ in workers):
            for _,p,log in workers:
                if p.poll() is None:p.terminate()
                log.close()
            raise RuntimeError('parallel benchmark failed to initialize')
        time.sleep(.1)
    (RUN/'artifacts/benchmark_go').write_text('go')
    for tag,p,log in workers:
        code=p.wait(timeout=90);log.close();assert code==0,(tag,code)
    rows=[json.loads(p.read_text()) for p in (RUN/'artifacts').glob('benchmark_single_*.json')]
    parallel=[json.loads((RUN/'artifacts'/f'benchmark_{tag}.json').read_text()) for tag,_,_ in workers]
    aggregate=36/(max(r['ended_unix'] for r in parallel)-min(r['started_unix'] for r in parallel))
    atomic_json(RUN/'artifacts/benchmark.json',{'single':rows,'six_workers':parallel,'aggregate_steps_per_second':aggregate,
                                             'synthetic_only':True,'optimizer_steps':0,'threads_total':12})
    print('synthetic parallel aggregate steps/s',aggregate,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','restart','benchmark','worker'])
    parser.add_argument('--name');parser.add_argument('--tag');parser.add_argument('--barrier',action='store_true');args=parser.parse_args()
    if args.mode=='audit':audit()
    elif args.mode=='restart':torch.set_num_threads(1);restart_check()
    elif args.mode=='benchmark':benchmark()
    else:bench_worker(args.name,args.tag,args.barrier)
