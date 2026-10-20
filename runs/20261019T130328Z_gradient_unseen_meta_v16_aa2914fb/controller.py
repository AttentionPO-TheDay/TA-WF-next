from runtime import *
import traceback

def main():
 start=time.monotonic();remaining=lambda:C['budget']['wall_seconds']-(time.monotonic()-start)
 def command(name,script,args):
  assert remaining()>0
  with (RUN/'logs'/f'{name}.log').open('w') as log:p=subprocess.run([os.sys.executable,str(RUN/script)]+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining())
  if p.returncode:raise RuntimeError(f'{name} failed; logs/{name}.log')
 def parallel(stage):
  processes=[];handles=[]
  try:
   for seed in C['seeds']:
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(C['budget']['seed_gpu_map'][str(seed)]),SOURCE_SEED=str(seed));log=(RUN/'logs'/f'{stage}_{seed}.log').open('w');handles.append(log);processes.append(subprocess.Popen([os.sys.executable,str(RUN/'worker.py'),stage],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
   for p in processes:
    code=p.wait(timeout=min(remaining(),C['budget']['worker_seconds']))
    if code:raise RuntimeError(f'worker {stage} failed; see per-seed log')
  finally:
   for p in processes:
    if p.poll() is None:p.terminate()
   for h in handles:h.close()
 with (ROOT/'.gradient_unseen_meta_v16.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);status('running','冻结梯度未见元监督检验；612/510/612源角色隔离，固定骨干/候选，三GPU仅前向。')
  command('freeze','experiment.py',['freeze']);command('prepare','experiment.py',['prepare']);parallel('preflight');parallel('compute');parallel('audit');assert 3*(time.monotonic()-start)/3600<=C['budget']['gpu_hours']
  status('running','15缓存及首256行重提取通过；开始等标签预算IM/GU重排拟合。');command('fit','experiment.py',['fit']);command('evaluate','experiment.py',['evaluate']);command('reload_audit','experiment.py',['audit']);command('report','report.py',[]);verify()
  assert sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())<C['budget']['output_gib']*1024**3
  dec=json.loads((RUN/'artifacts/registered_decisions.json').read_text());a=json.loads((RUN/'artifacts/reload_audit.json').read_text());assert a['passed'];paths=[p for p in RUN.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_audit.json'];save('delivery_audit.json',dict(passed=True,seconds=time.monotonic()-start,states=a['metric_states'],mechanism_passed=dec['mechanism_passed'],promotion_passed=dec['promotion_passed'],hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}))
  status('completed',f'IM/GU检验完成；{a["metric_states"]}源诊断/query状态重载通过。机理门槛={dec["mechanism_passed"]}，方法升级={dec["promotion_passed"]}；gradient-unseen非严格模型外确认，不部署。')

if __name__=='__main__':
 try:main()
 except BaseException as e:save('controller_failure.json',dict(error=str(e),traceback=traceback.format_exc()));status('failed',str(e));raise
