from runtime import *
import traceback

def main():
 start=time.monotonic()
 with (ROOT/'.tail_robust_reference_v15.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  status('running','配置已冻结，A/B/C八组因子对照与逐扰动留出检验；零GPU训练，开始源缓存核验。')
  for name,script,args in [('freeze','experiment.py',['freeze']),('fit','experiment.py',['fit']),('evaluate','experiment.py',['evaluate']),('audit','experiment.py',['audit']),('report','report.py',[])]:
   remaining=C['budget']['wall_seconds']-(time.monotonic()-start)
   if remaining<=0:raise TimeoutError('total budget')
   with (RUN/'logs'/f'{name}.log').open('w') as log:
    p=subprocess.run([os.sys.executable,str(RUN/script)]+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
   if p.returncode:raise RuntimeError(f'{name} exited {p.returncode}; see logs/{name}.log')
  assert sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())<C['budget']['output_gib']*1024**3
  verify();audit=json.loads((RUN/'artifacts/reload_audit.json').read_text());assert audit['passed'];dec=json.loads((RUN/'artifacts/registered_decisions.json').read_text());files=list(RUN.glob('*.py'))+[RUN/'config.json',RUN/'PLAN.md',RUN/'RESULTS.md']+list((RUN/'checkpoints').glob('*'))+list((RUN/'artifacts').glob('*'))+list(RUN.glob('figure_*'))
  save('delivery_audit.json',dict(passed=True,seconds=time.monotonic()-start,hashes={str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()},query_metric_states=audit['query_states'],primary_passed=dec['primary_passed'],integrated=False))
  status('completed',f'A/B/C八组完成，{audit["query_states"]}目标状态及120留出状态重载通过；ABC主判据={dec["primary_passed"]}。全部阴性保留，保留v12不部署。')

if __name__=='__main__':
 try:main()
 except BaseException as e:
  save('controller_failure.json',dict(error=str(e),traceback=traceback.format_exc()));status('failed',str(e));raise
