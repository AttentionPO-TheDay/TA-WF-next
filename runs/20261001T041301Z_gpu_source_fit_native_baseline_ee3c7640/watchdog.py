from pathlib import Path
import subprocess,sys,json,os,signal
r=Path(__file__).resolve().parent;root=r.parents[1]
c=json.loads((r/'config.json').read_text());assert c['status']=='frozen'
with (r/'logs/pipeline.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-u',str(r/'runner.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (r/'artifacts/process.json').write_text(json.dumps({'watchdog_pid':os.getpid(),'worker_pid':p.pid,'gpu':0,'timeout_seconds':c['pipeline_seconds']+60},indent=2)+'\n')
 try:p.wait(timeout=c['pipeline_seconds']+60)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  with (r/'RESULTS.md').open('a') as f:f.write('\n外层watchdog达到整批预算，停止；已完成产物保留。\n')
  subprocess.run([sys.executable,str(root/'scripts/experiment.py'),'status','--id',r.name,'--state','stopped','--summary','整批预算上限停止；保留已完成任务'],check=True)
  s=root/'STATUS.md';s.write_text(s.read_text().replace('已启动，GPU0串行小集拟合×3及DF/Transformer各3seed；未来评分关闭。','整批预算上限停止；保留已完成任务，未来评分关闭。'))
 sys.exit(p.returncode)
