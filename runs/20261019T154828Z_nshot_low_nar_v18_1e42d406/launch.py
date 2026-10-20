"""Single bounded CPU worker; no GPU or unregistered background jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
C=json.loads((RUN/'config.json').read_text())


def main():
    start=time.monotonic();env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    timeline=[]
    for stage in ['freeze','train','score','summarize','audit']:
        remaining=C['budget']['max_wall_seconds']-(time.monotonic()-start);assert remaining>0
        cmd=[sys.executable,str(RUN/'audit.py')] if stage=='audit' else [sys.executable,str(RUN/'runner.py'),stage]
        began=time.monotonic();print(json.dumps({'stage_start':stage,'elapsed':time.monotonic()-start}),flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,check=True,timeout=remaining)
        timeline.append({'stage':stage,'seconds':time.monotonic()-began,'finished_unix':time.time()})
        (RUN/'artifacts/stages.json').write_text(json.dumps(timeline,indent=2)+'\n')
    assert json.loads((RUN/'artifacts/audit.json').read_text())['passed']
    decision=json.loads((RUN/'artifacts/decision.json').read_text())
    p=RUN/'RESULTS.md';text=p.read_text().replace('全部预定支持集拟合与终点评价完成，待终端审计。','全部预定支持集拟合、终点评价和终端审计完成。')
    p.write_text(text+'\n终端审计通过：864模型文件、54个OOF缓存、108个选择器及1410评价状态全部重载复算；标签/折隔离与原控制预测核验通过。\n')
    from runner import registry
    registry('completed',f"594新监督头/54支持状态/1410评价及独立审计完成；主判据={decision['passed']}，未部署，详见RESULTS.md。")
    print(json.dumps({'completed':True,'seconds':time.monotonic()-start,'primary_passed':decision['passed']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        (RUN/'artifacts/launch_failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
        from runner import registry
        registry('failed',f'执行失败：{type(exc).__name__}，保留证据，不追加调参。');raise
