"""Bounded CPU calibration experiment, without new gradient training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];C=json.loads((RUN/'config.json').read_text())


def main():
    began=time.monotonic();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    timeline=[]
    for stage in ['freeze','calibrate','score','summarize','audit']:
        remaining=C['budget']['max_wall_seconds']-(time.monotonic()-began);assert remaining>0
        cmd=[sys.executable,str(RUN/'audit.py')] if stage=='audit' else [sys.executable,str(RUN/'runner.py'),stage]
        start=time.monotonic();print(json.dumps({'stage_start':stage,'elapsed':time.monotonic()-began}),flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,check=True,timeout=remaining)
        timeline.append({'stage':stage,'seconds':time.monotonic()-start,'finished_unix':time.time()})
        (RUN/'artifacts/stages.json').write_text(json.dumps(timeline,indent=2)+'\n')
    assert json.loads((RUN/'artifacts/audit.json').read_text())['passed']
    decision=json.loads((RUN/'artifacts/decision.json').read_text());p=RUN/'RESULTS.md'
    p.write_text(p.read_text().replace('全部预定校准和终点评价完成，待审计。','全部预定校准、终点评价和审计完成。')+'\n324个复用折模型、36OOF、648阈值和726评价状态已完整重载；独立投票/指标/前沿复算通过。\n')
    from runner import registry
    registry('completed',f"648校准器/726状态与独立审计完成；主判据={decision['passed']}，未部署，见RESULTS.md。")
    print(json.dumps({'completed':True,'seconds':time.monotonic()-began,'primary_passed':decision['passed']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        (RUN/'artifacts/launch_failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
        from runner import registry
        registry('failed',f'执行失败：{type(exc).__name__}，保留证据，不追加调参。');raise
