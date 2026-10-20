"""Frozen bounded CPU launch with stage timeouts and retained failure records."""
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
    stages=[]
    for stage in ['freeze','train','score','summarize','audit']:
        remaining=C['budget']['max_wall_seconds']-(time.monotonic()-began);assert remaining>0
        command=[sys.executable,str(RUN/'audit.py')] if stage=='audit' else [sys.executable,str(RUN/'runner.py'),stage]
        start=time.monotonic();print(json.dumps({'stage_start':stage,'elapsed':start-began}),flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True,timeout=remaining)
        stages.append({'stage':stage,'seconds':time.monotonic()-start,'finished_unix':time.time()})
        (RUN/'artifacts/stages.json').write_text(json.dumps(stages,indent=2)+'\n')
    assert json.loads((RUN/'artifacts/audit.json').read_text())['passed']
    p=RUN/'RESULTS.md';p.write_text(p.read_text().replace('全部预定拟合及终点评价完成，待审计。','全部预定拟合、终点评价和独立审计完成。')+'\n90个新头完整重拟合权重一致，54个旧折模型重放；186状态独立近邻/投票/指标/前沿/转移、汇总与主判决复算通过。\n')
    from runner import registry
    d=json.loads((RUN/'artifacts/decision.json').read_text())
    registry('completed',f"90新头/186状态完成并通过独立审计；主判据={d['passed']}，仅开发证据，未部署。")
    print(json.dumps({'completed':True,'seconds':time.monotonic()-began,'primary_passed':d['passed']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        (RUN/'artifacts/failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},indent=2)+'\n')
        from runner import registry
        registry('failed',f'执行失败：{type(e).__name__}，证据保留，不追加调参。');raise
