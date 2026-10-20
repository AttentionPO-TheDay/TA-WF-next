"""One bounded sequential worker; never kills another process or task."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
C=json.loads((RUN/'config.json').read_text())


def main():
    started=time.monotonic()
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(C['budget']['feature_gpu']),OMP_NUM_THREADS=str(C['budget']['threads']),
               MKL_NUM_THREADS=str(C['budget']['threads']),OPENBLAS_NUM_THREADS=str(C['budget']['threads']),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    stages=[]
    for stage in ['freeze','features','train','score','summarize','audit']:
        remaining=C['budget']['max_wall_seconds']-(time.monotonic()-started)
        assert remaining>0
        begun=time.monotonic()
        cmd=[sys.executable,str(RUN/'audit.py')] if stage=='audit' else [sys.executable,str(RUN/'runner.py'),stage]
        print(json.dumps({'stage_start':stage,'elapsed':time.monotonic()-started}),flush=True)
        subprocess.run(cmd,cwd=ROOT,env=env,check=True,timeout=remaining)
        stages.append({'stage':stage,'seconds':time.monotonic()-begun,'finished_unix':time.time()})
        (RUN/'artifacts/stages.json').write_text(json.dumps(stages,indent=2)+'\n')
    assert json.loads((RUN/'artifacts/audit.json').read_text())['passed']
    decision=json.loads((RUN/'artifacts/decision.json').read_text())
    with (RUN/'RESULTS.md').open('a') as f:
        f.write('\n终端审核通过：162拟合/174状态/18支持清单完整重载，15特征状态各32行重提取、独立指标与前沿核验通过。详见artifacts/audit.json。\n')
    from runner import registry
    registry('completed',f"严格1shot与cap3/5共162拟合、174状态及独立审计完成；主判据={decision['passed']}，详见RESULTS.md。未部署/扩片。")
    print(json.dumps({'completed':True,'seconds':time.monotonic()-started,'primary_passed':decision['passed']}),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        (RUN/'artifacts/launch_failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n')
        from runner import registry
        registry('failed',f'执行失败：{type(exc).__name__}，保留部分证据，未追加试验。')
        raise
