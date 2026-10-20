"""Storage-only recovery: replay original frozen scoring, no method changes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
sys.path.insert(0,str(RUN))
from runner import C,A,sha,verify,verify_fits,registry


def main():
    verify();verify_fits()
    initial=json.loads((A/'stages.json').read_text());assert [s['stage'] for s in initial]==['freeze','train']
    began=initial[0]['finished_unix']-initial[0]['seconds']
    save_start={'resume_sha256':sha(__file__),'start_unix':time.time(),'initial_elapsed_seconds':time.time()-began,'stages':[]}
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    registry('running','/home空间耗尽中断后，本轮输出已SHA校验迁至数据盘；90头不重训，原冻结代码重放评分与独立审计。')
    for stage in ['score','summarize','audit']:
        remain=C['budget']['max_wall_seconds']-(time.time()-began);assert remain>0
        command=[sys.executable,str(RUN/'audit.py')] if stage=='audit' else [sys.executable,str(RUN/'runner.py'),stage]
        start=time.monotonic();print(json.dumps({'resume_stage':stage,'original_elapsed':time.time()-began}),flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True,timeout=remain)
        save_start['stages'].append({'stage':stage,'seconds':time.monotonic()-start,'finished_unix':time.time()})
        (A/'resume_stages.json').write_text(json.dumps(save_start,indent=2)+'\n')
    assert json.loads((A/'audit.json').read_text())['passed']
    p=RUN/'RESULTS.md';p.write_text(p.read_text().replace('全部预定拟合及终点评价完成，待审计。','全部预定拟合、终点评价和独立审计完成。')+'\n90个新头完整重拟合权重一致，54个旧折模型重放；186状态独立近邻/投票/指标/前沿/转移、汇总与主判决复算通过。\n\n评分曾因/home磁盘ENOSPC中断；本轮输出SHA256核验迁盘后以原冻结代码重放，未调整方法或训练权重。恢复详情见[RECOVERY.md](RECOVERY.md)。\n')
    decision=json.loads((A/'decision.json').read_text())
    registry('completed',f"90新头/186状态及独立审计完成，主判据={decision['passed']}；迁盘恢复未改方法，未部署。")
    print(json.dumps({'completed':True,'original_elapsed_seconds':time.time()-began,'primary_passed':decision['passed']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        (A/'resume_failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},indent=2)+'\n')
        registry('failed',f'恢复执行失败：{type(e).__name__}，证据保留。');raise
