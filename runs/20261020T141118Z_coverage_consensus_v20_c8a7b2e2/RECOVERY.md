# 存储恢复记录

本轮90个小头训练及fit_freeze已完成。终点评价在83个完整状态后因/home磁盘ENOSPC中断，尚无完整汇总。原日志保留，partial outputs复制到artifacts/interruption_01（含原scores/transfers与不完整临时文件）。

仅将本run的artifacts/checkpoints/lgs迁到/mnt/data2/rbf/TA-WF-next/run_storage/本runid，逐文件SHA256核对后清除原副本并建立原路径软链接；未移动或删除旧实验和共享数据。所有冻结文件相对路径及内容散列不变。

使用原冻结runner从score重放、summarize、audit，不训练新权重、不改票数或阈值；重放不是新增独立重复。resume.py为存储恢复调度，新增文件不替换原冻结实现。预算包含此前耗时，恢复后仍执行原审计。

全部评分与审计已完成（90个新头权重重拟合一致、186状态重放通过）。完成后制作描述图时发现原/home/rbf/TA-WF-local-backup/.venv已不在原路径；本轮未删除该目录。描述绘图另用/tmp隔离环境，仅处理完成的JSON指标，不参与拟合或改变训练环境/权重。
