# 实验结果

## 训练前评价修订 v4（2026-09-15）

用户授权修正逐样本 padding 并补同层对照，本次无训练、无 GPU 计算、无未来数据读取。以下原执行报告保留为 v3 历史；其中深层 D 与 smoke v3 已被本节及当前 PLAN 替代。

- source reference 195/204 条短于 5000，观测长度中位数 1002，旧深层 RF 不适合多数短 trace。保持 v3 split，D 改为 DF 第一卷积块（RF22）及 VarCNNDirection 第一残差块（RF35）的区域匹配。
- mask 按每个样本的全部 RF 输入检查边界及零值，零作为未知位置保守排除。source train/reference/holdout/valid 未发现内部零或非有限值；全部样本在新规则下均有至少 4 个有效位置，全部 reference 类保留。
- 新增 E：同层、同位置原始特征均值的 1-NN。增加 D-E、覆盖率和回退诊断；有效位置不足时按 PLAN 回退到 C。
- 新证据：`artifacts/source_length_audit_v4.json`、`local_layer_spec_v4.json`、`smoke_test_v4.json`。旧局部定义、旧 smoke 和 `PLAN_before_eval_v4.md` 保留。
- v4 验证：9 tests passed、1 optional legacy parity test skipped；CPU 真实 source smoke 覆盖两模型 A/B/C/D/E，并核实 hook 提取不改变分类输出。mask 的 RF 边界、内部零、无有效区域、reference 类覆盖与回退均有合成测试。
- v3 split SHA-256：`0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142`。当前仍待 GPU 后续训练，没有机制性能结论。

## 当前边界

状态：在训练前安全停止。协议、最终 split、统一 A/B/C/D 接口、局部层规范、真实 source smoke 和续跑命令已完成；没有启动新训练，没有生成 checkpoint，也没有读取未来 X/y 做预测或评分。因此本报告没有 D-C 数值，不对局部机制作研究判断。

阻塞证据：2026-09-15 非沙箱只读检查显示 3 张 RTX 4090 均为 100% utilization，已有 7 个 compute process，占用 13,947 / 13,377 / 18,314 MiB。详见 `artifacts/gpu_blocker_20260915.json`。为避免争抢现有工作，没有启动两次 30-epoch 训练，也没有改成 CPU、缩减数据/epoch 或更换问题。

## 已完成证据

- 最终 split：`artifacts/splits_v3.json`。16,309 supervised-train、204 reference（每类 2）、2,040 source-holdout（每类 20），102 类均覆盖。
- 内容隔离：`artifacts/split_content_audit_v4.json`。三角色 admitted-input 内容各自唯一且交叉为 0；三者与 official valid 的内容交叉也均为 0。source-holdout 只是同时间对照，不是全新确认集。
- 审计修订：初始 v1 按行划分发现 144 个跨角色重复组；v2 去除 train 内重复后又发现与 official valid 的 train/reference/holdout 重叠为 152/1/15。两版均未训练、未评分并保留作失败证据；v3 在任何未来反馈之前修正。
- 局部规范：`artifacts/local_layer_spec_v3.json`。DF `[B,256,18]`、RF 1786/stride 256，保留位置 3..15；VarCNNDirection `[B,512,157]`、RF 1755/stride 32，保留 28..128。两者固定聚合为 4 个区域，排除理论 RF 触及 padding 的位置；不解释为独立资源。
- smoke：`artifacts/smoke_test_v3.json`。真实 source 小批量上，DF 与 VarCNNDirection 的 A/B/C/D、global/local shape、D reference-region 配对均执行成功。smoke 使用随机初始化，仅验证接线，不是性能结果。
- 实现：`src/ta_wf_next/screening.py` 与 `scripts/run_temporal_screening.py`。推理不接收 query label；逐网站与局部区分力诊断只在全部预测固定后使用标签。单次 encoder pass 共用于 A/B/C/D，输出存储和等价比较成本。
- checkpoint 核查：新工作区没有 checkpoint。旧 DF 普通/适应 checkpoint 使用完整 official train，会污染本轮 reference/holdout；未找到协议相符的 Var-CNN checkpoint，故均不复用。

## 验证命令

```text
PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python -m py_compile src/ta_wf_next/screening.py scripts/run_temporal_screening.py tests/test_screening.py
PYTHONPATH=src /home/rbf/TA-WF/.venv/bin/python -m unittest discover -s tests -v
/home/rbf/TA-WF/.venv/bin/python scripts/experiment.py check
/home/rbf/TA-WF/.venv/bin/python scripts/run_temporal_screening.py prepare
/home/rbf/TA-WF/.venv/bin/python scripts/run_temporal_screening.py audit-split
/home/rbf/TA-WF/.venv/bin/python scripts/run_temporal_screening.py smoke
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
git diff --check
```

测试结果：7 tests passed，1 个需要显式旧源码的可选迁移一致性测试 skipped；`experiment.py check` 通过。最终续跑顺序与命令见 `RUN_COMMANDS.md`。

## 未解决风险

- NPZ 只有 `X,y`，无 session/site 字符串 ID；内容哈希隔离不能证明采集 session 独立。
- official valid 自身有 9 个重复副本，会轻微改变选模样本权重；已固定记录，未据未来结果处理。
- 使用的是明确标注的方向单分支 `VarCNNDirection`，不是带真实时间第二分支的完整 Var-CNN；这是为了与 DF 同一方向信息权限，报告时必须保留该限定。
- 完整评测代码尚未在已训练 checkpoint 上运行；性能 artifact、D-C、future-source 差、逐网站分布与跨 backbone 共同错误模式均待 GPU 空闲后产生。
