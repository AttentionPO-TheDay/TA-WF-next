# 20261013T132218Z_exp2_ema_anchor_4ceff5c9

实现版本v2；冻结后不按query表现改超参数。

问题：ema_anchor能否在相同数据与800次更新预算下超过完全去伪标签的低学习率基线，同时负适应率低于2%？

## 数据和权限
仅TemporalDrift已观察开发数据，3 seed=21729/23407/22026，Day14/90/270。源train标签仅用于CE，valid只用于既有checkpoint选模及排除重叠；未来support 1024条仅输入/行号，无标签；query输入和标签不进入适应，只在固定800步结束后预测评分。复用MVI精确输入去重与source/valid重叠排除后的split，不重新随机划分。WTT/AWF/其他因素关闭。输入为B×5000有符号时间戳和由其重建的B×2×1800 TAM。

## 配对与实现
source CE + target entropy/diversity + feature MMD，batch128/128，同Adam、loader seeds 2025/2026、每步dropout seed 2024+100000+step。BN momentum固定0.10。无GMM、无硬伪标签及对应第三路前向。

{
  "decay": 0.99,
  "weight": 0.5,
  "loss": "KL(teacher.softmax || student.softmax); soft distributions, no argmax or confidence labels",
  "teacher_initialization": "copy source model",
  "parameters_and_float_buffers": "EMA after optimizer step",
  "integer_buffers": "copy after step",
  "inference": "student only",
  "lr": "fixed 0.00025"
}

## 对照和裁决
历史frozen/no_pseudo_fixed/dynamic_lr_only各9条原始预测作为配对参考，不重复训练、不计独立重复。主要比较no_pseudo_fixed；Exp-4另比较dynamic_lr_only。终点固定第800步，不访问query进行选模。主筛查线F1超过73.504%且负适应<2%，另报告74.5%目标；完整报告3×3单元，不选择最好seed或日期。负适应率分母为静态正确样本，纠错率分母为静态错误样本。

## 成本与预算
每run9项新任务；GPU1单进程顺序队列，每任务2400秒、run21600秒，停止条件为NaN、协议违规、超时或CUDA失败。记录逐步lr/损失、执行更新数、forward/backward/optimizer/task/query秒、support-only batch1/128 p50/p95、显存；共享GPU成本只作描述性报告。预检使用support进行两步梯度检查，不接触query标签。hash冻结代码/配置/PLAN/数据prepared/checkpoint/历史split与预测。完成后独立指标复算及checkpoint重载验证。

Exp-4的窗口是适应过程窗口，support不是按采集时间排序，不能将它解释成真实在线漂移监测。此组是有界开发试验，不保证创新或目标达成。初版实现偏差及保留产物见repair_history.md。
