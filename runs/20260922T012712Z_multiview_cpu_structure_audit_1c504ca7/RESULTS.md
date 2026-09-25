# 实验结果

## CPU 结构审计完成

2026-09-22。使用 `/home/rbf/TA-WF/.venv/bin/python` 的 NumPy 运行；系统 `python3` 没有 NumPy，因此未安装依赖。脚本只读取四个 NPZ 的 `X`，从未访问 `y`，未训练、未评分、未使用 GPU。

审计对象为 NetworkDrift/train 16146 行、valid 1795 行、JP 3125 行和 BehaviorDrift/subpage 20400 行，共 41466 行。四组数据前 5000 个观测均通过有限值和 padding 结构检查；内部零值为 0，因而当前“零为后缀 padding”的假设没有被这四组数据直接违反。

## 关键结果

- 原始 packet budget 的截断比例（train / valid / JP / subpage）：256 为 96.3% / 96.0% / 94.6% / 95.8%；512 为 81.5% / 82.8% / 77.9% / 79.8%；1000 为 56.1% / 54.9% / 51.1% / 50.5%；2500 为 16.4% / 17.0% / 15.1% / 15.8%。因此不能把 256 或 512 当作“多数样本的完整 trace”。
- run-token budget 也不是原始包 budget。以 256 个 run 为例，train/valid/JP/subpage 中仍有 64.9% / 64.2% / 57.1% / 61.0% 的样本被截断；中位数只覆盖约 0.84 / 0.84 / 0.92 / 0.88 的原始 packet。
- 第 128 个 run 的原始结束位置中位数约为 train 428、valid 428、JP 464、subpage 425；第 256 个 run 约为 995、969、1074、965。相同 run 槽位对应的原始位置分散明显，不能把 run 槽位当作稳定的绝对 packet 位置。
- 第 512 个 run 的原始结束位置中位数约为 1957、1949、2070、1896；JP 的长尾更长。这支持使用显式 run mask/有效长度，而不是只靠零填充让模型自行推断。
- 50/250 窗口的 partial 比例均约 94%–96%；最后窗口几乎总是 partial，窗口模型必须保留 observed_count 或 mask，不能把 partial window 当成完整窗口。

## 决定与限制

下一轮 CPU/GPU learned baseline 应同时记录 `packet_length`、`run_count`、`run_valid_mask` 和 `window_observed_count`；建议先固定原始观测预算 5000，再单独预注册 run-token 上限（例如 256 或 512），不要把两种预算混称。run 位置应使用序列位置编码或显式 mask，不应解释为固定原始包区间。

本实验没有回答哪种视角能提高识别率，也没有读取目标标签或证明抗漂移性；它只冻结了输入适配的重要边界。完整无损工件为 `artifacts/structure.json`，执行日志为 `logs/audit.stdout` 和 `logs/audit.stderr`。
