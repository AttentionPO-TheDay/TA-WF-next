# 实验结果

状态：completed。冻结复用本项目 `20260923T082606Z_gpu_feature_distill_b4f0ceb2` 中已选的三 seed 融合教师与普通 CE packet 学生；未继续训练骨干或重选 checkpoint。只访问固定 TemporalDrift source 2040/valid 510，source 拟合 alpha=10 闭式 ridge 探针和重建器，valid 仅评分；无未来日期、WTT-Time/AWF 访问。冻结特征 GPU 提取及 CPU 拟合合计约 7.7 秒。

## 判别力与可转移性

下表为 valid macro-F1（%，三 seed 均值）。类别探针仅用 source 标签拟合；“重建后”使用 CE 学生 packet 表示经 source 拟合的线性重建器，再送入同一个教师特征类别探针。

| 特征 | 真实教师特征探针 | 学生重建特征探针 | 重建损失 | valid 重建 R² |
|---|---:|---:|---:|---:|
| 教师 packet 分支 | 18.044 | 16.137 | -1.907pp | 0.790 |
| 教师 window 分支 | 23.880 | 14.745 | -9.135pp | 0.714 |

教师两个分支拼接的 ridge 探针 valid F1 为 28.251%，单独对 CE 学生特征训练的探针为 17.920%；原冻结融合教师和 CE 学生分类头的 valid F1 锚点分别为 33.112%、24.873%。教师 window 探针三个 seed 为 23.091/24.208/24.341%，均高于教师 packet 探针 19.484/17.895/16.753%。这不支持“window 特征本身没有类别信号”的解释。

学生表示线性重建教师 window 特征有一定总体拟合（valid 相对 source 均值预测 R²≈0.714、标准化 MSE≈0.275），但用重建特征分类时 F1 从 23.880% 降到 14.745%。相比之下，packet 特征重建的分类损失约 1.907pp。故当前更具体的瓶颈是：从冻结 CE 学生的 128 维 packet 表示做线性映射时，window 分支中对类别关键的细节没有被保留。总体 R²较高不足以保证判别边界可迁移。

## 限制与下一步判断

教师 window 探针 source F1 约 57.93%、valid 23.88%，本身存在明显源域到验证集的落差；不能只依据 source 的高判别力设计蒸馏目标。这个诊断使用冻结 CE 学生特征和线性 ridge，不能证明端到端非线性学生一定学不会，也不能从结果唯一归因于模型容量、目标坐标、损失权重或训练样本少。教师与学生 checkpoint 已在同一 valid 上选模，因此这些指标只属于 TemporalDrift 机制开发。

若继续训练期独用路线，应事先固定一个更侧重类别关系、而非逐维总体 MSE 的转移目标，并保留同教师 packet 特征控制；在同一小 valid 上连续调 MSE 权重不能作为独立验证。若部署允许推理保留 window，已有融合模型正向证据更直接。

## 核验

`artifacts/integrity.json`：18 组 valid 探针/重建预测、36 组冻结特征/统计、24 组类别探针参数、12 组重建器参数，散列、系数重算、预测和指标复算均 0 错误。脚本核对 source/valid 方向交叉 0、原 run 教师/学生 valid F1 锚点与 source window 标准化统计。此前 checkpoint、代码、配置、固定抽样及完整性文件散列见 `artifacts/input_seal.json`；输出散列见 `artifacts/output_seal.json`。原始数据只读。
