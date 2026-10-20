# TemporalDrift 核心迁移矩阵

依赖纯净源域三 seed checkpoint 完成并通过重载审计。仅在 Day14/90/270 的既定开发角色上比较五条件，每个 seed×日期×条件共 45 单元；单元最大 2,400 秒、800 候选更新。reference 1024 条按固定规则无标签使用，query 与 reference 输入隔离，query 标签只用于终端评分。

冻结前须逐条件核对实现：`tent_bn` 的 query 批统计量权限、`tent_entropy_gate` 的 BN/LN/分类头冻结集合、`proteus_default` 的完整伪标签流程与 1.0x 强度、`no_pseudo_fixed` 的 0.25x 强度及分类头锁定。此前同名实现未必满足本提案，不能按名称直接复用。固定 logits L2、熵分布、Accuracy、Macro-F1、NAR、Recovery 记录规则。已有 Day14/90/270 评分是方法开发暴露，本矩阵结果亦为开发证据。

执行冻结版：纯净权重三个 seed（42/100/2026）分别在 Day14/90/270 的同一独立 query 上配对比较 `frozen`、`tent_bn`、`tent_entropy_gate`、`proteus_default`、`no_pseudo_fixed`。九个 `frozen` 单元复用并逐文件核验已完成的纯净冻结 run；本轮新增 36 个适应单元，合计主表 45 单元。固定 102 类 Macro-F1 为主指标，Accuracy、NAR、Recovery 为辅；每个 seed×日期等权汇总，不按目标 query 选步数或参数。1024 条 reference 无标签，来源与 query 索引严格沿用冻结基线；query 标签仅在最终预测固定后读入计算指标，query 输入不进入梯度或参考库。

四种适应统一 800 候选步、batch128、固定 source/support loader seeds 2025/2026、Adam、按终端步评分。`tent_bn` 仅更新 BN 仿射并用 batch 统计量；其 query 前向也读取 query batch 统计量，因此单独声明为 transductive-forward 控制，和其他条件的信息权限差异须明示。`tent_entropy_gate` 以源 BN 统计量固定，分类头冻结，只更新 Transformer 与末端 LayerNorm 仿射；只对支持集熵 0.2–0.4 nats 的样本反传。`proteus_default` 更新全模型，lr=0.001，包含源 CE、每40步刷新的两分量 GMM 硬伪标签 CE、目标熵/边际多样性及标准 Gaussian MMD。`no_pseudo_fixed` 去除伪标签分支，lr=0.00025，分类头参数冻结，保留源 CE、目标熵/多样性与 MMD。GMM 阈值0.6；不对目标 query 拟合。

正式执行前的 `preflight.py` 只在 Day14 无标签 support 上完成每条件两步作用域/重载 smoke，未用 query 标签或评分。运行入口拒绝 draft；先核对三日期 source/valid/reference/query 的精确输入隔离和全部冻结散列，再进行适应。每步记录平均 logits L2、熵、20箱熵分布、门控保留数和loss；适应后保存 checkpoint、query logits/预测、逐单元指标，`audit.py` 重载36个 checkpoint 的首批 query 输出并重算45条结果。程序或协议断言失败时保留部分证据并标记 failed，不用失败结果调参或挑选单元。

资源预算：GPU0 单进程、CPU2线程、每新增单元≤2400秒、总墙钟≤43200秒、输出≤2GiB。已经完成的九个冻结单元不重复耗费适应预算。该矩阵仅是已观察 TemporalDrift 的方法开发证据；不得用其结果回写已冻结的外部未来评价版本。

2026-10-18 版本说明：原占位 run `20261018T125337Z_temporal_core_matrix_5002c34a` 被并行工作流写入互不兼容的 config 与 runner，未冻结、未评分。本独立 run 复制最小实现并统一配置、作用域、四个新增条件和审计，保留原 run 不覆盖。九个冻结单元只作为已审计历史参考；本 run 的支持集 smoke 只用源输入，不读真实目标 query。所有修订在本 run 正式目标适应与 query 评分之前冻结，不依据本轮结果调整。

训练与选模：源权重采用已完成纯净源 run 的官方 valid 最佳 Accuracy 选模规则。本矩阵不训练新源权重；适应固定终端 800 步，不用目标 query 选择 checkpoint。三个 seed 使用同一 GPU 顺序执行，最多一个进程；冻结权重与输入 SHA256 写入本 run 的 freeze.json。支持集包含目标输入但不含标签；Proteus 的伪标签由无标签支持集预测产生。不做额外试验搜索。任何失败保留产物并登记，不挑选成功单元报告。
