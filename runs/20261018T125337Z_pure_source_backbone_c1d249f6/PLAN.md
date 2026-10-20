# 纯净源域 wide_bn 基石模型

## 问题

从头训练 wide_bn，仅以审计清单中的 18,553 条 TemporalDrift/train.npz 行作为源域训练，建立可追溯的开发基线。旧 checkpoint 仅作历史背景，不进入初始化、训练或选模。

## 数据与权限

`conservative_train_candidates.csv` 的 18,553 行须全部且仅来自 `TemporalDrift/train.npz`，覆盖 102 类，按原始行号读取。清单 `applied=False` 表示历史审计未应用；本 run 在用户当前明确指定后将其作为独立实验输入，不能反写原始文件或称为新确认集。原 `TemporalDrift/valid.npz` 的 2,160 行只用于预设 checkpoint 选择及开发指标；无独立 Day 0 测试集。Day14/90/270、WTT-Time、AWF 和其他因素数据在本 run 关闭。

## 模型与训练

运行本地复制并校验的 wide_bn progressive 模型；输入为前 5,000 个有符号相对时间戳及按原规则重建的 2×1,800 TAM。生成器输出 fixed 30 维与 learned 80 维；末端分类头为 102 类。采用源期既有 AdamW、batch 64、span mask、12,800 步和原三段学习率。三 seed 为 42、100、2026。20 次固定验证机会，按验证 accuracy 最大且最早达到的 checkpoint 选择；无追调和目标日期选模。

## 预算、停止和产物

物理 GPU 1，单进程，每 seed 最多 18,000 秒，总计最多 54,000 秒、输出最多 4 GiB。NaN、输入审计不符、预算耗尽或 checkpoint 重载验证失败即停止并保留证据。保存逐 seed checkpoint、训练历史、valid 预测和 Accuracy/Macro-F1；单个别名 `pure_source_wide_bn.pth` 不替代三 seed 文件。该 valid 成绩是已观察源期开发指标，不称独立 Day 0 test。
