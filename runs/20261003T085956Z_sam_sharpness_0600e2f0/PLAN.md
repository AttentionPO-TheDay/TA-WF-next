# 20261003T085956Z_sam_sharpness_0600e2f0

## 问题

当前模型 source 接近完全拟合而 valid 停在约89%，本轮测试锐度感知优化（SAM）是否能提高参数扰动下的泛化。SAM 使用同一个模型和 source CE，不引入教师、EMA 或一致性目标。为区分 SAM 参数扰动与两次前向/反向本身的影响，设置完全匹配的 `two_pass_control`。

## 条件

- `two_pass_control`：每步对同一个 span-mask batch 做两次 CE 前向/反向；丢弃第一次梯度，只用第二次梯度更新，不做参数扰动。
- `sam_ce`：同样两次前向/反向；第一次梯度用于 `rho=0.05` 的全参数 L2 扰动，第二次在扰动参数下计算 CE 梯度，恢复参数后用第二次梯度更新。

两条件同 seed、source索引流、mask流、模型初始化、前向顺序、BN和dropout规则及训练步数；SAM只改变第一次反向后的参数扰动。

## BN、随机性和SAM规则

第一遍 forward 在 train 模式，更新 BatchNorm running statistics；保存第一遍后的 BN buffers。第二遍保持 dropout train，但把 BatchNorm 临时设为 eval，避免第二遍重复更新 running statistics；第二遍结束后恢复保存的 BN buffers。该规则对 control 和 SAM 相同。

对全部 `requires_grad` 参数计算梯度 L2 norm；`epsilon=rho*grad/norm`，`rho=0.05`，只用于 SAM 条件。第二遍梯度用于 AdamW 更新，第一遍梯度不进入 optimizer state。梯度非有限、norm为0、扰动恢复不精确或BN buffer变化超出规则立即停止。

## 数据与权限

TemporalDrift source15300、valid510、102类；source标签用于CE，valid只用于预定选模/评价。未来日期、WTT-Time、AWF、TTA关闭。

## 判定

两条件各运行seed 21729/23407/22026，共6项新训练；历史单视图CE只读参考。主比较 `sam_ce - two_pass_control`，候选需平均accuracy至少+1pp、3/3 seed正且Macro-F1不下降；同时报告相对历史CE，不把两次前向控制收益归因于SAM。

## 预算

GPU0、最多2路并行、单项7200秒、总43200秒；FP32、确定性、AdamW学习率和12800步继承wide+BN基线。固定20个valid选模步点，最高accuracy、并列最早。固定mask诊断只读，不选模；结果仅属已观察source/valid开发。
