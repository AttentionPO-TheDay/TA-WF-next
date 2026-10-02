# 源期150条生成器约束与温和dropout候选

状态：用户已授权执行；实现和启动检查通过后冻结配置再训练。用户当前要求继续提高源期准确率，不进入Day14；本文件是候选设计，不代表机制已有效。

## 问题与依据

固定当前最好150条/类+1层生成器分类器，验证准确率54.248%、Macro-F1 52.440%。已完成的20条正则化实验中强分类器正则化有害；生成器锚定+1.700pp但仅2/3seed正。此处拟检验更多训练样本下较温和的约束，不能把旧结果说成支持该方案。当前曲线也未定位唯一瓶颈，表示信息保留仍是另一候选；不同时修改表示。

## 四条件

R00显式复用上一轮B_150_l1三seed，不重复训练。R10仅增加generator初始化锚定，lambda=0.001；R01只将classifier dropout由0.1调至0.15（包括attention dropout），weight_decay保持0.0001；R11两者组合。L=source CE+lambda/2*sum((theta_generator-theta_initial)^2)，theta_initial为本job从头初始化常量，不加载旧项目权重。强度为固定有界候选，不搜索。生成器和分类器仍端到端学习。

## 数据、配对与评价

复用20260927T032819Z_source_size_local_depth_factorial_dae714b7的source150、共同source80与valid510，执行前核对配置数据清单及全部散列；原始数据只读，无新增抽样，不重新划分valid。source用于监督与诊断；valid仅固定选模/评分，不用于梯度。禁止Day14及其他未来/外部数据、预训练、TTA与蒸馏。

三seed21729/23407/22026，初始权重和batch索引配对；每job12800步、batch64，保持原LR计划和3680..12800的20次选模，valid Macro-F1最大、并列最早。报告best/latest source/common80/valid accuracy与Macro-F1，完整曲线及参数偏移。

主比较R10/R01/R11各自对R00，辅以组合对单项和2×2描述性交互。候选门槛预定为valid accuracy均值至少+1pp、三seed accuracy均正、Macro-F1均值不下降；不以训练准确率提高或差距缩小判为成功，不称统计显著。仅使用同一已观察开发valid，历史R00不作为新重复。

## 执行资源和实现检查

9个新任务，CPU6并发×2线程，单job14400秒、pipeline30000秒、RSS总24GiB；错误/超时保留结果，不自动重启或追调。需先实现并核对lambda=0精确回退、锚定仅作用生成器、dropout范围、初始化与采样配对、历史checkpoint重载，再冻结实际配置代码。训练入口必须拒绝draft。完成后独立重载best/latest与预测核验，自动更新RESULTS/STATUS/EXPERIMENTS。训练开始前不存在新性能结果；本轮不根据中途结果追加条件。

## 实现及核验细则

显式改编本项目20260927T032819Z_source_size_local_depth_factorial_dae714b7的run-local训练、监督、核验框架，不修改共享src；复用/home/rbf/TA-WF/.venv环境但不导入旧项目训练代码。source150=15300、common_source80=8160、valid=510，复用原prepared和manifest并冻结散列，configs/datasets.json的配置同样冻结。

新任务固定400个block×32步；每步记录CE、锚定项和总损失，评价点记录生成器参数相对初始化的L2偏移及token变化。所有条件相同AdamW weight_decay0.0001，生成器没有dropout，只有分类器dropout为0.1或0.15。初始化锚点为参数clone、requires_grad=False。

执行前用合成输入核对全部条件共同初始化、显式Dropout与MultiheadAttention概率、lambda=0梯度与更新精确回退、锚定项梯度等于lambda*(theta-theta_initial)且不流向分类器、source CE梯度仍更新生成器。历史B三seed的best/latest重新评分18组预测；新9任务每任务best/latest×source/common_source/valid共54组，累计72组核验全部通过才记completed。核验器独立计算F1、重建初始化并复算锚定项、优化器步数与LR、训练样本流、选模时点和所有预测。启动后自动核验汇总，助手不持续轮询。

启动检查修订：首版仅进行了12次丢弃式source接口单步（人工标签），无新模型保存或性能评分，覆盖不足；首版检查与冻结记录作废并保存在artifacts/preflight_v1_superseded。v2使用完全合成输入补齐零系数精确回退与梯度隔离，最终代码齐备后重新冻结。
