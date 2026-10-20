# 多原型 × 可靠性门控 × Q/V LoRA v2

## 问题与暴露
已观察诊断v1中Network负收益，以及外部v4 AdaIN/projector阴性结果后提出本版本。检验4子原型、源域可靠性门控与rank4 Q/V LoRA的主效应及交互；不保证收益，不作为独立确认。仅运行已批准的10个诊断切片，不新增WTT/AWF评分。

## 数据、预算与冻结
复用诊断v1同一十切片、逐行清单与源/valid隔离审计；不重新分割。3个纯源模型seed42/100/2026来自唯一18,553条Temporal源行，不训练底座。source labels仅拟合每类原型；reference 1024/切片去除标签，仅用于适应；query输入不进入适应，query标签仅终态指标与事后分析。Hostname仅原有分组依据。内容对应的102身份假设沿用诊断规范，不宣称官方网站映射或已知丢包率。

完整2×2×2：single/gmm4 × ungated/routed × residual/lora，240新单元。各自从纯模型和零残差重新开始，所有组合共享固定400步、batch64、source回放32、Adam lr0.00025、梯度裁剪1、无dropout、最终步选模。旧v1 frozen/tent_bn/gate/prototype结果仅明确标注历史参考；v1用800步，不能冒充本轮匹配重跑。GPU0/1/2各一seed，最多4小时/12 GPU小时，单元600秒，输出4GiB；错误/不有限/超预算停止并保留结果。

## 方法定义
共同128维空间为Transformer后final_norm的time-pool，按纯源均值/标准差标准化。每类独立GMM，K=1或4，diag covariance，reg=1e-3，n_init3，max_iter200，固定随机种子；只用18553源行拟合。query视为退化高斯N(z,0)，到高斯分量W2²=||z-mu||²+tr(Sigma)，按128维平均；这是点到分布距离，不冒称包含测试样本协方差。固定teacher预测类别内，以prior·exp(-W2/T)软分配，T为源域最近W2中位数。
门控源域校准H95、D99，D为全类最小平均平方Mahalanobis距离，tau=1/源D中位数。超过任一阈值严格w=0，否则w=exp(-tau·D)。权重由冻结teacher计算，训练时加权原型拉近和类心余弦关系损失，推理时w=0逐行返回冻结模型logits；仅加权损失并不等于推理旁路。
损失=.1*期望W2 + 1*类间余弦拓扑误差 + 1*source表征恒等误差；无熵最小化。源锚1024按source row hash固定选取。所有组合匹配此损失、表示空间、步数与索引流。
Residual沿用110→16→110零初始化轻残差，3866参数；LoRA使用两层attention的Q、V全部rank4旁路，K不变，4096参数约0.39%，明确不符合<0.1%的原估算。LoRA只由共同原型/关系和源锚损失更新；底座/分类头/BN完全冻结。禁用MHA eval fastpath避免跳过参数化；零初始化、梯度流、原参数及BN不变用纯源样本验证。LoRA作用在Transformer内部，不能声称修改Transformer之前的Tokens。

## 指标与裁决
完整逐seed×slice报告Macro-F1/Accuracy/ΔF1/NAR/Recovery、reference与query门控覆盖率、阈值/距离/熵分位数、参数量、成本。版本/地域/行为分别等权后再三因素等权，均值±seed样本标准差；报告全因子配对主效应与交互，不以最好切片挑选方法。门控拒绝行logits必须与冻结模型逐位相同；NAR可能>0，不能声称零误杀。全组合呈现阴性结果，不据query反馈返调。源域模型前向及LoRA梯度检查通过后冻结输入、代码、GMM与校准文件散列再启动三GPU评分。
