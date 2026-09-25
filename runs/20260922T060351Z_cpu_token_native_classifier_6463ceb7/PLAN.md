# 20260922T060351Z_cpu_token_native_classifier_6463ceb7

问题：生成器产生的exact/coarse run token直接作为主模型输入训练分类器是否可学习

状态：frozen，数组访问前冻结。

按用户最新澄清，生成器输出token直接作为主模型训练/推理输入，不做辅助回归。已有run CNN已是直接输入实验，本轮改为显式字段embedding和位置编码，属新的实现，不声称首次直接token训练。

数据：沿用TemporalDrift v2 source2040/valid510清单，source已排除完整valid方向重叠并去重。source标签训练，valid逐轮macro-F1选模，平局取早；报告valid开发指标，不当独立测试。本轮不读取未来日期/WTT/AWF。数据根取configs/datasets.json。

token：原始前5000观察，保留前512run，右侧padding。exact仅方向和log1p(count)/log(5001)；coarse仅方向和floor(log2(count))的类别id。方向Embedding(3,16,padding0)；exact长度Linear(1,16)，coarse长度Embedding(14,16,padding0)，coarse类别bin+1。concat得到32维，加Embedding(512,32)位置编码；两层Conv1d(32,32,k5,p2)-ReLU，每层后mask清零；把512槽按原位置分成16段，每段masked mean得到512维，再Linear512-102分类。无边界、邻居、原始包数旁路。

预算：exact/coarse×训练seed1729/3407/2026，共6次，15epochs、batch128、AdamW lr0.001 wd0.0001，CPU4线程、GPU隐藏、timeout600秒。全部从头训练，不加载旧模型。重复单位为训练seed，样本清单固定。非有限/空trace/重复隔离错误或超时停止。

评价accuracy/macro-F1、train loss/accuracy、参数/耗时，保存token输入、hash、模型、配置、预测、历史；预测先封存再最终评分。结论只判断token模型的开发可学习性，不用漂移衰减作为准入条件。与历史packet/run架构不等价，不直接作优胜归因。具体编码为本轮候选，不预设token必须embedding才可学习。
