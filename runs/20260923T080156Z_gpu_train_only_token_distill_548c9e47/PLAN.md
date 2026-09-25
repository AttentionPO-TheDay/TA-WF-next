# 20260923T080156Z_gpu_train_only_token_distill_548c9e47

问题：生成window token只在训练阶段通过教师蒸馏参与时，packet-only学生是否获得超越普通训练与packet教师蒸馏的valid及时间日期泛化收益？

状态：frozen，训练前登记。

问题：在推理端只有 packet 方向序列时，训练阶段生成的 window token 是否通过教师蒸馏改善同结构学生的 valid 和跨日期分类？上一轮 `20260923T075107Z_gpu_packet_window_main_378247f0` 显示融合模型在训练和推理都保留 token 时有增益；本轮从头训练，不加载上一轮 checkpoint，显式复用其局部模型定义和窗口输入规则。

数据：通过 `configs/datasets.json` 定位 TemporalDrift，沿用 `20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37` 的 source 2040、valid 510、Day14/30/90/150/270 各 2040 行固定清单。前 5000 个有符号时间戳仅提方向；50/250 window 的 partial 项置 (0.5,0)，以 full source window 统计标准化。source 标签仅用于教师与学生训练；valid 标签分别为教师与学生按 macro-F1 选最早最佳 epoch；五未来日期只在学生选模后作开发评分。不存在 query 适配或未来梯度。

教师：相同 219494 参数 packet+window 网络，`packet_teacher` 的 window 输入为常量 0，`fusion_teacher` 的 window 为真实生成 token；同 seed 初始化与 batch 顺序，AdamW 1e-3/1e-4，45 epochs，valid 选模。教师选模后只对 source 生成 logits，冻结，供学生训练。教师 valid 分数仅作教师质量记录，不作为学生主指标。

学生：相同 packet-only CNN 与分类头，仅接受 packet；三个条件 `ce`、`kd_packet`、`kd_fusion` 共享逐 seed 初始化、顺序、架构、优化器和 45 epochs/valid 选模。CE 条件仅交叉熵；两蒸馏条件用 `(1-0.5)*CE + 0.5*T²*KL(teacher/T || student/T)`，T=2.0。蒸馏 logit 仅来自 source；学生推理不生成 window、不加载教师。

指标与门槛：记录三个学生在 valid 及五日期的 accuracy/macro-F1、逐 seed 差、最佳 epoch、训练 loss/accuracy、教师 valid F1。`kd_fusion` 需相对 `ce` 和 `kd_packet` 各自在 valid 平均 F1 >0 且至少 2/3 seed 正、五未来日期均值差 >0 且至少 4/5 日期均值正，才视为训练期 token 作用的候选证据。若只改善训练集或仅优于 CE 而不优于 packet 教师蒸馏，则不能归因于生成 token。

GPU 0 一块，三 seed×两教师×45 epochs 与三 seed×三学生×45 epochs，1200 秒上限；失败保留记录。配置非 frozen、CUDA 不可用、已有 checkpoint/最终预测、固定清单异常、方向交叉、非有限输入时停止。外部 WTT-Time/AWF 不访问，TemporalDrift 结果只作开发证据。
