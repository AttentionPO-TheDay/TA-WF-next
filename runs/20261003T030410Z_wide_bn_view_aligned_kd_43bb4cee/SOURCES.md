# 来源与复用

- `model.py`、`framework_base.py` 等为上一轮 wide+BN 实验的显式最小代码复制；未导入旧项目运行时代码。
- `prepared.pt`、三组历史 teacher logits、teacher checkpoint 和初始化状态来自已完成的 `20261002T053706Z_progressive_capacity_bn_mixup_1ad233da`，均作为冻结历史输入。
- `matched_kd` 和 `gated_matched_kd` 在线调用冻结教师，对学生使用的同一 masked batch 产生概率目标；`clean_kd` 使用 clean source 教师目标并让学生以 clean 输入训练。
- 所有数据角色、标签权限、选模和预算见 `config.json` 与 `PLAN.md`；未来日期、WTT-Time、AWF、TTA 均关闭。
