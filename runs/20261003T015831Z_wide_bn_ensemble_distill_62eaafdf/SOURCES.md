# 来源与复用

- `model.py`、`framework_base.py` 等是 `20261002T053706Z_progressive_capacity_bn_mixup_1ad233da` 中 wide+BN 结构的显式最小复制；未导入旧项目运行时代码。
- `artifacts/prepared.pt`、`artifacts/freeze.json` 和三组教师 logits/初始化来自上述历史 run，作为冻结历史参考，不视为本 run 新训练。
- 教师 logits 对应三个 wide+BN seed 的 source/valid 最佳 checkpoint；本 run 只在 source 上构造平均软目标，valid 仅用于学生 checkpoint 选择和评价。
- 监督、蒸馏、输入和资源权限见 `config.json` 与 `PLAN.md`；未来日期、WTT-Time、AWF 和适应数据均关闭。
