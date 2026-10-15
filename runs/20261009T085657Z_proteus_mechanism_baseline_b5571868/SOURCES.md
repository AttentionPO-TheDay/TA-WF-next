# 来源

- 官方 Proteus 固定提交：`Xinhao-Deng/Adaptive-WF-Attack@4cdab4163bf3de7036a2498dac533c888b97664d`，本地只读快照位于 `/tmp/proteus_audit.HOKVvy/Adaptive-WF-Attack-4cdab4163bf3de7036a2498dac533c888b97664d`。核心参考为 `exp/proteus.py` 的 `compute_gaussian_kernel`、`calculate_mmd_loss`、`compute_gmm_probabilities` 和 `adapt_model`。
- 本 run 显式复制本项目已有模型最小闭包到本目录；不导入旧工作区训练器，不加载官方 RF checkpoint。
- 三个 wide+BN checkpoint、prepared tensor 和 TemporalDrift 日期输入只读引用本项目既有 runs；冻结 split 和历史预测只读引用 `20261009T070330Z_tta_rollback_acceptance_91f808d9`。
- 官方论文/代码审计及其与本项目独立 query 协议的差异见 `/home/rbf/TA-WF/HANDOFF.md` 和旧项目 Proteus protocol audit；本 run 不执行官方 target-label best-F1 选择。
