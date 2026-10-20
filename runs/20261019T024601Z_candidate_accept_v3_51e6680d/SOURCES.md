# 显式复用

架构及最小辅助逐文件复制自 `runs/20261018T183916Z_gmm_routing_lora_v2_28dc7c4b`；来源链见其SOURCES.md。methods.py保留QVLoRA、encode、tokenize、固定源归一化等实现，新目标/校准/验收另写candidate.py。source GMM1及128维源特征显式引用旧产物，不作新拟合或独立重复。runtime只修改STATUS前缀；无需旧项目sys.path。诊断缓存与纯净源权重只读。
