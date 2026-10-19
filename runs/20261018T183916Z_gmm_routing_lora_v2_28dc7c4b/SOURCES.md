# 显式复用来源

模型、输入、架构辅助及ResidualTokenProjector逐文件复制自 `runs/20261018T164924Z_multifactor_diagnostic_v1_4fc8f13d`，其原始来源链见旧SOURCES.md。runtime修改为指向旧只读诊断缓存，并禁用MHA fastpath。无旧项目sys.path/隐式导入。底座与数据缓存通过config.json显式定位；GMM/门控/LoRA/训练代码为本轮新实现。历史控制不当作新的独立重复。
