# 明确来源

本目录基础架构、输入、LoRA、邻域和v6 RefinedState/objective显式复制自 runs/20261019T052905Z_prototype_refinement_v6_da9bb898，源链见其SOURCES.md；未隐式导入旧项目。BN逐层population统计算法最小复制自v3/tent.py，并拆为纯统计/纯affine条件。新增isolation.py、worker.py、scoring.py、controller.py及audit.py属于本轮。历史TENT和v6 LoRA仅在配置声明的控制/分解中加载，SHA与每cell记录核验。

frontier_base.py显式复制自v6，仅调用既有curve/writecsv函数；report.py为本轮预定比较和作图。服务器掉线恢复后核实：无v7 GPU进程，artifacts为空，故没有重复任何已完成适应。
