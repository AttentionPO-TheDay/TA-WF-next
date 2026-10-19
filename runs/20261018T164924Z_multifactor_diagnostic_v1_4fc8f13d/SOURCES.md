# 实现与结果来源

model.py 及其九个本地依赖从纯净源训练 run 显式复制，逐文件来源和 SHA256 在 artifacts/code_sources.json；无 sys.path 修改或旧项目训练导入。token_alignment.py 从本项目既有源域 Token run 复制，保留旧 AdaIN/残差 projector 对照。runtime.py 的参数作用域规则取自已审计纯净时序核心矩阵，并在本 run 显式重新实现。prepare.py 的 TAM 公式与纯净源训练相同并作逐数组核对。

本 run 新编写原型关系目标、清单构建、流水线、重载审计和图表，不复用目标性能冒充新实验。源 checkpoint、源 Token reference、逐行身份清单、内容支持标签证据、外部 v4 全表均按 config 明示引用并冻结散列。
