# WF 模型来源与接口

迁入日期：2026-09-15。直接来源为旧项目中的独立模型文件，未复制整个旧包。

| 本地文件 | 直接来源 | 来源文件 SHA-256 |
|---|---|---|
| df.py | /home/rbf/TA-WF/src/ta_wf/models/DF.py | 5c033a5a6576607e741a76dae368d7cc293a7d191a55713ef5161f5c37dd625e |
| varcnn.py | /home/rbf/TA-WF/src/ta_wf/models/VarCNN.py | bec5439cdb5284e4290775629b67ae4f8430750435a40aacb9043bcdb68d9415 |

这两个文件最近的旧仓库提交为 `41c7575a708f0c599ca593fe6ce29b59009f11a9`（模型代码初始化）。以上哈希标识实际读取的工作树文件。旧文件未附上游 URL 或许可证声明；目前只能确认旧项目来源，不能称为作者官方实现或已完成论文忠实复现。对外分发前仍需核实上游授权。

## 迁移范围

- DF：保留层结构、state_dict 键和 `(logits, features)` 返回行为；移除未使用的导入，添加特征提取方法。
- VarCNN：保留旧方向＋时间双分支实现及参数键；移除未使用的导入，给 Encoder 添加池化前特征接口。
- VarCNNDirection：显式新增的方向单分支派生版本。复用原方向 Encoder，删除时间分支，将分类器宽度由 1024 改为 512。不能标作未经修改的完整 Var-CNN。
- 运行时只依赖 PyTorch，不导入旧项目，不读取 checkpoint，不包含数据预处理、适应逻辑或训练器。

## 输入输出

所有输入均为浮点 Tensor，输出 logits 不含 softmax。训练模式含 BatchNorm，常规训练 batch 至少为 2。

| 类 | 输入 | forward 返回 | 额外接口 |
|---|---|---|---|
| DF | [B, 1, 5000] | logits [B, C]、features [B, 512] | forward_local → [B, 256, 18]；forward_features → [B, 512] |
| VarCNNDirection | [B, 1, L] | logits [B, C]、features [B, 512] | forward_local → [B, 512, ceil(L/32)]；forward_features → [B, 512] |
| VarCNN | [B, 2, L]，方向＋真实时间表示 | logits [B, C]、features [B, 1024] | 各 encoder 的 forward_local；不将双分支地图混称为单一局部表示 |

DF 原分类器固定接收 256×18 维特征，标准长度为 5000；不通过新增自适应池化暗改原架构。其他长度不能默认兼容。VarCNNDirection 已检查 3000、4097、5000 长度。

模型不会自动裁剪、补零、标准化或把时间戳转换成方向。这些属于后续数据适配器及实验配置。局部接口只暴露最终卷积图，不提供有效长度 mask；padding 可能影响卷积与池化，后续局部匹配必须显式处理有效位置和感受野。特征接口独立调用会重算网络；比较一致性应使用 eval 模式，训练模式的 dropout 和 BN 会改变结果。

## 验证方式

`tests/test_models.py` 使用随机输入检查接口、梯度、长度适配和可选迁移一致性，不运行优化器、不加载真实数据或 checkpoint。

本次 4 项检查全部通过。测试解释器为 `/home/rbf/TA-WF/.venv/bin/python`，PyTorch 为 `2.12.1+cu130`，实际计算仅在 CPU。DF 与双分支 VarCNN 的同权重输出以 rtol=0、atol=0 比较通过；方向 Encoder 同样通过精确比较。系统 Python 当前未安装 PyTorch，本次未安装或复制虚拟环境。

普通独立检查：`PYTHONPATH=src python -m unittest discover -s tests -v`。

迁移时可显式指定 `WF_MIGRATION_SOURCE=/home/rbf/TA-WF/src/ta_wf/models`，额外比较随机初始化的相同权重下 DF/双分支 VarCNN 的精确输出以及单分支 Encoder 一致性。该可选检查按文件路径读取参考源码，正常包和普通测试均不需要旧目录。
