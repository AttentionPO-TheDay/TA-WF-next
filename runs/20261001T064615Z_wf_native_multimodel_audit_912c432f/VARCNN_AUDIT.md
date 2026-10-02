# Var-CNN 原生实现审计（2026-10-01）

结论：推荐进入下一轮的候选是 **Var-CNN 官方支持的纯方向配置 `mixture=[["dir"]]` 的显式 PyTorch 移植**。本项目现有 `VarCNNDirection` 不是该配置的忠实实现，历史分数不能视为作者原生方向版本的性能上限。本次仅源代码审计；未训练、未读真实数据或 checkpoint。

## 来源与范围

- 作者仓库：https://github.com/sanjit-bhat/Var-CNN 。README 第3–11行声明对应 PETS 2019 论文及作者；仓库描述一致。最初候选 `vperium/Var-CNN` 在 GitHub connector 返回404，不作为来源。
- 固定提交：`e5db76a86fcbf8839f764b9aca241749d1c1f700`，作者提交时间2019-09-02。通过 GitHub connector 先解析 master，再以该提交逐文件抓取；未执行上游代码。
- 9份文本快照位于 `artifacts/varcnn/`；`fetch_manifest.json` 记录每份 Git blob SHA、固定URL、快照SHA-256和本地比较文件SHA-256。快照由文本工具写入，末尾换行可能规范化，故 blob SHA 与快照SHA-256分别保存。
- MIT许可证，版权归 Sanjit Bhat、David Lu、Albert Kwon、Srinivas Devadas；移植需保留声明。残差基础代码注明借鉴 broadinstitute/keras-resnet，进一步分发时保留此来源。
- 原依赖：TensorFlow1.3.0、Keras2.0.8；另有 h5py/tqdm/sklearn。README记录 Ubuntu16.04、CUDA8、cuDNN6。不能假设当前PyTorch环境可以原样运行；不建议为本轮安装过时GPU栈。

以下行号均指 `artifacts/varcnn/` 中固定快照，除明确写“本地”者。

## 官方可选输入与结构

| 项目 | 官方证据 | 对本项目的含义 |
|---|---|---|
| 默认完整配置 | `config.json:17`：`[["dir","metadata"],["time","metadata"]]`；README解释分别训练后平均预测 | 完整默认版本是两个模型的集成，不能把单方向分支称完整默认Var-CNN |
| 纯方向配置 | `run_model.py:24–30`允许非空任意dir/time/metadata组合；`var_cnn.py:225–268`条件建立输入 | `[["dir"]]`是官方代码直接支持的变体，不是本项目发明 |
| 方向编码 | `var_cnn.py:176–204`：ResNet18、4阶段×2块、64/128/256/512通道，dilation=(1,2)/(4,8)，GAP | 有序方向序列5000包、末尾零padding；不额外masked pooling |
| 单模态分类头 | `var_cnn.py:270–284`：只有多个模态时才插入Dense1024/BN/ReLU/dropout0.5 | 纯方向应直接512→类别数Dense，不应加隐藏层 |
| 时间输入 | `preprocess_data.py:150–158`差分相邻包相对时间 | 当前跨方向时间回退语义未闭合，不可直接套用；本轮关闭 |
| metadata | `wang_to_varcnn.py:20–49`：总包数、入/出数、比例、总时长、平均时间 | 包含时间且计数使用完整trace，非仅前5000包；关闭，不能把7维元数据称同等方向权限 |

官方原始解析器实际硬编码5000长度（`wang_to_varcnn.py:21–34`），尽管README将seq_length描述为可配。5000与当前预算匹配，不额外修正或扩大。

## 与现有本地 `src/ta_wf_next/models/varcnn.py` 的关键差异

1. **因果padding**：官方膨胀块 `var_cnn.py:48–62` 为 `padding='causal'`，每层只在左侧补 `dilation*(kernel_size-1)`；本地第15/18行对称padding。输出长度相同不代表特征语义相同。
2. **首块shortcut**：官方第68–77行每个stage的block0都有1×1投影及BN，含stage0的64→64；本地第22行只在stride/通道变化时投影，遗漏stage0投影。
3. **首层卷积及pool**：官方第184行conv use_bias=False，本地第56行默认bias=True；官方第187行Keras SAME池化需按长度动态分配padding，本地第59行固定左右padding1。在5000输入后长度2500时，两者窗口对齐不同。
4. **分类头**：官方纯方向GAP后直接分类；本地第132–138行另加Linear512→512/BN/ReLU/dropout0.5，显著不同。
5. **初始化/BN/optimizer默认值**：官方残差卷积显式he_normal（第16行），首层卷积与Dense使用Keras默认；本地使用PyTorch默认初始化。BN epsilon主干相同1e-5，但Keras与PyTorch默认momentum约定不同。移植必须显式记录且核对Keras2.0.8默认值及running variance语义，不能仅凭同名层认定等价；Adam默认epsilon等亦须核对。

因此可复用本地代码作理解参照，不应直接加一个“native”别名；应按固定官方源码重建最小模型，保留当前历史实现不动。应有合成短序列因果卷积/stride、SAME pool窗口、stage0投影、输出形状和梯度检查；若未做跨框架同权重数值比较，只能称“按源码移植并结构核验”，不能称精确数值复现。

## 原生训练配方与公平比较边界

- 默认batch50、最大150 epochs、base_patience5（config）。CE，Adam lr0.001（`var_cnn.py:286–289`），未显式权重衰减、增强或label smoothing。
- 每epoch validation accuracy驱动ReduceLROnPlateau，factor=sqrt(0.1)、patience5、cooldown0、min_lr1e-5；EarlyStopping patience10；保存最佳val_acc checkpoint（第291–300行）。并非固定45epoch AdamW训练。
- 原训练数据额外随机留出5%做内部validation（`preprocess_data.py:201–212`）；原作者test是另一个集合。本项目使用固定source150与valid510，不能运行其随机划分器重写数据角色。保留全部15300条训练、valid510选模应明确为“本项目固定划分适配”。
- 官方generator以一次打乱后的顺序循环，fit_generator shuffle=False（`run_model.py:45–54`，`data_generator.py:74–98`），不应无声改成每epoch shuffle。
- 默认完整模型含时间/完整trace metadata与集成；论文完整模型数字不构成本轮纯方向准确率承诺，尤其不能保证90%。

## 建议的有界下一轮（草案，非本次执行）

1. 只运行一个配置：官方纯方向膨胀ResNet18，长度5000、102类、无时间/metadata、无增强，scratch seeds1729/3407/2026；沿用固定source150抽样与valid510清单，未来关闭，不载旧权重。
2. 最多150 epochs×3 seeds，batch50、每epoch306步；最大45900步/seed，总137700步。GPU0，Var-CNN最多1任务同时，单seed60分钟、此模型累计3GPU小时；触达资源上限即停止并如实标为预算截断，不能称收敛。
3. Adam0.001、原生plateau/early-stop规则；精确冻结旧Keras回调的min_delta/epsilon等默认值后才可启动。每epoch1次valid，全轮最多150次/seed。以accuracy择checkpoint，等值保留较早checkpoint；同步报告macro-F1，不另外搜索。
4. **选模机会不可与历史20次选模模型直接声称匹配**。若多模型公平竞争轮采用最多150次验证，则各对照也冻结同样上限和accuracy选模；历史CNN/DF仅标作历史参考。若团队改用统一20次，则需把plateau/early-stop的计数单位明确定义为验证事件，并标作训练协议适配，不能仍称原生epoch规则。
5. 记录训练样本呈现次数、参数量、GPU秒、实际验证次数和停止原因；导出best与last、逐样本预测并独立重载复算。三seed均须报告，达到90%判断使用平均accuracy，不取最佳seed。

建议优先级：高于继续调整现有VarCNNDirection隐藏层。它能检验“原生方向ResNet配方是否超过当前开发基线”，但尚不能归因哪个具体层造成差距，更不能证明时间泛化。
