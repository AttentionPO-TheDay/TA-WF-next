# ARES 源码审计（未训练）

结论：保留为第二梯队候选，不建议直接加入首轮 5000 包公平比较。Proteus release 的 ARES 只需要方向，但固定 10000 长度且 eval 仍随机循环平移；直接移入会同时改变输入预算及预测可重复性。其来源只能表述为 Proteus bundled WFlib port，尚未建立与 ARES 作者原始实现的逐项一致性。

## 来源与证据

固定 release commit `4cdab4163bf3de7036a2498dac533c888b97664d`，只读来源 `/tmp/proteus_audit.HOKVvy/Adaptive-WF-Attack-4cdab4163bf3de7036a2498dac533c888b97664d`。最小必要源码按原相对路径保存在 `artifacts/ares/`，逐文件 SHA256 见 `artifacts/ares/manifest.json`。没有 import/执行任何上游模块，没有数据读取、训练、性能评分或 GPU 操作。

根 README 只提供 WFlib 链接和安装 bundled wflib_copy 的说明，没有 ARES 作者仓库或原论文标题。因此 ARES 原始 multi-tab 任务、作者代码与该单标签移植的一致性仍是未关闭来源项，不能凭模型名称宣称完整原生复现。根 README 声明 MIT，但 archive 中 LICENSE 文件缺失，manifest 如实记缺失；代码可供本地审计，发布复制代码前需补核许可证来源。

## 实际 TemporalDrift 配方

证据：`scripts/TemporalDrift/ARES.sh:1–17`、`exp/train.py`、`wflib_copy/WFlib/tools/{data_processor,model_utils,evaluator}.py`。

| 项目 | release 的实际行为 |
|---|---|
| 输入 | DIR：`np.sign(X)`；右截断/补零至 10000，float32，形状 N×1×10000 |
| 标签 | num_tabs 默认 1；int64 单类别标签；CrossEntropyLoss；不是多标签原始任务复现 |
| 初始化 | 无预训练；脚本固定 Python/PyTorch/NumPy seed=1013 |
| 优化 | AdamW，lr=0.002，其余未指定，故采用库默认值（通常 weight_decay=0.01, betas=(0.9,0.999), eps=1e-8；正式复现应钉住版本） |
| 预算 | 30 epochs，batch512，shuffle=True、drop_last=True；无 scheduler |
| 选模 | 每 epoch valid，四舍五入至4位的小数 Macro-F1 严格增大才保存；30次机会 |
| 返回 | logits 与256维 mean pooled feature |

以本项目 15300 条训练样本作静态预算推算，每 epoch 29 次更新，30 epoch 共870次；不足一个 batch 的452条随shuffle每轮舍去。18553条则36次/epoch、1080次总更新。上述只是公式计算，没有读真实数据或训练。不能把相同 epochs 等同于相同更新数/算力；新实验应预先选择“原配方比较”或“匹配预算比较”，并披露差异。release shell 后续还会自动测试未来日期并运行 Proteus，不得执行该 shell。

`wflib_copy/scripts/ARES.sh` 另含 MTAF、8000长度、300epochs、StepLR 配方，与 TemporalDrift 路径及模型单输入卷积不一致。不能混抄这个配方作为方向版 ARES；应以实际 TemporalDrift 调用链为本次审计对象。

## 结构、长度与可重复性

`models/ARES.py:105–163`：输入沿长度四等分，共享 LocalProfiling：4个残差卷积块，每块2层Conv1d(k7)+BN+ReLU，通道32/64/128/256，池化k8/stride4与dropout0.1。10000输入每片2500，经池化长度2500→624→155→37→8，合并为32个256维token。加入32×256位置embedding，接4层8头top-20 attention，FFN宽1024，mean pool后单Linear分类。

- **5000不能直接替换。** 每片1250→311→76→18→3，总12tokens；与固定32位置embedding不符，top_m=20也大于12。将位置embedding或top_m改小属于显式结构变体。把前5000方向补到10000保持信息预算但不是原生输入分布，应标明adapter，不能暗中加入后5000真实包。
- **eval 非确定性。** forward第154–155行无条件采样 `np.random.randint(0,2501)` 后 `torch.roll`，同一batch用同一偏移，即使model.eval仍执行。预测依赖NumPy RNG、batch边界及历史调用；只保存state_dict不足以重现历史valid选择值。
- 正式实验应预先定义训练增强与确定性评价策略，并明确标为修订。若坚持精确复现release，必须保存/恢复NumPy状态、固定batch与顺序，且承认批次相关随机评价的限制；不能根据valid结果在多个偏移中挑优。
- 无padding mask，循环平移会把尾部零padding带到头部。不能把它直接等同于物理上合理的包丢弃或时间偏移。

## 参数与资源（静态推算）

按源码和标准timm Mlp两Linear结构计算，102类时参数约 **4,150,406**（卷积956,960；四attention块3,159,040；位置8,192；分类26,214），未运行实例确认，正式移植需重新核验。batch512拆四片后首层一个float32激活约 512×4×32×2500×4 bytes = **625 MiB**；训练会保存多层激活，不能据参数仅约4M认定显存开销小。应先做合成输入显存/梯度/重载可重复性检查，再决定并发。若调整batch需记录BN行为和更新预算差异。

## 后续优先级

当前源期90%目标没有证据由该模型保证。首轮优先来源闭合且满足5000方向输入的候选；ARES列第二梯队，先补作者代码/原论文来源，再冻结长度、padding、eval平移策略、预算及选模机会。若采用Proteus端口，名称应写“ARES Proteus-release adaptation”，不要称ARES原论文完整复现。任何新训练使用既有固定source/valid清单和三seed，未来保持关闭。
