# 生成器视角消融结果

9项新训练完成、3项融合历史复用；valid accuracy packet_direction=56.405%/packet_native=70.523%/tam_only=73.333%/fusion=74.183%；48组预测指标核验通过。

本版融合未稳定超过两个单视角，证据混合；按配对数值选择下一个受控编码/融合问题，不宣称互补性已成立。

|条件|seed|来源|best步数|source accuracy|valid accuracy|valid Macro-F1|
|---|---:|---|---:|---:|---:|---:|
|packet_direction|21729|新训练|7520|89.431%|54.902%|53.219%|
|packet_native|21729|新训练|7040|94.418%|70.588%|69.409%|
|tam_only|21729|新训练|6560|96.065%|72.941%|72.145%|
|fusion|21729|历史复用|11840|100.000%|74.314%|73.615%|
|packet_direction|23407|新训练|7520|91.235%|58.235%|57.112%|
|packet_native|23407|新训练|7040|94.732%|72.353%|71.452%|
|tam_only|23407|新训练|8960|98.137%|72.353%|71.104%|
|fusion|23407|历史复用|7520|99.333%|73.725%|72.696%|
|packet_direction|22026|新训练|5600|77.765%|56.078%|54.697%|
|packet_native|22026|新训练|6560|93.595%|68.627%|67.580%|
|tam_only|22026|新训练|7520|97.118%|74.706%|74.031%|
|fusion|22026|历史复用|12800|99.993%|74.510%|73.209%|

三seed均值：

|条件|accuracy|Macro-F1|≥90%|
|---|---:|---:|---|
|packet_direction|56.405%|55.009%|False|
|packet_native|70.523%|69.480%|False|
|tam_only|73.333%|72.426%|False|
|fusion|74.183%|73.173%|False|

```json
{
  "fusion - packet_direction": {
    "accuracy_mean_delta_pp": 17.77777777777778,
    "accuracy_per_seed_pp": [
      19.41176470588235,
      15.490196078431373,
      18.43137254901961
    ],
    "macro_f1_mean_delta_pp": 18.16411785702444,
    "pass": true
  },
  "fusion - packet_native": {
    "accuracy_mean_delta_pp": 3.66013071895425,
    "accuracy_per_seed_pp": [
      3.7254901960784292,
      1.3725490196078494,
      5.882352941176472
    ],
    "macro_f1_mean_delta_pp": 3.6929310703820586,
    "pass": true
  },
  "fusion - tam_only": {
    "accuracy_mean_delta_pp": 0.8496732026143835,
    "accuracy_per_seed_pp": [
      1.3725490196078494,
      1.3725490196078494,
      -0.19607843137254832
    ],
    "macro_f1_mean_delta_pp": 0.7468330507546288,
    "pass": false
  },
  "packet_native - packet_direction": {
    "accuracy_mean_delta_pp": 14.117647058823527,
    "accuracy_per_seed_pp": [
      15.686274509803921,
      14.117647058823524,
      12.549019607843137
    ],
    "macro_f1_mean_delta_pp": 14.47118678664238,
    "pass": true
  },
  "tam_only - packet_direction": {
    "accuracy_mean_delta_pp": 16.928104575163395,
    "accuracy_per_seed_pp": [
      18.039215686274503,
      14.117647058823524,
      18.627450980392158
    ],
    "macro_f1_mean_delta_pp": 17.41728480626981,
    "pass": true
  },
  "packet_native - fusion": {
    "accuracy_mean_delta_pp": -3.66013071895425,
    "accuracy_per_seed_pp": [
      -3.7254901960784292,
      -1.3725490196078494,
      -5.882352941176472
    ],
    "macro_f1_mean_delta_pp": -3.6929310703820586,
    "pass": false
  },
  "tam_only - fusion": {
    "accuracy_mean_delta_pp": -0.8496732026143835,
    "accuracy_per_seed_pp": [
      -1.3725490196078494,
      -1.3725490196078494,
      0.19607843137254832
    ],
    "macro_f1_mean_delta_pp": -0.7468330507546288,
    "pass": false
  }
}
```

packet_direction仅方向/顺序/观测，时间幅值通道零；packet_native保留原packet逐包时间；tam_only为按方向分通道的时间计数而非不含方向的纯时间。融合是同版已完成seed，不是新增重复。
所有新训练从头初始化，GPU预算、步数、样本顺序、20次accuracy选模与原融合匹配。移除视角时冻结独立参数并屏蔽attention/readout；有效信息、参数参与容量及计算仍不同，不能把差值唯一归因于信息本身。共享分类器的未使用半侧及view embedding行没有有效信号。
固定510 valid已经参与多轮开发，三seed不是三个独立数据集。无未来日期、WTT/AWF、TTA或漂移调整，源期90%即便达成也不是独立泛化或抗漂移确认。
packet_native−packet_direction诊断当前接法中逐包时间幅值的增量；fusion−packet_native诊断加入TAM整体增量；fusion−tam_only诊断加入原packet整体增量。模型参数/表示可用自由度随消融改变。
复用来源见SOURCES.md、manifest.json、artifacts/historical_audit.json；冻结代码/数据/配置hash见artifacts/freeze.json。阴性与全部seed均报告，未自动追加训练或改法。

## 完成后复核与下一候选

用户告知完成后，重新检查55份冻结文件hash并独立复算全部48组预测指标，均通过；9项新任务均完成12800步、20次valid选模，3项fusion明确为旧结果复用。诊断见artifacts/postrun_audit.json，未重新训练或访问未来/外部数据。

三个主要增量：packet_native−packet_direction +14.118pp，3/3seed正，过门槛；fusion−packet_native +3.660pp，3/3seed正，过门槛；fusion−tam_only +0.850pp，逐seed+1.373/+1.373/−0.196pp，仅2/3正且均值未达1pp，未过门槛。不能称双视角稳定超过两个单视角，也不能反过来声称packet完全无用或TAM与融合统计等价。

相对TAM单视角，fusion每seed平均改对66.33条、同时改错62.00条（固定valid510），净增只有4.33条。该错误转移说明本接法收益与损失同时存在；它是事后描述，不能用于query真标签路由或证明具体优化原因。三seed共同错TAM57条、fusion60条，不能只据共同错误数决定单模型优劣或部署集成。

纯方向本版best valid56.405%、last52.484%，同时last source97.632%；TAM best valid73.333%、last71.830%，last source99.802%。训练末期较高拟合没有转成更高valid，不建议仅追加训练步数；不能唯一归因于生成器过拟合或不可逆压缩。历史方向CNN+MLP+mask73.987%表明方向输入在其他编码/训练配方下仍有判别力，56.405%仅是本版接法结果，不是方向信息的性能上限。

保留“自有生成器→分类模型→后续漂移调整”的总体接口。下一优先候选为在TAM单视角、同Transformer及同输入/训练/选模预算下，对比当前局部编码与保序多尺度局部编码，控制token/读出接口，记录参数与有效容量差异；它是待验证设计，不宣称已经定位编码为唯一瓶颈。当前双视角融合保留作开发性能参考，不先堆更多融合分支。本次未实现或启动此新实验。

最佳平均74.183%，90%目标仍差15.817pp；当前时间输入在源期有增益不证明它在时间漂移下稳定，亦不能直接推广到没有逐包时间戳的WTT/AWF。漂移机制仍需另立权限及评价对照，不能把此源期消融当抗漂移结论。
