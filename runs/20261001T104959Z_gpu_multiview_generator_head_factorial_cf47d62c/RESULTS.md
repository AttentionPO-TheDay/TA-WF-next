# 多视角生成器×分类器结果

12任务完成；valid accuracy fixed_mlp=63.203%/fixed_transformer=67.582%/learned_mlp=66.797%/learned_transformer=74.183%；48份预测独立指标核验、24份best预测重载通过。

保留当前可学习生成器+Transformer为优先候选；仍需独立设计漂移阶段与对照。

|条件|seed|best步数|source accuracy|valid accuracy|valid Macro-F1|
|---|---:|---:|---:|---:|---:|
|fixed_mlp|21729|9920|99.915%|63.137%|62.419%|
|fixed_transformer|21729|8000|99.098%|67.647%|66.114%|
|learned_mlp|21729|10880|99.980%|66.667%|65.392%|
|learned_transformer|21729|11840|100.000%|74.314%|73.615%|
|fixed_mlp|23407|7040|98.699%|61.765%|61.219%|
|fixed_transformer|23407|8000|99.157%|67.255%|66.566%|
|learned_mlp|23407|7040|99.261%|67.843%|67.193%|
|learned_transformer|23407|7520|99.333%|73.725%|72.696%|
|fixed_mlp|22026|6080|92.562%|64.706%|63.270%|
|fixed_transformer|22026|8480|99.765%|67.843%|67.294%|
|learned_mlp|22026|8960|99.922%|65.882%|65.113%|
|learned_transformer|22026|12800|99.993%|74.510%|73.209%|

三seed均值：

|条件|accuracy|Macro-F1|≥90%|
|---|---:|---:|---|
|fixed_mlp|63.203%|62.303%|False|
|fixed_transformer|67.582%|66.658%|False|
|learned_mlp|66.797%|65.899%|False|
|learned_transformer|74.183%|73.173%|False|

配对增量门槛：accuracy均值≥1pp、3/3seed为正、平均Macro-F1不下降。交互只报告数值，不宣称统计确认。

```json
{
  "comparisons": {
    "learned_mlp - fixed_mlp": {
      "accuracy_mean_delta_pp": 3.5947712418300637,
      "accuracy_per_seed_pp": [
        3.529411764705881,
        6.07843137254902,
        1.17647058823529
      ],
      "macro_f1_mean_delta_pp": 3.5967550327411946,
      "pass": true
    },
    "learned_transformer - fixed_transformer": {
      "accuracy_mean_delta_pp": 6.601307189542482,
      "accuracy_per_seed_pp": [
        6.666666666666665,
        6.470588235294117,
        6.666666666666665
      ],
      "macro_f1_mean_delta_pp": 6.515007163796101,
      "pass": true
    },
    "fixed_transformer - fixed_mlp": {
      "accuracy_mean_delta_pp": 4.3790849673202645,
      "accuracy_per_seed_pp": [
        4.509803921568634,
        5.490196078431375,
        3.1372549019607843
      ],
      "macro_f1_mean_delta_pp": 4.355628672237666,
      "pass": true
    },
    "learned_transformer - learned_mlp": {
      "accuracy_mean_delta_pp": 7.385620915032683,
      "accuracy_per_seed_pp": [
        7.647058823529418,
        5.882352941176472,
        8.62745098039216
      ],
      "macro_f1_mean_delta_pp": 7.273880803292572,
      "pass": true
    },
    "learned_transformer - fixed_mlp": {
      "accuracy_mean_delta_pp": 10.980392156862747,
      "accuracy_per_seed_pp": [
        11.176470588235299,
        11.960784313725492,
        9.80392156862745
      ],
      "macro_f1_mean_delta_pp": 10.870635836033767,
      "pass": true
    }
  },
  "interaction": {
    "accuracy": {
      "mean_pp": 3.0065359477124187,
      "per_seed_pp": [
        3.1372549019607843,
        0.39215686274509665,
        5.490196078431375
      ]
    },
    "macro_f1": {
      "mean_pp": 2.9182521310549063,
      "per_seed_pp": [
        4.5279198566395955,
        0.15521515521517149,
        4.071621381309953
      ]
    }
  }
}
```

历史RF 85.948%、方向CNN+MLP+mask 73.987%仅为已观察性能参照，不是本轮相同输入表示、训练配方及预算的归因对照。
本轮固定/可学习共享统计及mask；可学习条件额外80维CNN特征改变有效特征自由度。新增特征整体增益不能单独归因于学习本身。此模型是自有系统的新版本，不是旧版原封不动或官方RF生成器移植；独立实现不等于研究新颖性。
仅source/valid监督训练及预定选模；固定510 valid已多轮开发，三seed不是三个独立验证集。无未来/WTT/AWF评分、预训练、TTA或漂移调整。当前时间输入版本不能自动声称适用于缺时间戳的数据集。
代码、配置和数据hash见artifacts/freeze.json及manifest.json；本轮结果不自动授权额外训练。

## 完成后核验与框架判断

本次用户告知完成后，重新核对冻结文件hash，并独立复算全部48组best/last×source/valid预测指标，均通过。未重新训练、未增加数据角色；诊断记录见artifacts/postrun_audit.json。

最佳可学习生成器+Transformer三seed source accuracy均值99.776%，valid74.183%，last valid73.464%；共同错60/510条。best steps为11840/7520/12800，不能仅凭个别seed末步最好推断追加训练会稳定提升。训练/验证差距25.593pp提示需检查泛化，不能唯一归因于生成器过拟合或表示压缩。

本轮生成器在Transformer侧+6.601pp、Transformer在可学习生成器侧+7.386pp，两个增量3/3seed正且过预定门槛。因此保留组合为当前架构候选有依据。但距离90%还差15.817pp，低于历史RF85.948%约11.765pp；比历史方向CNN+MLP+mask73.987%仅高0.196pp，配方/输入不同，不能据此证明新系统已有整体性能突破。原自动结论“保留优先候选”指本轮因素比较，不意味着基础模型达标或整套实现无需改变。

建议下一轮首先在本版相同训练与选模预算下分离packet-only、TAM-only、融合的贡献，定位融合是否带来增量。生成器时间编码的保序/多尺度设计是后续候选，尚未验证为当前瓶颈；不要同时更换编码器、分类头、增强与训练配方后作单因素归因。保留“自有生成器→分类模型→后续漂移调整”的总体接口，局部编码与汇聚可按证据改动。本次只分析完成结果，未启动后续训练或漂移评价。
