# 网站指纹生成器跨分类头结果

12项新训练+3项历史基线完成；shallow_transformer=77.647%/shallow_mlp=71.373%/progressive_transformer=86.536%/progressive_mlp=84.641%/rf_matched=91.699%；通过候选['progressive_transformer', 'progressive_mlp', 'rf_matched']；60组预测指标和冻结hash通过。

|条件|accuracy三seed均值|Macro-F1|≥90%|
|---|---:|---:|---|
|shallow_transformer|77.647%|76.889%|False|
|shallow_mlp|71.373%|70.491%|False|
|progressive_transformer|86.536%|86.169%|False|
|progressive_mlp|84.641%|84.277%|False|
|rf_matched|91.699%|91.588%|True|

|条件|seed|best步数|source accuracy|valid accuracy|valid Macro-F1|秒|
|---|---:|---:|---:|---:|---:|---:|
|shallow_transformer|21729|7520|97.131%|77.647%|76.830%|358.3|
|shallow_mlp|21729|6560|92.712%|71.373%|70.623%|308.3|
|progressive_transformer|21729|7040|99.222%|87.647%|87.302%|360.0|
|progressive_mlp|21729|12320|100.000%|84.314%|83.872%|335.5|
|rf_matched|21729|7520|99.039%|91.373%|91.232%|353.9|
|shallow_transformer|23407|7520|96.157%|78.235%|77.384%|355.6|
|shallow_mlp|23407|10400|97.941%|71.765%|70.915%|318.0|
|progressive_transformer|23407|12800|99.993%|85.686%|85.371%|374.7|
|progressive_mlp|23407|12320|100.000%|84.510%|84.213%|333.4|
|rf_matched|23407|8000|99.144%|91.765%|91.672%|336.0|
|shallow_transformer|22026|11840|99.033%|77.059%|76.452%|343.8|
|shallow_mlp|22026|10400|98.595%|70.980%|69.936%|305.0|
|progressive_transformer|22026|7520|99.673%|86.275%|85.834%|382.8|
|progressive_mlp|22026|10880|100.000%|85.098%|84.746%|375.0|
|rf_matched|22026|12800|99.824%|91.961%|91.861%|348.8|

```json
{
  "comparisons": {
    "shallow_mlp - shallow_transformer": {
      "accuracy_mean_delta_pp": -6.2745098039215685,
      "accuracy_per_seed_pp": [
        -6.2745098039215685,
        -6.470588235294117,
        -6.07843137254902
      ],
      "macro_f1_mean_delta_pp": -6.3975186034009575,
      "pass": false
    },
    "progressive_transformer - shallow_transformer": {
      "accuracy_mean_delta_pp": 8.888888888888886,
      "accuracy_per_seed_pp": [
        9.999999999999998,
        7.4509803921568585,
        9.215686274509805
      ],
      "macro_f1_mean_delta_pp": 9.280387368622666,
      "pass": true
    },
    "progressive_mlp - shallow_mlp": {
      "accuracy_mean_delta_pp": 13.267973856209148,
      "accuracy_per_seed_pp": [
        12.941176470588234,
        12.745098039215685,
        14.117647058823524
      ],
      "macro_f1_mean_delta_pp": 13.785967119300457,
      "pass": true
    },
    "progressive_mlp - shallow_transformer": {
      "accuracy_mean_delta_pp": 6.993464052287579,
      "accuracy_per_seed_pp": [
        6.666666666666665,
        6.2745098039215685,
        8.039215686274503
      ],
      "macro_f1_mean_delta_pp": 7.3884485158994995,
      "pass": true
    },
    "progressive_transformer - progressive_mlp": {
      "accuracy_mean_delta_pp": 1.8954248366013078,
      "accuracy_per_seed_pp": [
        3.3333333333333326,
        1.17647058823529,
        1.176470588235301
      ],
      "macro_f1_mean_delta_pp": 1.8919388527231666,
      "pass": true
    },
    "rf_matched - shallow_transformer": {
      "accuracy_mean_delta_pp": 14.052287581699341,
      "accuracy_per_seed_pp": [
        13.725490196078427,
        13.529411764705879,
        14.901960784313717
      ],
      "macro_f1_mean_delta_pp": 14.699369258192787,
      "pass": true
    },
    "rf_matched - progressive_transformer": {
      "accuracy_mean_delta_pp": 5.163398692810453,
      "accuracy_per_seed_pp": [
        3.7254901960784292,
        6.07843137254902,
        5.686274509803912
      ],
      "macro_f1_mean_delta_pp": 5.418981889570121,
      "pass": true
    }
  },
  "interaction": {
    "accuracy": {
      "mean_pp": -4.37908496732026,
      "per_seed_pp": [
        -2.941176470588236,
        -5.294117647058827,
        -4.901960784313719
      ]
    },
    "macro_f1": {
      "mean_pp": -4.505579750677791,
      "per_seed_pp": [
        -2.7767058649411425,
        -5.311953841365613,
        -5.428079545726616
      ]
    }
  },
  "cross_head_gate": true
}
```

四组合分别联合训练，只检验生成器结构跨分类头适配，非冻结权重迁移/缺时间适配/跨数据集通用或抗漂移。RF为官方结构同配方对照，原生85.948%仅不同配方历史参考。参数与计算不匹配，不能唯一归因某单一结构因素。详见PLAN.md/SOURCES.md。

## 完成后独立审计与研究解释

12新任务+3历史基线完成，整批1446.96秒（约24.1分钟）。独立postrun_audit.py核验48冻结hash、60组预测指标、完整logits与argmax一致、checkpoint配置hash/步数、共享模块初始化、索引、全部20次选模及参数/梯度检查通过。完成后审计脚本新增，不改变冻结训练文件。12/15末步valid低于best，progressive_transformer23407/22026与rf_matched22026末步持平（22026 Transformer仍取更早7520步），不能把持平误写为末步首次最佳。

主要结论：新逐级生成器在Transformer头上+8.889pp/F1+9.280pp、3/3正，在MLP头上+13.268pp/F1+13.786pp、3/3正，预定跨头结构适配门槛通过。progressive_transformer三seed87.647/85.686/86.275%，均值86.536/F1 86.169%；在新生成器下Transformer对MLP+1.895pp、3/3正且F1不降，支持保留新生成器+Transformer为当前自有框架开发基线。参数/通道/层级/残差改变是整体结构包，不把全部收益唯一归因于任意单个因素；生成器1472→180336可训练参数，整体322726→501590。

RF同配方91.699/F1 91.588%，三seed91.373/91.765/91.961均超过90；比新自有模型+5.163pp/3seed均正，比原生历史85.948高5.752pp（输入缩放、batch、样本暴露数、优化器、遮挡/日程等配方包变化，非单因果归因）。这说明固定源期协议下至少有成熟对照能够超过90%，并非证明自有生成器达到目标或独立未来泛化达标。自有模型距90%为3.464pp。RF参数1040562 vs新模型501590(trainable)，相同步数/样本权限不等于参数/计算相同。

自动summary preferred_candidate=rf_matched表示所有参比中的数值排序；研究层面将progressive_transformer作为自有框架当前候选、rf_matched作为强对照分别保留，不能把官方RF改称自有通用生成器。旧77.647历史基线仍保留比较证据。

通用性边界：目前证据为同一TAM/source-valid数据下、分别联合训练的生成器结构跨两个头有效；不代表同一组已训练生成器权重可跨头复用，不代表方向-only或缺时间可用，更不证明跨数据集/抗漂移。三seed是优化重复，valid已多次用于方法开发。

描述性错例：原基线三seed共同错53，matched RF三seed全对其中29；新Transformer共同错23，RF三seed全对其中7；新MLP共同错48，RF全对其中21。交集受种子相关性影响，不能由共同错数单独排序模型性能，无oracle路由或集成成绩。

## 下一建议（未启动）

围绕用户“网站指纹通用生成器”目标，下一优先验证权重复用：冻结本轮progressive_transformer学得的generator，重置并训练MLP分类部分（含投影/位置/最终分类层），与冻结progressive_mlp自身generator后重新训练相同MLP、以及冻结同架构随机generator后训练MLP比较；每组3seed，共9新任务。所有分类部分相同初始化、source索引、训练和选模预算，生成器冻结到参数级核验，所有旧表示/预训练source预算和valid选模历史单列。前两组具有相同预训练数据及训练预算；随机组只有头训练预算，为学习表示诊断参考，不称同总算力。旧全模型结果只作联合训练参照，不直接把差值解释为冻结单因果。该实验可以检验Transformer学得的生成器权重供另一种分类头使用是否仍有价值；没有新数据，仍是开发而非独立确认。

90%准确率线继续以86.536为基线、91.699为强参照，先做新生成器容量/归一化等具体差异审计再冻结单因素，不自动混入冻结复用实验或再次无界扫正则。当前证据无需立即放弃Transformer。跨数据集与无时间输入支持须另立方法版本及冻结输入/权限，未来保持关闭。
