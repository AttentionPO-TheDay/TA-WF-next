# 梯度未见纠错元监督检验 v16

主臂GU_all机理门槛：False；超越v12方法门槛：False。模型和候选固定；不因其他臂更好替换主臂。

## 完成后的解释补充

本段在全量预注册训练/评价和终端核验完成后增补，仅修订解释；原报告保留，所有模型、分数、阈值、预测及判决不变。

1. 等原始标签预算612行下，GU_all相对IM_all的query低风险pAUC在Version/USA分别提高+0.207/+0.053pp，但同IM实际D预算Recovery分别−0.0032/0.0000pp；Version只有1/3seed不下降。主机理门槛未通过，不能把pAUC微升等同为纠错能力提升。
2. 相对共同v12对照，同实际D预算Recovery为Version+0.0440pp、USA−0.0757pp；主方法升级门槛同样未通过。GU_aug和GU_clean均预先声明，但不替换主臂，未据目标结果返调。
3. clean监督构成确有改变：IM在三seed均100%正确、零改写；GU clean准确率87.42%–88.56%，产生R/D/W。但每个切片seed只有3–10个自然R，GU_clean虽然满足预定>=3的可行性规则，仍是极小样本诊断，不能由其源诊断pAUC更高推断稳定目标收益。GU_all/GU_aug对照未显示加入自然clean错误能够稳定改善实际风险预算下的目标纠错。
4. IM_all与IM_aug完全一致，原因是本轮固定IM源子集clean没有改写，clean不进入changed分类器拟合；不是该消融未执行。GU对应消融确实改变了训练集合和模型。
5. 当前结果支持“单纯把元监督换成梯度未见源样本，不足以解决本配方的低风险迁移瓶颈”。不能据此证明记忆不是因素，也不能证伪严格骨干交叉拟合或更大模型外错误池；本轮官方valid有历史checkpoint/温度选择暴露，资格始终仅gradient-unseen。
6. 本轮510行校准角色不同于v14/v15，所有新臂和v12均在同一角色按同一效用重校准。源阈值直接迁移的操作点只能作本轮成对比较，不能把跨run校准样本改变造成的数值变化冒充方法收益；诊断前沿与部署阈值仍严格分开。

30个重排器、15新源缓存及各首256行重提取、96个source-diagnostic/query状态、24个描述量构造状态、58个有限缓存及角色内容隔离核验通过。保留v12，不部署、不扩片、不启动严格交叉拟合或新骨干训练。

## 资格与角色

官方valid未参与骨干/辅助编码器梯度，但参与过历史checkpoint/温度选择；本轮是gradient-unseen检验，不是严格模型外交叉拟合或新确认。已观察valid现在显式分为元监督612/校准510/源诊断612，内容组互斥；元训练与校准不得再称独立验证。IM元监督使用612既有源train行，各主要臂标签预算匹配。

## query低NAR诊断前沿

三seed均值；缺seed均值NA。pAUC/.01为FPR<=.01区间平均候选TPR，FPR分母全部frozen正确者。query标签选出的前沿不可部署。

|切片|方法|状态|pAUC/.01 %|Recovery@NAR1 %|W|W/R|
|---|---|---:|---:|---:|---:|---:|
|version_045|v9_margin|3|42.846|7.515|914.333|1.077|
|version_045|v12_primary|3|52.331|7.866|1000.667|1.128|
|version_045|v14_primary|3|52.404|7.949|979.333|1.094|
|version_045|IM_all|3|52.333|7.919|974.333|1.093|
|version_045|IM_aug|3|52.333|7.919|974.333|1.093|
|version_045|GU_all|3|52.540|7.919|975.000|1.092|
|version_045|GU_aug|3|52.550|7.934|980.000|1.097|
|version_045|GU_clean|3|52.112|7.919|943.667|1.056|
|network_USA|v9_margin|3|40.561|5.195|102.667|1.567|
|network_USA|v12_primary|3|60.123|6.254|118.667|1.532|
|network_USA|v14_primary|3|59.697|6.310|119.667|1.526|
|network_USA|IM_all|3|59.684|6.285|119.667|1.530|
|network_USA|IM_aug|3|59.684|6.285|119.667|1.530|
|network_USA|GU_all|3|59.737|6.285|119.667|1.530|
|network_USA|GU_aug|3|59.786|6.285|119.667|1.530|
|network_USA|GU_clean|3|59.770|6.206|117.333|1.519|

## 匹配实际损伤预算

|切片|对照预算|方法|Recovery增量pp|不下降seed|
|---|---|---|---:|---:|
|version_045|IM_all|IM_all|0.000|3|
|version_045|IM_all|IM_aug|0.000|3|
|version_045|IM_all|GU_all|-0.003|1|
|version_045|IM_all|GU_aug|0.006|2|
|version_045|IM_all|GU_clean|-0.012|2|
|version_045|v12_primary|IM_all|0.020|2|
|version_045|v12_primary|IM_aug|0.020|2|
|version_045|v12_primary|GU_all|0.044|2|
|version_045|v12_primary|GU_aug|0.023|2|
|version_045|v12_primary|GU_clean|0.044|2|
|network_USA|IM_all|IM_all|0.000|3|
|network_USA|IM_all|IM_aug|0.000|3|
|network_USA|IM_all|GU_all|0.000|3|
|network_USA|IM_all|GU_aug|0.000|3|
|network_USA|IM_all|GU_clean|-0.078|1|
|network_USA|v12_primary|IM_all|-0.102|0|
|network_USA|v12_primary|IM_aug|-0.102|0|
|network_USA|v12_primary|GU_all|-0.076|1|
|network_USA|v12_primary|GU_aug|-0.102|0|
|network_USA|v12_primary|GU_clean|-0.181|0|

## 冻结源阈值直接迁移

以下仅源calibration阈值，未用query调阈值；满足观察NAR并非风险保证。

|切片|方法|状态|NAR均值 %|NAR最大 %|Recovery %|W|
|---|---|---:|---:|---:|---:|---:|
|version_045|v9_margin|3|0.259|0.778|1.519|128.667|
|version_045|v12_primary|3|0.385|0.801|4.743|468.333|
|version_045|v14_primary|3|0.320|0.616|4.632|428.000|
|version_045|IM_all|3|0.400|0.832|4.973|478.667|
|version_045|IM_aug|3|0.400|0.832|4.973|478.667|
|version_045|GU_all|3|0.336|0.609|4.726|446.667|
|version_045|GU_aug|3|0.318|0.585|4.509|421.000|
|version_045|GU_clean|3|0.309|0.663|4.260|364.667|
|network_USA|v9_margin|3|0.046|0.137|0.909|10.667|
|network_USA|v12_primary|3|0.158|0.343|2.675|37.333|
|network_USA|v14_primary|3|0.182|0.343|2.613|34.333|
|network_USA|IM_all|3|0.182|0.343|2.588|34.000|
|network_USA|IM_aug|3|0.182|0.343|2.588|34.000|
|network_USA|GU_all|3|0.225|0.274|2.731|38.333|
|network_USA|GU_aug|3|0.203|0.274|2.731|38.333|
|network_USA|GU_clean|3|0.203|0.274|2.783|39.000|

## 未用于选模的源诊断

|切片|方法|状态|pAUC/.01 %|Recovery@NAR1 %|
|---|---|---:|---:|---:|
|version_045|v9_margin|3|9.425|0.809|
|version_045|v12_primary|3|37.328|2.646|
|version_045|v14_primary|3|42.132|3.042|
|version_045|IM_all|3|42.564|3.042|
|version_045|IM_aug|3|42.564|3.042|
|version_045|GU_all|3|42.144|3.042|
|version_045|GU_aug|3|42.284|3.042|
|version_045|GU_clean|3|44.388|3.152|
|network_USA|v9_margin|3|12.062|1.160|
|network_USA|v12_primary|3|40.794|2.856|
|network_USA|v14_primary|3|44.726|3.197|
|network_USA|IM_all|3|44.041|3.197|
|network_USA|IM_aug|3|44.041|3.197|
|network_USA|GU_all|3|44.339|3.197|
|network_USA|GU_aug|3|44.475|3.197|
|network_USA|GU_clean|3|45.374|3.170|

## 可行性

```json
[
  {
    "seed": 42,
    "slice": "version_045",
    "arm": "IM_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 40,
    "D": 104,
    "W": 206
  },
  {
    "seed": 42,
    "slice": "version_045",
    "arm": "IM_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 40,
    "D": 104,
    "W": 206
  },
  {
    "seed": 42,
    "slice": "version_045",
    "arm": "GU_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 53,
    "D": 101,
    "W": 219
  },
  {
    "seed": 42,
    "slice": "version_045",
    "arm": "GU_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 48,
    "D": 90,
    "W": 209
  },
  {
    "seed": 42,
    "slice": "version_045",
    "arm": "GU_clean",
    "feasible": true,
    "min_each_type": 3,
    "raw_label_budget": 612,
    "R": 5,
    "D": 11,
    "W": 10
  },
  {
    "seed": 42,
    "slice": "network_USA",
    "arm": "IM_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 37,
    "D": 74,
    "W": 137
  },
  {
    "seed": 42,
    "slice": "network_USA",
    "arm": "IM_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 37,
    "D": 74,
    "W": 137
  },
  {
    "seed": 42,
    "slice": "network_USA",
    "arm": "GU_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 54,
    "D": 65,
    "W": 163
  },
  {
    "seed": 42,
    "slice": "network_USA",
    "arm": "GU_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 46,
    "D": 55,
    "W": 157
  },
  {
    "seed": 42,
    "slice": "network_USA",
    "arm": "GU_clean",
    "feasible": true,
    "min_each_type": 3,
    "raw_label_budget": 612,
    "R": 8,
    "D": 10,
    "W": 6
  },
  {
    "seed": 100,
    "slice": "version_045",
    "arm": "IM_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 63,
    "D": 72,
    "W": 164
  },
  {
    "seed": 100,
    "slice": "version_045",
    "arm": "IM_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 63,
    "D": 72,
    "W": 164
  },
  {
    "seed": 100,
    "slice": "version_045",
    "arm": "GU_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 66,
    "D": 65,
    "W": 162
  },
  {
    "seed": 100,
    "slice": "version_045",
    "arm": "GU_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 56,
    "D": 59,
    "W": 156
  },
  {
    "seed": 100,
    "slice": "version_045",
    "arm": "GU_clean",
    "feasible": true,
    "min_each_type": 3,
    "raw_label_budget": 612,
    "R": 10,
    "D": 6,
    "W": 6
  },
  {
    "seed": 100,
    "slice": "network_USA",
    "arm": "IM_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 68,
    "D": 62,
    "W": 124
  },
  {
    "seed": 100,
    "slice": "network_USA",
    "arm": "IM_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 68,
    "D": 62,
    "W": 124
  },
  {
    "seed": 100,
    "slice": "network_USA",
    "arm": "GU_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 62,
    "D": 56,
    "W": 160
  },
  {
    "seed": 100,
    "slice": "network_USA",
    "arm": "GU_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 59,
    "D": 52,
    "W": 150
  },
  {
    "seed": 100,
    "slice": "network_USA",
    "arm": "GU_clean",
    "feasible": true,
    "min_each_type": 3,
    "raw_label_budget": 612,
    "R": 3,
    "D": 4,
    "W": 10
  },
  {
    "seed": 2026,
    "slice": "version_045",
    "arm": "IM_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 46,
    "D": 75,
    "W": 133
  },
  {
    "seed": 2026,
    "slice": "version_045",
    "arm": "IM_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 46,
    "D": 75,
    "W": 133
  },
  {
    "seed": 2026,
    "slice": "version_045",
    "arm": "GU_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 47,
    "D": 94,
    "W": 160
  },
  {
    "seed": 2026,
    "slice": "version_045",
    "arm": "GU_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 43,
    "D": 84,
    "W": 157
  },
  {
    "seed": 2026,
    "slice": "version_045",
    "arm": "GU_clean",
    "feasible": true,
    "min_each_type": 3,
    "raw_label_budget": 612,
    "R": 4,
    "D": 10,
    "W": 3
  },
  {
    "seed": 2026,
    "slice": "network_USA",
    "arm": "IM_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 44,
    "D": 64,
    "W": 107
  },
  {
    "seed": 2026,
    "slice": "network_USA",
    "arm": "IM_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 44,
    "D": 64,
    "W": 107
  },
  {
    "seed": 2026,
    "slice": "network_USA",
    "arm": "GU_all",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 38,
    "D": 72,
    "W": 141
  },
  {
    "seed": 2026,
    "slice": "network_USA",
    "arm": "GU_aug",
    "feasible": true,
    "min_each_type": 10,
    "raw_label_budget": 612,
    "R": 33,
    "D": 65,
    "W": 136
  },
  {
    "seed": 2026,
    "slice": "network_USA",
    "arm": "GU_clean",
    "feasible": true,
    "min_each_type": 3,
    "raw_label_budget": 612,
    "R": 5,
    "D": 7,
    "W": 5
  }
]
```

## 预注册判决

```json
{
  "primary": "GU_all",
  "mechanism_passed": false,
  "promotion_passed": false,
  "slices": [
    {
      "slice": "version_045",
      "complete": true,
      "IM_all_same_damage_rec_gain_pp": -0.0031844361937136134,
      "IM_all_nondecreasing_seeds": 1,
      "v12_primary_same_damage_rec_gain_pp": 0.043994162801137826,
      "v12_primary_nondecreasing_seeds": 2,
      "pauc_gain_vs_IM_pp": 0.20697924098600495,
      "mechanism_passed": false,
      "same_v12_R_feasible": false,
      "same_v12_R_W_ratio": null,
      "same_v12_R_D_change": null,
      "v12_pauc_gain_pp": 0.20961326127503233,
      "v12_rec1_gain_pp": 0.052835950033294776,
      "promotion_passed": false
    },
    {
      "slice": "network_USA",
      "complete": true,
      "IM_all_same_damage_rec_gain_pp": 0.0,
      "IM_all_nondecreasing_seeds": 3,
      "v12_primary_same_damage_rec_gain_pp": -0.07571937223979502,
      "v12_primary_nondecreasing_seeds": 1,
      "pauc_gain_vs_IM_pp": 0.053084295890681865,
      "mechanism_passed": false,
      "same_v12_R_feasible": false,
      "same_v12_R_W_ratio": null,
      "same_v12_R_D_change": null,
      "v12_pauc_gain_pp": -0.38552422356236127,
      "v12_rec1_gain_pp": 0.03035328753214772,
      "promotion_passed": false
    }
  ],
  "gradient_unseen_only": true,
  "strict_out_of_model": false,
  "integrated": false
}
```

## 审核与限制

preflight三个源模型/两个固定LoRA/辅助原始输出与历史完整2160valid逐位复现。每个新缓存前256行独立重提取，全部模型参数与BN缓冲冻结；所有拟合模型/source/query分数与阈值及计数复算，散列冻结清单见artifacts。原始数据只读，源训练清单未扩大，无骨干/LoRA/辅助训练。

GU与IM仍可能在错误构成/类别难度/先前选模影响上不同。原clean与合成的消融不能独立消除这些因素；不能以此证明记忆的因果作用。源诊断不是严格新确认，历史标签暴露保留。清洁源错误数量可能过少，GU_clean不足必须记不可行而非扩样。重复扰动不是独立样本，三seed共用query，微小增幅不能作显著性结论。

Version045/USA仅已观察开发，沿用内容支持102类身份假设，非官方映射；不能概括所有时间漂移场景。WTT/AWF关闭，未部署，未自动新增交叉拟合训练。
