# TAM输入×优化配方结果

9项新训练+3项mask历史复用完成；log_current=77.647%/raw_current=74.118%/log_rfstyle=70.196%/raw_rfstyle=69.542%；通过候选[]；48组预测指标和冻结hash通过。

|条件|accuracy三seed均值|Macro-F1|≥90%|
|---|---:|---:|---|
|log_current|77.647%|76.889%|False|
|raw_current|74.118%|73.248%|False|
|log_rfstyle|70.196%|69.289%|False|
|raw_rfstyle|69.542%|68.723%|False|

|条件|seed|best步数|source accuracy|valid accuracy|valid Macro-F1|秒|
|---|---:|---:|---:|---:|---:|---:|
|log_current|21729|7520|97.131%|77.647%|76.830%|358.3|
|raw_current|21729|6560|96.863%|73.922%|73.355%|374.4|
|log_rfstyle|21729|12800|92.575%|70.000%|69.320%|374.2|
|raw_rfstyle|21729|10400|91.732%|70.784%|70.350%|373.8|
|log_current|23407|7520|96.157%|78.235%|77.384%|355.6|
|raw_current|23407|8480|98.131%|74.314%|73.609%|362.4|
|log_rfstyle|23407|8480|85.706%|70.588%|69.831%|366.8|
|raw_rfstyle|23407|9920|87.941%|68.039%|67.002%|366.7|
|log_current|22026|11840|99.033%|77.059%|76.452%|343.8|
|raw_current|22026|9920|99.209%|74.118%|72.781%|360.3|
|log_rfstyle|22026|12320|91.052%|70.000%|68.717%|364.9|
|raw_rfstyle|22026|10400|90.804%|69.804%|68.816%|365.1|

```json
{
  "comparisons": {
    "raw_current - log_current": {
      "accuracy_mean_delta_pp": -3.529411764705881,
      "accuracy_per_seed_pp": [
        -3.7254901960784292,
        -3.9215686274509776,
        -2.941176470588236
      ],
      "macro_f1_mean_delta_pp": -3.6405805523452552,
      "pass": false
    },
    "log_rfstyle - log_current": {
      "accuracy_mean_delta_pp": -7.450980392156866,
      "accuracy_per_seed_pp": [
        -7.647058823529418,
        -7.647058823529407,
        -7.058823529411773
      ],
      "macro_f1_mean_delta_pp": -7.599481128892882,
      "pass": false
    },
    "raw_rfstyle - raw_current": {
      "accuracy_mean_delta_pp": -4.575163398692813,
      "accuracy_per_seed_pp": [
        -3.1372549019607843,
        -6.2745098039215685,
        -4.313725490196085
      ],
      "macro_f1_mean_delta_pp": -4.5256337067179055,
      "pass": false
    },
    "raw_rfstyle - log_rfstyle": {
      "accuracy_mean_delta_pp": -0.6535947712418277,
      "accuracy_per_seed_pp": [
        0.7843137254902044,
        -2.5490196078431393,
        -0.19607843137254832
      ],
      "macro_f1_mean_delta_pp": -0.5667331301702789,
      "pass": false
    },
    "raw_rfstyle - log_current": {
      "accuracy_mean_delta_pp": -8.104575163398692,
      "accuracy_per_seed_pp": [
        -6.8627450980392135,
        -10.196078431372547,
        -7.254901960784322
      ],
      "macro_f1_mean_delta_pp": -8.16621425906316,
      "pass": false
    }
  },
  "interaction": {
    "accuracy": {
      "mean_pp": 2.875816993464053,
      "per_seed_pp": [
        4.509803921568634,
        1.3725490196078383,
        2.7450980392156876
      ]
    },
    "macro_f1": {
      "mean_pp": 3.073847422174976,
      "per_seed_pp": [
        4.504732740026817,
        0.9462119323710771,
        3.770597594127034
      ]
    }
  }
}
```

所有条件同一自有生成器＋Transformer＋固定span_mask，原始TAM行/标签、初始state、索引、步数和20次选模相同。raw/log1p改变幅值映射，RF启发配方同时改变optimizer、LR、权重衰减及日程，不能唯一归因于某项或称官方RF完整复现。
baseline3seed明确历史复用，不是新增独立重复；新模型从头初始化，无teacher/旧checkpoint训练。固定valid反复开发，三seed是优化重复，不证明外部泛化/抗漂移。交互仅描述数值；不自动追加参数扫描或合并未通过模块，源期90%目标独立判断。
审计与冻结规则见PLAN.md/AUDIT.md/SOURCES.md，完整产物在artifacts/checkpoints/logs。无未来日期、WTT/AWF或适应。

## 完成后独立复核与解释（2026-10-02）

9项新训练全部完成，3项mask基线明确历史复用，整批耗时1152.79秒（约19.2分钟）。53个冻结文件hash、12报告best/last×source/valid共48组accuracy与Macro-F1独立重算一致；初始化state、索引流、12800步/20次验证和最早最大accuracy选模核验通过。checkpoint重载预测已由worker/preflight核验，本次仅读取既有缓存/预测，无新训练或未来访问。证据artifacts/postrun_audit.json。

所有新条件相对A均3/3seed下降，全部不采用。B raw_current平均accuracy74.118%，相对A−3.529pp；C log_rfstyle70.196%，相对A−7.451pp；D raw_rfstyle69.542%，相对A−8.105pp。D−B优化配方增量也3/3负；D−C原计数增量两负一正、均值−0.654pp。accuracy交互+2.876pp只表示两个阴性改动的损失未简单相加，不是正向协同或可用组合。

B best平均source accuracy98.068%高于A97.440%，valid却更低；B末轮source99.702%、valid71.830%也低于best74.118%。在当前结构/训练下，原计数没有泛化收益，不能解释为log1p丢失了必要输入信息。log1p保留。

C/D best平均source accuracy89.778%/90.159%，低于A；末轮92.272%/92.898%，valid69.216%/67.843%。优化bundle同时改变Adam/AdamW、lr、coupled/decoupled weight decay及日程，不能唯一归因于某一项，也不能称全体RF训练方法无效。C seed21729 best在末步12800，而另外两seed末轮较best下降，不能只凭单seed末步最佳自动延长全批预算。停止当前bundle，不追加步数或隐蔽调参。

继续保留多尺度生成器＋Transformer＋log1p＋原AdamW三段LR＋span_mask开发候选77.647%/F1 76.889%，距90%12.353pp；其source97.440%、三seed共同错53/510。B/C/D共同错65/72/86条，仅同一已观察valid的描述，非独立确认。

RF与当前计数/行/标签相同，但历史RF性能85.948%不能由本轮幅值与配方移植复现；这不证明剩余差距唯一来自结构。下一候选应针对当前TAM检验生成器中的局部归一化或逐级时间编码，与当前候选做预算匹配对照，保留已有效的Transformer与mask；不能把此前较弱的token后两块卷积阴性当作所有层级编码均失败，也不能保证改结构达到90%。本次仅记录建议，没有下一轮新训练或未来评价。
