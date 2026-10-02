# TAM生成器归一化×逐级编码结果

9项新训练+3项mask历史复用完成；flat_none=77.647%/flat_norm=78.497%/hier_none=77.255%/hier_norm=78.627%；通过候选[]；48组预测指标和冻结hash通过。

|条件|accuracy三seed均值|Macro-F1|≥90%|
|---|---:|---:|---|
|flat_none|77.647%|76.889%|False|
|flat_norm|78.497%|78.149%|False|
|hier_none|77.255%|76.798%|False|
|hier_norm|78.627%|78.193%|False|

|条件|seed|best步数|source accuracy|valid accuracy|valid Macro-F1|秒|
|---|---:|---:|---:|---:|---:|---:|
|flat_none|21729|7520|97.131%|77.647%|76.830%|358.3|
|flat_norm|21729|9920|99.131%|80.000%|79.812%|409.7|
|hier_none|21729|12320|99.379%|77.255%|76.872%|386.4|
|hier_norm|21729|11360|98.634%|79.020%|78.678%|409.5|
|flat_none|23407|7520|96.157%|78.235%|77.384%|355.6|
|flat_norm|23407|12320|98.444%|77.451%|77.038%|401.7|
|hier_none|23407|8960|98.039%|78.039%|77.448%|379.6|
|hier_norm|23407|9920|98.072%|78.039%|77.565%|397.5|
|flat_none|22026|11840|99.033%|77.059%|76.452%|343.8|
|flat_norm|22026|9920|98.915%|78.039%|77.598%|399.5|
|hier_none|22026|10400|98.771%|76.471%|76.074%|382.4|
|hier_norm|22026|8000|96.706%|78.824%|78.337%|391.9|

```json
{
  "comparisons": {
    "flat_norm - flat_none": {
      "accuracy_mean_delta_pp": 0.8496732026143797,
      "accuracy_per_seed_pp": [
        2.352941176470591,
        -0.7843137254901933,
        0.9803921568627416
      ],
      "macro_f1_mean_delta_pp": 1.2603464074052404,
      "pass": false
    },
    "hier_none - flat_none": {
      "accuracy_mean_delta_pp": -0.39215686274510037,
      "accuracy_per_seed_pp": [
        -0.39215686274509665,
        -0.19607843137254832,
        -0.5882352941176561
      ],
      "macro_f1_mean_delta_pp": -0.09071719856033056,
      "pass": false
    },
    "hier_norm - flat_norm": {
      "accuracy_mean_delta_pp": 0.13071895424836183,
      "accuracy_per_seed_pp": [
        -0.9803921568627527,
        0.588235294117645,
        0.7843137254901933
      ],
      "macro_f1_mean_delta_pp": 0.044002838120490065,
      "pass": false
    },
    "hier_norm - hier_none": {
      "accuracy_mean_delta_pp": 1.3725490196078418,
      "accuracy_per_seed_pp": [
        1.764705882352935,
        0.0,
        2.352941176470591
      ],
      "macro_f1_mean_delta_pp": 1.3950664440860607,
      "pass": false
    },
    "hier_norm - flat_none": {
      "accuracy_mean_delta_pp": 0.9803921568627416,
      "accuracy_per_seed_pp": [
        1.3725490196078383,
        -0.19607843137254832,
        1.764705882352935
      ],
      "macro_f1_mean_delta_pp": 1.3043492455257302,
      "pass": false
    }
  },
  "interaction": {
    "accuracy": {
      "mean_pp": 0.5228758169934622,
      "per_seed_pp": [
        -0.5882352941176561,
        0.7843137254901933,
        1.3725490196078494
      ]
    },
    "macro_f1": {
      "mean_pp": 0.1347200366808206,
      "per_seed_pp": [
        -1.1761114702291353,
        0.46347443406268285,
        1.1167971462089143
      ]
    }
  }
}
```

所有条件同一Transformer＋log1p＋原AdamW配方＋固定span_mask，生成器仅按两因素改变，原始TAM行/标签、共享初始state、索引、步数和20次选模相同。归一化只沿逐bin通道统计，新增64可训练参数；逐级编码改变两层之间采样网格、聚合位置及感受范围，不能唯一归因于层级概念或称参数/计算完全匹配。
baseline3seed明确历史复用，不是新增独立重复；新模型从头初始化，无teacher/旧checkpoint训练。固定valid反复开发，三seed是优化重复，不证明外部泛化/抗漂移。交互仅描述数值；不自动追加参数扫描或合并未通过模块，源期90%目标独立判断。
审计与冻结规则见PLAN.md/SOURCES.md，完整产物在artifacts/checkpoints/logs。无未来日期、WTT/AWF或适应。

## 完成后独立复核与解释（2026-10-02）

9项新训练全部完成、3项mask基线明确历史复用，整批1242.43秒（约20.7分钟）。49个冻结文件hash一致，12报告best/last×source/valid共48组accuracy/Macro-F1独立复算一致；共享初始化、norm四个参数tensor初始gamma1/beta0、相同索引、12800步/20机会和最早最大accuracy选模核验通过。checkpoint重载预测已由worker/preflight完成，本次没有新模型推理/训练/未来访问。证据artifacts/postrun_audit.json。

最高观察三seed mean为hier_norm78.627%/F1 78.193%，相对flat_none accuracy+0.980pp、仅2/3seed正，未满足平均≥1pp且3/3正的预定门槛。其逐seed accuracy79.020/78.039/78.824%，相对baseline+1.373/−0.196/+1.765pp；不能把+0.980四舍五入成达到1pp，也不能根据本次接近门槛事后修改规则。观察最高距90%11.373pp，正式保留baseline77.647%仍差12.353pp。

flat_norm78.497%/F1 78.149%，相对baseline+0.850pp、2/3正；seed21729达到80.000%但不能替代三seed均值目标。hier_none77.255%，相对baseline−0.392pp、三seed全负，单独逐级编码不采用。hier_norm−hier_none+1.373pp但seed23407准确率相等、只有2/3严格为正，也未过一致性门槛；相对flat_norm仅+0.131pp且方向混合。归一化有描述性正向信号，尚未建立稳定增益；不能推断所有norm或层级设计无效，也不能声称有效协同。

四条件best checkpoint mean source accuracy97.440/98.830/98.730/97.804%，末轮99.296/99.135/99.248/98.895%；末轮valid76.405/76.928/75.425/77.059%，全部12任务末轮valid低于best。训练拟合高，未有直接延长预算的证据，也不能唯一把泛化差距归于生成器、token化或分类器。三seed共同错样本53/49/51/48条，仅同一valid描述。

## 既有预测错误重合诊断

为缩小下一步候选，只读取已有RF与当前模型valid预测，核验RF/current的valid行、标签、原TAM完全相同，无新推理或训练，证据artifacts/historical_error_overlap.json。baseline三seed共同错53条中，RF三个seed都正确22条，至少两个seed正确26条；有21条RF也三个seed都错。hier_norm三seed共同错48条中RF三个seed均正确16条。逐seedbaseline错而RF对69/65/80条，RF错而baseline对22/29/36条。不同模型/配方能识别部分持续难例，但不证明某个结构是唯一瓶颈，也没有用真标签做线上路由或声称oracle集成性能。

继续正式保留多尺度生成器＋Transformer＋log1p＋原AdamW＋mask77.647%开发基线；归一化与组合的78.497/78.627%作为未过门槛的观察候选保留，不能当已确认替代。下一方向建议检验source-only RF软目标蒸馏能否将较强模型的判别信号传给自有生成器＋Transformer，与同一baseline纯CE作受控比较；teacher来源、训练预算、soft-target生成权限及loss必须另行冻结，valid/query不得进入蒸馏梯度。错误重合只是该候选的设计依据，不能保证蒸馏有效或90%可达。本次未启动该实验，未来与漂移调整仍关闭。
