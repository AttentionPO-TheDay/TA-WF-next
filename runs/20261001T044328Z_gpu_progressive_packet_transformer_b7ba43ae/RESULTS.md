# 渐进packet CNN实验结果

|条件|seed|训练accuracy|valid accuracy|valid F1|
|---|---:|---:|---:|---:|
|transformer|21729|100.000%|71.373%|70.665%|
|mlp|21729|91.562%|71.961%|71.231%|
|transformer|23407|100.000%|70.588%|70.164%|
|mlp|23407|99.954%|71.569%|70.879%|
|transformer|22026|100.000%|72.745%|72.153%|
|mlp|22026|99.908%|72.941%|72.137%|
|historical_df|21729|90.588%|71.176%|70.156%|
|historical_df|23407|90.248%|71.569%|70.190%|
|historical_df|22026|85.039%|69.608%|68.582%|
|historical_transformer|21729|60.477%|52.941%|50.622%|
|historical_transformer|23407|59.908%|51.373%|49.692%|
|historical_transformer|22026|62.183%|53.922%|52.329%|

6任务完成；CNN-Transformer/MLP valid F1=70.994/71.416%；重载核验12份新预测、历史指标核验12份通过。

{
  "means": {
    "transformer": {
      "accuracy": 0.7156862745098039,
      "macro_f1": 0.7099412025882614
    },
    "mlp": {
      "accuracy": 0.7215686274509804,
      "macro_f1": 0.7141553798416544
    },
    "historical_transformer": {
      "accuracy": 0.5274509803921568,
      "macro_f1": 0.5088108285195944
    },
    "historical_df": {
      "accuracy": 0.707843137254902,
      "macro_f1": 0.6964260727524856
    }
  },
  "comparisons": {
    "transformer vs mlp": {
      "f1_delta_pp": [
        -0.5658176246411517,
        -0.7143319643319601,
        0.0158964129552297
      ],
      "mean_delta_pp": -0.42141772533929406,
      "pass": false
    },
    "transformer vs historical_transformer": {
      "f1_delta_pp": [
        20.043283321830042,
        20.472082383847095,
        19.82374651492299
      ],
      "mean_delta_pp": 20.11303740686671,
      "pass": true
    },
    "transformer vs historical_df": {
      "f1_delta_pp": [
        0.5092565386683123,
        -0.025392555669367933,
        3.570674967733789
      ],
      "mean_delta_pp": 1.3515129835775777,
      "pass": false
    },
    "mlp vs historical_transformer": {
      "f1_delta_pp": [
        20.609100946471195,
        21.186414348179056,
        19.807850101967762
      ],
      "mean_delta_pp": 20.534455132206006,
      "pass": true
    }
  }
}

解释限制：两新条件共享CNN、位置编码和读出初始化、批次流、训练预算；Transformer与MLP参数量不同，不是严格容量匹配。历史多视角与新packet输入信息一致但表示/架构不同；DF训练配方、参数量和呈现次数不同。历史结果仅复用，不算新重复。valid已观察、只有一次训练样本抽样、三seed非独立数据。无未来评价，不证明抗漂移。
