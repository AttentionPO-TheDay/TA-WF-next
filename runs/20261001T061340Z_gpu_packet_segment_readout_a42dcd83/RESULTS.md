# 分段汇聚结果

|条件|seed|source accuracy|valid accuracy|valid F1|
|---|---:|---:|---:|---:|
|global_repeat|21729|99.908%|72.353%|71.932%|
|segments|21729|100.000%|73.333%|72.964%|
|global_repeat|23407|99.980%|71.765%|71.333%|
|segments|23407|100.000%|72.745%|72.184%|
|global_repeat|22026|99.941%|72.941%|72.389%|
|segments|22026|100.000%|73.137%|72.034%|
|historical_mean|21729|99.817%|74.118%|73.470%|
|historical_mean|23407|99.993%|74.706%|74.267%|
|historical_mean|22026|100.000%|73.137%|72.444%|

6任务完成；valid F1 historical_mean=73.394%/global_repeat=71.885%/segments=72.394%；12份新预测与6份历史预测指标核验通过。

{
  "means": {
    "historical_mean": {
      "accuracy": 0.7398692810457517,
      "macro_f1": 0.7339365427600723
    },
    "global_repeat": {
      "accuracy": 0.7235294117647059,
      "macro_f1": 0.7188459325714227
    },
    "segments": {
      "accuracy": 0.730718954248366,
      "macro_f1": 0.7239398038417647
    }
  },
  "comparisons": {
    "segments - global_repeat": {
      "mean_delta_pp": 0.509387127034185,
      "per_seed_pp": [
        1.0319691202044123,
        0.8510607040018603,
        -0.35486844310371746
      ],
      "pass": false
    },
    "segments - historical_mean": {
      "mean_delta_pp": -0.9996738918307472,
      "per_seed_pp": [
        -0.5061496237966812,
        -2.0835537012007577,
        -0.4093183504948028
      ],
      "pass": false
    },
    "global_repeat - historical_mean": {
      "mean_delta_pp": -1.5090610188649323,
      "per_seed_pp": [
        -1.5381187440010935,
        -2.934614405202618,
        -0.05444990739108535
      ],
      "pass": false
    }
  }
}

限制：global_repeat与segments参数总量相同，但重复输入有效秩与表达能力不同，不能称有效容量完全匹配；全局重复扩大线性头改变优化参数化，历史mean为部署门槛。相对四分段按有效token分配，不是已验证资源或burst边界。固定valid已观察，不证明抗漂移或达到90%。
