# GPU保序卷积分类器对照结果

3项GPU CNN完成；accuracy 73.203% vs baseline 75.948%；增量-2.745pp PASS=False；24组指标和冻结hash通过。

|模型|seed|best步数|source accuracy|valid accuracy|valid F1|
|---|---:|---:|---:|---:|---:|
|baseline|21729|7040|97.725%|76.667%|75.895%|
|temporal_cnn|21729|7520|97.248%|72.549%|71.424%|
|baseline|23407|10400|99.562%|75.490%|74.978%|
|temporal_cnn|23407|7040|96.562%|73.725%|73.204%|
|baseline|22026|10400|99.373%|75.686%|75.013%|
|temporal_cnn|22026|6560|95.046%|73.333%|72.485%|

```json
{
  "means": {
    "baseline": {
      "accuracy": 0.7594771241830065,
      "macro_f1": 0.7529537093262583
    },
    "temporal_cnn": {
      "accuracy": 0.7320261437908497,
      "macro_f1": 0.7237078643941387
    }
  },
  "accuracy_mean_delta_pp": -2.7450980392156876,
  "accuracy_per_seed_pp": [
    -4.117647058823537,
    -1.764705882352935,
    -2.352941176470591
  ],
  "macro_f1_mean_delta_pp": -2.92458449321195,
  "pass": false,
  "metric_groups_verified": 24,
  "target_reached": false,
  "historical_baseline_seeds": 3,
  "cpu_history_used_as_training": false
}
```

新CNN三seed全部GPU从头初始化；baseline三seed明确历史GPU复用，不是新重复；CPU部分与已完成单seed保留在前run，不混入均值或用于warm start。此设备切换经用户授权，有既往valid反馈暴露，属于开发而非独立确认。
结构、参数和计算量不同，其他共享初始state、数据、索引、步数、选模机会一致；不能称纯attention消融或计算量匹配。候选门槛与90%分别判断，无未来日期/适应或抗漂移结论。

## 完成后独立复核与结论（2026-10-02）

3项GPU CNN全部完成，总管耗时236.45秒；3项Transformer基线明确历史复用。独立复核41个本run冻结文件和67个前run冻结文件hash一致；本run24组best/last×source/valid指标、前runmask额外12组指标共36组accuracy/Macro-F1从保存预测重算一致，9报告的初始化、索引、12800步/20次评价及最早最大accuracy选模一致。checkpoint预测重载已由worker/preflight核验，本次没有新模型推理、训练或未来访问；证据artifacts/postrun_audit.json。原CPU实验由用户授权停止，部分产物保留，未混入GPU均值。

CNN mean valid accuracy73.203%、F1 72.371%，相对Transformer75.948%/75.295%下降2.745/2.925pp；逐seed accuracy差−4.118/−1.765/−2.353pp，全部为负。此具体两块保序卷积替代方案不采用，不否定所有CNN、混合CNN/attention结构，也不能唯一归因为缺attention（参数与感受范围同时不同）。

CNN best checkpoint mean source accuracy96.285%、last99.455%，last valid72.157%，三个seed last valid均低于best；没有支持直接延长训练的证据。CNN三seed共同错89/510，比Transformer53/510更多，描述性错误集中不等于独立确认。

前run单独训练的多尺度生成器＋Transformer＋span_mask仍为当前自有框架开发候选：valid mean accuracy77.647%、F1 76.889%，相对同结构不增强baseline+1.699pp、3/3seed正、F1+1.594pp，符合预定门槛；距90%仍差12.353pp。该增强未应用于本轮CNN，不能宣称对所有架构通用。它仅是已观察source/valid开发收益，不是外部泛化或漂移鲁棒性证明。

下一步建议先核对该候选与历史RF 85.948%参考的编码/汇聚、归一化及训练配方差异，缩小可检验的瓶颈，再冻结针对性的表示或训练对照，而非继续无界增加模块。RF数值只作不同配方的性能参考，不是同预算结构归因；当前尚未定位单一瓶颈，不保证后续达到90%。此处仅建议，未追加新训练或未来评分。
