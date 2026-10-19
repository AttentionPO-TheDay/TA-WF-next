# 固定v9候选的低NAR软重排序 v12

主判据：False。零训练、零新前向，只改变固定v9候选的排序。

FPR以全部frozen正确样本为分母，等于候选阈值NAR。partial ROC正例只含v9可真正纠正的候选，其他错误改写另记；pAUC归一化值为raw area/.01的低风险平均TPR，非McClish标准化。确定性面积用最大可行Recovery左阶梯，不拆同分。

|切片|评分|pAUC/.01 %|确定性平均Recovery %|Rec@NAR1 %|无效改写数|每纠错无效改写|
|---|---|---:|---:|---:|---:|---:|
|version_045|v9_margin|42.846|4.483|7.515|914.333|1.077|
|version_045|v11_min|44.325|4.662|7.298|1199.667|1.478|
|version_045|equal_logodds|45.412|4.783|7.431|1013.667|1.225|
|version_045|weighted_logodds|45.434|4.785|7.431|1012.000|1.223|
|version_045|weighted_soft|45.133|4.754|7.360|994.000|1.215|
|version_045|margin_plus_logodds|52.453|5.504|7.926|1012.000|1.134|
|version_045|margin_plus_soft|52.331|5.491|7.866|1000.667|1.128|
|version_045|shuffled_margin_soft|34.365|3.589|6.191|823.333|1.183|
|network_USA|v9_margin|40.561|3.035|5.195|102.667|1.567|
|network_USA|v11_min|46.522|3.381|5.188|119.667|1.848|
|network_USA|equal_logodds|51.449|3.756|5.847|113.667|1.584|
|network_USA|weighted_logodds|51.529|3.762|5.873|114.667|1.592|
|network_USA|weighted_soft|50.656|3.698|5.824|113.667|1.577|
|network_USA|margin_plus_logodds|60.347|4.422|6.229|120.667|1.563|
|network_USA|margin_plus_soft|60.123|4.406|6.254|118.667|1.532|
|network_USA|shuffled_margin_soft|28.051|2.100|4.749|94.000|1.593|

主判决（只margin_plus_soft，不用次评分替换）：

```json
{
  "primary": "margin_plus_soft",
  "passed": false,
  "slices": [
    {
      "slice": "version_045",
      "pauc_norm_gain": 0.09484375403923057,
      "rec1_gain_pp": 0.3510500647909309,
      "step_mean_recovery_gain_pp": 1.0079063309565865,
      "invalid_per_correction_not_worse": false,
      "passed": false
    },
    {
      "slice": "network_USA",
      "pauc_norm_gain": 0.19562018791728342,
      "rec1_gain_pp": 1.059612769143782,
      "step_mean_recovery_gain_pp": 1.3706176352367347,
      "invalid_per_correction_not_worse": true,
      "passed": true
    }
  ],
  "candidate_pool": "v9_smooth_unchanged",
  "global_auroc_used_for_selection": false,
  "integrated": false
}
```

全部96个valid/query逐seed结果与标准差见cells/summary CSV。sourcevalid在两目标切片中重复用于诊断，不是独立重复。候选生成完全不变，raw v9全接纳性能不变；NAR1阈值使用query标签作终端诊断，不能部署或当作风险保证。没有将其他错误改写算成修复，完整计数见cells.csv。

15个视图相关，源NLL权重只表示源域支持度，不能解释为独立证据连乘或真实误改概率。soft加权均值并非完整贝叶斯模型。源valid曾用于checkpoint/温度与本轮校准，不能称未见确认。main/aux源均值SD在查询评分前冻结，没有目标拟合、阈值迁移或超参搜索。v10/v11辅助模型成本与推理缓存成本仍存在，本轮零新计算不代表完整方法零开销。

本轮只在已观察Version045/USA开发切片评价，不声称物理丢包率已知；标签对应沿用内容支持假设而非官方映射。WTT/AWF关闭，旧结果封存不改。

全部96状态评分逐位重载一致、局部ROC独立积分/混淆矩阵/实际阈值及无效改写重算通过；24历史基线前沿精确一致，候选类别与source校准不变，输入输出SHA封存。

## 完成后解释补充（不改变预注册判据）

本段为终端结果核验后的解释补充。配置、评分代码、校准、候选类别与原主判决均不修改；补充前的交付清单另存为 artifacts/delivery_before_interpretation.json。

低风险重排序已出现实际收益：主评分相对 v9_margin，Version045/USA 的归一化 pAUC 分别提高 9.484/19.562 个百分点，Recovery@NAR≤1% 分别提高 0.351/1.060 个百分点，确定性风险区间平均 Recovery 分别提高 1.008/1.371 个百分点。逐 seed 核验中 pAUC 为 6/6 正向，Recovery 为 5/6 正向、1/6 不变；这只是已观察两片的开发证据，未建立统计显著性或外部有效性。

完整主判据仍未通过，原因是 Version 每次真正纠错伴随的旧错→其他错比率由 1.077 升至 1.128；USA 由 1.567 降至 1.532。两片无效改写绝对数量均上升，不能宣称已完成候选清洗。该失败不否定同 NAR 预算下的纠错改进，也不能用局部收益替代完整验收。

收益主要与保留 v9 裕度并加入软对数证据的组合相符。源域可靠性权重接近均匀，等权与加权证据差异很小；方差惩罚相对无惩罚融合没有一致优势。这些是描述性消融，不足以分别确立可靠性加权或方差惩罚的独立机制作用。未选用次评分替换预注册主评分，未部署由 query 标签选出的诊断阈值。
