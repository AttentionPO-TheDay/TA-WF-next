# 源域候选确认重排 v14

预注册主判据：False。仅训练12个微型线性重排器，主网络、LoRA和辅助网络冻结；v9候选不变。

## 完成后解释补充

本段及匹配表的NA标注为终端审核后的报告修订，不改变算法、配置、阈值、模型、预测或主判决。修订前报告和交付清单保留在artifacts。

1. 相对v12，Version的归一化pAUC只增加0.0734个百分点，Recovery@NAR1增加0.0826个百分点，W由1000.667降至979.333；USA的pAUC下降0.4259个百分点，Recovery增加0.0556个百分点，W由118.667升至119.667。仅3/6状态的Recovery提高、2/6状态的pAUC提高，均值微增不能视作稳定收益。
2. 在v12实际损伤计数预算下，Version纠错率仅增加0.0377pp，USA下降0.0770pp。达到各seed的v12纠错数且满足NAR≤1%的可行性，Version仅2/3、USA仅1/3；不能只挑可行seed声明同纠错清洗成功。因此完整主判据False，保留v12，不替换为v14或margin-only对照。
3. 无目标标签调阈值的主策略，在六个状态的最大NAR为0.2773%，都有至少一次真正纠错；Version/USA平均Recovery为2.671%/2.207%。这是一项实际操作点的开发证据，但纠错量有限，且两片没有一致优于各自源阈值v12对照。尤其Version无效改写绝对量仍增加，不能称候选清洗已经完成。
4. 增补的源合成验证诊断显示：主评分相对v12的归一化pAUC增量在Version/USA状态分别为+3.6851/+2.5485pp，而真实切片为+0.0734/−0.4259pp。margin-only对照在源合成验证也提升，在两真实切片却均退化。这更支持“源上可学的排序到真实低风险区间迁移不稳”作为下一项痛点，尚不能单独归因为模型容量或某个合成扰动。源valid已用于校准，不是独立确认，此增补未用于选择模型。

本轮已完整执行。下一步应先定位合成/真实的R/D/W证据与排序差异，再决定是否需要更符合实际的源扰动或直接优化低风险目标；不因本轮微弱均值增益扩大训练或矩阵。

实施记录：第一次源预检遗漏旧实现关闭MHA fastpath的设置；在任何源扰动生成、训练或新目标评分前停止。恢复原推理设置后，全部12个旧源valid状态逐位一致。失败快照、日志及repair_history.json均保留，未放宽容差或更改实验参数。

## 低NAR诊断前沿

下表为三个源seed均值。pAUC/.01 是FPR≤.01区间平均TPR，FPR分母为所有frozen正确样本，等于候选阈值NAR；不是McClish标准化。

|切片|评分|pAUC/.01 %|Recovery@NAR1 %|平均Recovery %|W|W/R|
|---|---|---:|---:|---:|---:|---:|
|version_045|v9_margin|42.846|7.515|4.483|914.333|1.077|
|version_045|v12_primary|52.331|7.866|5.491|1000.667|1.128|
|version_045|margin_control|50.858|7.819|5.337|993.000|1.126|
|version_045|confirmation_primary|52.404|7.949|5.500|979.333|1.094|
|network_USA|v9_margin|40.561|5.195|3.035|102.667|1.567|
|network_USA|v12_primary|60.123|6.254|4.406|118.667|1.532|
|network_USA|margin_control|58.800|6.183|4.299|119.333|1.558|
|network_USA|confirmation_primary|59.697|6.310|4.371|119.667|1.526|

## 同v12纠错数 / 同实际损伤预算

若不足全部3个seed可行，R/D/W均值标记NA，不对不同seed子集作均值比较；逐seed可行结果保留在matched_frontiers.csv。

|切片|匹配|评分|可行seed|R|D|W|
|---|---|---|---:|---:|---:|---:|
|version_045|v12_corrections|v12_primary|3|881.000|127.000|1000.667|
|version_045|v12_corrections|margin_control|1|NA|NA|NA|
|version_045|v12_corrections|confirmation_primary|2|NA|NA|NA|
|version_045|v12_actual_damage|v12_primary|3|881.000|127.000|1000.667|
|version_045|v12_actual_damage|margin_control|3|870.000|125.667|983.667|
|version_045|v12_actual_damage|confirmation_primary|3|885.333|126.333|969.333|
|network_USA|v12_corrections|v12_primary|3|81.667|14.000|118.667|
|network_USA|v12_corrections|margin_control|1|NA|NA|NA|
|network_USA|v12_corrections|confirmation_primary|1|NA|NA|NA|
|network_USA|v12_actual_damage|v12_primary|3|81.667|14.000|118.667|
|network_USA|v12_actual_damage|margin_control|3|80.333|13.667|118.333|
|network_USA|v12_actual_damage|confirmation_primary|3|80.667|14.000|116.333|

## 源验证阈值直接迁移

源阈值只使用816条已观察valid的五种合成输入校准，目标不重新选阈值。各合成输入不是独立trace。以下报告目标实际NAR和Recovery，不声称风险保证。

|切片|评分|目标NAR均值 %|目标NAR最大 %|Recovery均值 %|W均值|
|---|---|---:|---:|---:|---:|
|version_045|v9_margin|0.000|0.000|0.000|0.000|
|version_045|v12_primary|0.134|0.362|1.985|153.667|
|version_045|margin_control|0.184|0.370|2.684|235.333|
|version_045|confirmation_primary|0.153|0.277|2.671|223.333|
|network_USA|v9_margin|0.000|0.000|0.000|0.000|
|network_USA|v12_primary|0.182|0.411|2.336|31.667|
|network_USA|margin_control|0.159|0.342|2.160|31.000|
|network_USA|confirmation_primary|0.136|0.274|2.207|27.000|

## 预注册判决

```json
{
  "primary": "confirmation_primary",
  "primary_passed": false,
  "slices": [
    {
      "slice": "version_045",
      "passed": false,
      "pauc_norm_gain": 0.0007338630370081756,
      "rec1_gain_pp": 0.08263576643177672,
      "same_damage_rec_gain_pp": 0.03768386496634385,
      "matched_correction_feasible": false,
      "matched_correction_W_ratio": null,
      "matched_correction_D_change": null,
      "cleanup_gate": false,
      "recovery_gate": false
    },
    {
      "slice": "network_USA",
      "passed": false,
      "pauc_norm_gain": -0.004259322888231831,
      "rec1_gain_pp": 0.055605812784673136,
      "same_damage_rec_gain_pp": -0.07698501193025703,
      "matched_correction_feasible": false,
      "matched_correction_W_ratio": null,
      "matched_correction_D_change": null,
      "cleanup_gate": false,
      "recovery_gate": false
    }
  ],
  "source_threshold_all_target_NAR_le_1pct_and_positive_recovery": true,
  "query_thresholds_diagnostic_only": true,
  "integrated": false
}
```

## 成本与信息边界

源train固定24行/类，共2448行；valid固定8行/类，共816行。主模型已用整个18553行训练，元训练并非模型外预测。合成扰动保持网站标签是建模假设，强截尾不保证可识别；源阈值不保证真实域风险。辅助三seed×五视图仍有额外推理成本，本轮只是复用其目标缓存，不能称方法零开销。

目标仍为已观察Version045/USA诊断切片，标签身份沿用内容支持假设，不是官方映射；Version/Network数据事实及外部数据角色不变。query标签未用于拟合任何参数；本轮的方法设计受v13目标诊断启发，不能称独立确认。

主评分未因对照更好而替换；所有24目标评分状态、12源特征状态、各seed及所有源阈值结果均保留。全局AUROC不用于主验收。目标诊断前沿的阈值由query标签选择，不能部署；源阈值策略与之明确分开。
