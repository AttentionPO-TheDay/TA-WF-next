# 固定折模型校准与投票 v19

全部预定校准、终点评价和审计完成。仅已观察TemporalDrift开发；受v18反馈影响，无独立确认主张。

同v17/v18标签预算，cap3=303/305、cap5=496/504；不新增梯度训练或GPU任务。648个固定路由阈值、726评价状态。对应模型未用自己校准折标签训练；选择后不full-refit。折间仍有依赖，经验低NAR不是风险认证。

## 主判决

```json
{
  "primary": {
    "route": "agree_head1_shrink",
    "mode": "zero",
    "votes": 2,
    "cap": 5,
    "gate": "both dates all9 actual NAR<=1%; mean deltaF1>=1pp and >=50% rawfullhead Recovery; sameD Recovery vs rawfullhead diagnosticNAR1 nonnegative mean and >=2/3 modelseed means nonnegative"
  },
  "passed": false,
  "roles": [
    {
      "role": "day90",
      "all9_NAR_le_1pct": true,
      "delta_f1_pp": 3.1006875249238792,
      "Recovery_retention": 0.33612853880078275,
      "same_damage_Recovery_gain_pp": -5.176743671241183,
      "nonnegative_model_seeds": 0,
      "passed": false
    },
    {
      "role": "day270",
      "all9_NAR_le_1pct": true,
      "delta_f1_pp": 4.825070058226146,
      "Recovery_retention": 0.317984208381152,
      "same_damage_Recovery_gain_pp": -5.341579889880319,
      "nonnegative_model_seeds": 0,
      "passed": false
    }
  ],
  "risk_certified": false,
  "deployed": false
}
```

## 完整操作点

|日期|预算|固定策略|F1 %|ΔF1 pp|NAR均值 %|NAR最大 %|Recovery %|纠错保留 %|同D纠错差 pp|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|day90|cap3|agree_head1_shrink_uncalibrated_vote2|75.135|+3.961|0.982|1.203|14.185|61.3|+3.424|
|day90|cap3|agree_head1_shrink_uncalibrated_vote3|73.415|+2.240|0.278|0.349|7.173|31.0|-3.332|
|day90|cap3|agree_head1_shrink_zero_vote2|73.340|+2.165|0.299|0.829|6.747|29.3|-3.758|
|day90|cap3|agree_head1_shrink_zero_vote3|71.917|+0.743|0.038|0.154|2.085|9.1|-8.419|
|day90|cap3|agree_head1_shrink_onepct_vote2|73.340|+2.165|0.299|0.829|6.747|29.3|-3.758|
|day90|cap3|agree_head1_shrink_onepct_vote3|71.917|+0.743|0.038|0.154|2.085|9.1|-8.419|
|day90|cap3|local_head1_uncalibrated_vote2|74.917|+3.742|0.798|0.938|13.075|56.5|+2.570|
|day90|cap3|local_head1_uncalibrated_vote3|72.852|+1.678|0.112|0.182|5.132|22.2|-5.373|
|day90|cap3|local_head1_zero_vote2|73.352|+2.177|0.325|0.675|6.870|29.9|-3.634|
|day90|cap3|local_head1_zero_vote3|71.783|+0.608|0.019|0.070|1.695|7.4|-8.809|
|day90|cap3|local_head1_onepct_vote2|73.352|+2.177|0.325|0.675|6.870|29.9|-3.634|
|day90|cap3|local_head1_onepct_vote3|71.783|+0.608|0.019|0.070|1.695|7.4|-8.809|
|day90|cap3|local_protect_uncalibrated_vote2|74.915|+3.741|0.793|0.928|13.058|56.5|+2.554|
|day90|cap3|local_protect_uncalibrated_vote3|72.855|+1.681|0.110|0.177|5.138|22.2|-5.367|
|day90|cap3|local_protect_zero_vote2|73.359|+2.184|0.322|0.675|6.885|30.0|-3.619|
|day90|cap3|local_protect_zero_vote3|71.784|+0.610|0.019|0.070|1.700|7.5|-8.805|
|day90|cap3|local_protect_onepct_vote2|73.359|+2.184|0.322|0.675|6.885|30.0|-3.619|
|day90|cap3|local_protect_onepct_vote3|71.784|+0.610|0.019|0.070|1.700|7.5|-8.805|
|day90|cap3|full_head1|76.123|+4.949|3.774|4.305|23.118|100.0|+0.000|
|day90|cap3|full_local_head1|75.642|+4.468|1.052|1.262|15.831|68.5|+4.737|
|day90|cap5|agree_head1_shrink_uncalibrated_vote2|76.353|+5.179|0.903|1.086|17.789|65.5|+3.401|
|day90|cap5|agree_head1_shrink_uncalibrated_vote3|74.273|+3.099|0.250|0.303|9.763|35.9|-4.574|
|day90|cap5|agree_head1_shrink_zero_vote2|74.275|+3.101|0.206|0.354|9.160|33.6|-5.177|
|day90|cap5|agree_head1_shrink_zero_vote3|71.998|+0.823|0.013|0.040|2.142|7.7|-12.195|
|day90|cap5|agree_head1_shrink_onepct_vote2|74.760|+3.585|0.313|0.765|10.904|40.0|-3.433|
|day90|cap5|agree_head1_shrink_onepct_vote3|72.368|+1.193|0.030|0.094|3.290|12.0|-11.047|
|day90|cap5|local_head1_uncalibrated_vote2|75.923|+4.749|0.758|0.889|16.169|59.5|+1.832|
|day90|cap5|local_head1_uncalibrated_vote3|73.505|+2.331|0.121|0.159|7.160|26.3|-7.177|
|day90|cap5|local_head1_zero_vote2|73.243|+2.068|0.211|0.794|6.359|23.3|-7.978|
|day90|cap5|local_head1_zero_vote3|71.694|+0.520|0.017|0.124|1.475|5.3|-12.862|
|day90|cap5|local_head1_onepct_vote2|74.491|+3.317|0.331|0.794|10.198|37.5|-4.139|
|day90|cap5|local_head1_onepct_vote3|72.267|+1.093|0.025|0.124|3.037|11.2|-11.300|
|day90|cap5|local_protect_uncalibrated_vote2|75.919|+4.745|0.756|0.889|16.153|59.5|+1.816|
|day90|cap5|local_protect_uncalibrated_vote3|73.502|+2.327|0.122|0.164|7.152|26.3|-7.185|
|day90|cap5|local_protect_zero_vote2|73.314|+2.139|0.229|0.789|6.646|24.3|-7.691|
|day90|cap5|local_protect_zero_vote3|71.761|+0.587|0.023|0.124|1.707|6.1|-12.630|
|day90|cap5|local_protect_onepct_vote2|74.559|+3.385|0.350|0.789|10.482|38.5|-3.855|
|day90|cap5|local_protect_onepct_vote3|72.336|+1.161|0.031|0.124|3.271|12.0|-11.066|
|day90|cap5|full_head1|77.324|+6.149|3.827|4.379|27.157|100.0|+0.000|
|day90|cap5|full_local_head1|76.664|+5.490|1.045|1.192|19.125|70.4|+4.312|
|day270|cap3|agree_head1_shrink_uncalibrated_vote2|65.660|+6.430|1.014|1.351|16.243|59.5|+3.109|
|day270|cap3|agree_head1_shrink_uncalibrated_vote3|62.796|+3.566|0.314|0.462|8.337|30.6|-4.438|
|day270|cap3|agree_head1_shrink_zero_vote2|63.511|+4.281|0.351|0.787|9.648|34.9|-3.126|
|day270|cap3|agree_head1_shrink_zero_vote3|61.036|+1.806|0.080|0.248|3.749|13.4|-9.026|
|day270|cap3|agree_head1_shrink_onepct_vote2|63.511|+4.281|0.351|0.787|9.648|34.9|-3.126|
|day270|cap3|agree_head1_shrink_onepct_vote3|61.036|+1.806|0.080|0.248|3.749|13.4|-9.026|
|day270|cap3|local_head1_uncalibrated_vote2|65.631|+6.401|0.902|1.172|15.478|56.7|+2.535|
|day270|cap3|local_head1_uncalibrated_vote3|61.745|+2.515|0.135|0.239|5.719|21.0|-7.056|
|day270|cap3|local_head1_zero_vote2|63.493|+4.263|0.338|0.487|9.184|33.6|-3.591|
|day270|cap3|local_head1_zero_vote3|60.161|+0.930|0.019|0.051|1.803|6.6|-10.972|
|day270|cap3|local_head1_onepct_vote2|63.493|+4.263|0.338|0.487|9.184|33.6|-3.591|
|day270|cap3|local_head1_onepct_vote3|60.161|+0.930|0.019|0.051|1.803|6.6|-10.972|
|day270|cap3|local_protect_uncalibrated_vote2|65.626|+6.396|0.898|1.146|15.467|56.7|+2.537|
|day270|cap3|local_protect_uncalibrated_vote3|61.744|+2.514|0.135|0.239|5.721|21.0|-7.054|
|day270|cap3|local_protect_zero_vote2|63.494|+4.264|0.334|0.496|9.184|33.6|-3.590|
|day270|cap3|local_protect_zero_vote3|60.159|+0.929|0.017|0.034|1.801|6.6|-10.974|
|day270|cap3|local_protect_onepct_vote2|63.494|+4.264|0.334|0.496|9.184|33.6|-3.590|
|day270|cap3|local_protect_onepct_vote3|60.159|+0.929|0.017|0.034|1.801|6.6|-10.974|
|day270|cap3|full_head1|68.626|+9.396|4.828|5.341|27.280|100.0|+0.000|
|day270|cap3|full_local_head1|67.035|+7.805|1.186|1.497|18.994|69.7|+4.758|
|day270|cap5|agree_head1_shrink_uncalibrated_vote2|67.899|+8.669|1.030|1.232|21.180|66.2|+5.204|
|day270|cap5|agree_head1_shrink_uncalibrated_vote3|64.322|+5.092|0.328|0.411|11.730|36.7|-3.786|
|day270|cap5|agree_head1_shrink_zero_vote2|64.055|+4.825|0.187|0.610|10.175|31.8|-5.342|
|day270|cap5|agree_head1_shrink_zero_vote3|60.778|+1.548|0.026|0.178|3.033|9.5|-12.484|
|day270|cap5|agree_head1_shrink_onepct_vote2|65.515|+6.285|0.356|0.864|13.862|43.4|-1.654|
|day270|cap5|agree_head1_shrink_onepct_vote3|61.771|+2.541|0.058|0.205|5.142|16.1|-10.374|
|day270|cap5|local_head1_uncalibrated_vote2|67.543|+8.313|0.853|0.941|19.597|61.3|+4.081|
|day270|cap5|local_head1_uncalibrated_vote3|63.109|+3.879|0.162|0.218|8.625|27.0|-6.892|
|day270|cap5|local_head1_zero_vote2|64.440|+5.210|0.301|0.500|11.007|34.5|-4.509|
|day270|cap5|local_head1_zero_vote3|60.978|+1.748|0.024|0.085|3.299|10.4|-12.218|
|day270|cap5|local_head1_onepct_vote2|65.674|+6.444|0.495|0.739|14.282|44.7|-1.234|
|day270|cap5|local_head1_onepct_vote3|61.739|+2.509|0.060|0.118|5.131|16.1|-10.385|
|day270|cap5|local_protect_uncalibrated_vote2|67.536|+8.305|0.847|0.924|19.576|61.2|+4.059|
|day270|cap5|local_protect_uncalibrated_vote3|63.104|+3.874|0.161|0.218|8.617|27.0|-6.900|
|day270|cap5|local_protect_zero_vote2|64.442|+5.212|0.300|0.500|11.017|34.5|-4.500|
|day270|cap5|local_protect_zero_vote3|60.983|+1.753|0.024|0.085|3.310|10.4|-12.207|
|day270|cap5|local_protect_onepct_vote2|65.677|+6.447|0.490|0.730|14.284|44.7|-1.233|
|day270|cap5|local_protect_onepct_vote3|61.738|+2.508|0.062|0.118|5.133|16.1|-10.383|
|day270|cap5|full_head1|70.659|+11.428|4.804|5.046|31.970|100.0|+0.000|
|day270|cap5|full_local_head1|68.989|+9.759|1.144|1.240|23.248|72.7|+6.662|

2票与3票同时报告，uncalibrated投票是同折模型/结构对照；相同D诊断前沿只用于解释，不能部署query阈值。所有均值共享同一query，不能按726次独立重复解释。最差状态比平均NAR更能暴露不稳定性。

主判决不因次级方法更好改写。不据终点评价追加阈值/投票调整。不同于v18的设计同时固定了路由并取消full-refit，若收益改善，仍不能单独将其归因于refit这一因素。WTT/AWF关闭，旧结果不改。

324个复用折模型、36OOF、648阈值和726评价状态已完整重载；独立投票/指标/前沿复算通过。

## 完成后解释：固定局部路由与两票通过

这是预先列出的次级方法，在终点评价后选作讨论，未替换主臂；v18/v19主判决均保持False。以下不增加训练、标签或阈值选择。

|日期|每次标签数|F1 %|ΔF1 pp|NAR均值 %|NAR最大 %|Recovery %|原head纠错保留 %|同实际D较head纠错差 pp|同实际D较full-local纠错差 pp|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|day90|496|75.923|+4.749|0.758|0.889|16.169|59.5|+4.062|-0.901|
|day270|504|67.543|+8.313|0.853|0.941|19.597|61.3|+5.636|-1.336|

该候选在18个模型/抽样状态的直接输出上均观察到NAR<1%，不使用query定位阈值；这支持一个值得冻结再验证的操作点。不是未来域风险保证，也不是新的独立确认。

同实际D比较的控制阈值由query标签定位，只是诊断对照。若相对full-local前沿仍下降，则本轮证明的是更保守且可直接执行的折模型投票，不能据此宣称超过最强几何路由的整体Pareto前沿。

v18保护项对结果影响很小，优先保留简单head；v19额外零损伤阈值和三票全同意会大量丢失纠错，主方案因此失败。当前证据不支持继续加严拒绝。三个模型只是共享冻结特征后的三个小头，不需要三个骨干前向。

少数类只有一个支持样本时，一个折的训练近邻库会缺该类；三票局部同意将无法纠正到这个类，两票仍有可能。该结构事实不能单独解释全部纠错损失。

描述性风险—纠错对比图：[PNG](artifacts/cap5_risk_recovery.png)、[PDF](artifacts/cap5_risk_recovery.pdf)。横线表示9个状态的NAR最小至最大范围，不是置信区间；绘图不参与选模或阈值选择。
