# 跨视图稳定性 v11

主候选门槛：False；源不变性：True；相对v9可利用性：False。零新训练、零TTA整合。

所有结果为两个已观察诊断切片，三seed均值；同NAR1阈值由query标签定位，不能部署。主视图direction_iat由v10反馈选择，明确为新一轮开发；不能当作独立确认。候选类别始终不变，单独隔离排序信号作用。

|候选池|切片|评分|Rec@NAR1 %|较同池baseline pp|纠错vs损伤AUROC|完整改写AUROC|
|---|---|---|---:|---:|---:|---:|
|v10_view|version_045|baseline|1.383|0.000|0.627|0.627|
|v10_view|version_045|perturb_support|1.628|0.245|0.679|0.645|
|v10_view|version_045|seed_support|1.508|0.125|0.685|0.647|
|v10_view|version_045|joint_support|1.813|0.430|0.710|0.653|
|v10_view|version_045|shuffled_joint|0.300|-1.083|0.501|0.500|
|v10_view|network_USA|baseline|3.168|0.000|0.641|0.638|
|v10_view|network_USA|perturb_support|3.245|0.077|0.706|0.662|
|v10_view|network_USA|seed_support|2.961|-0.207|0.731|0.680|
|v10_view|network_USA|joint_support|3.223|0.056|0.755|0.680|
|v10_view|network_USA|shuffled_joint|0.358|-2.809|0.488|0.493|
|v9_smooth|version_045|baseline|7.515|0.000|0.679|0.622|
|v9_smooth|version_045|perturb_support|7.373|-0.142|0.665|0.577|
|v9_smooth|version_045|seed_support|7.328|-0.187|0.674|0.572|
|v9_smooth|version_045|joint_support|7.298|-0.217|0.675|0.551|
|v9_smooth|version_045|shuffled_joint|4.407|-3.108|0.476|0.490|
|v9_smooth|network_USA|baseline|5.195|0.000|0.651|0.597|
|v9_smooth|network_USA|perturb_support|5.654|0.459|0.730|0.643|
|v9_smooth|network_USA|seed_support|5.211|0.016|0.716|0.620|
|v9_smooth|network_USA|joint_support|5.188|-0.007|0.708|0.604|
|v9_smooth|network_USA|shuffled_joint|3.660|-1.535|0.519|0.525|

各arm源valid不变性（drop正数=下降；不按目标表现删除扰动）：

|视图模型|扰动|F1下降 pp|原预测保持 %|
|---|---|---:|---:|
|direction_order|original|0.000|100.000|
|direction_order|time095|0.000|100.000|
|direction_order|time105|0.000|100.000|
|direction_order|thin01|0.985|85.262|
|direction_order|tail98|3.717|85.185|
|direction_iat|original|0.000|100.000|
|direction_iat|time095|0.287|98.256|
|direction_iat|time105|0.256|98.627|
|direction_iat|thin01|0.485|89.568|
|direction_iat|tail98|1.831|91.451|
|tam_cnn|original|0.000|100.000|
|tam_cnn|time095|2.524|80.201|
|tam_cnn|time105|3.256|79.846|
|tam_cnn|thin01|0.328|97.531|
|tam_cnn|tail98|0.605|96.312|

预注册判决：

```json
{
  "primary_arm": "direction_iat",
  "primary_pool": "v10_view",
  "source_invariance_passed": true,
  "primary_passed": false,
  "utility_passed": false,
  "pools": [
    {
      "pool": "v10_view",
      "passed": false,
      "slices": [
        {
          "slice": "version_045",
          "rec1_gain_pp": 0.4298353517146569,
          "auc": 0.7102697926262489,
          "auc_gain": 0.08314743041095285,
          "rec1_vs_shuffled_pp": 1.5132381118135727,
          "rec1_vs_v9_pp": -5.702197766494733,
          "passed": false
        },
        {
          "slice": "network_USA",
          "rec1_gain_pp": 0.05564401630245319,
          "auc": 0.7551162720964034,
          "auc_gain": 0.11440806684563132,
          "rec1_vs_shuffled_pp": 2.8649148958743518,
          "rec1_vs_v9_pp": -1.971486243761202,
          "passed": false
        }
      ]
    },
    {
      "pool": "v9_smooth",
      "passed": false,
      "slices": [
        {
          "slice": "version_045",
          "rec1_gain_pp": -0.2168363986786419,
          "auc": 0.6751764002693886,
          "auc_gain": -0.0037929102041047713,
          "rec1_vs_shuffled_pp": 2.890987180695742,
          "rec1_vs_v9_pp": -0.21683639867865967,
          "passed": false
        },
        {
          "slice": "network_USA",
          "rec1_gain_pp": -0.006901251219002091,
          "auc": 0.7078392879855016,
          "auc_gain": 0.056362699373515235,
          "rec1_vs_shuffled_pp": 1.5278128356598466,
          "rec1_vs_v9_pp": -0.006901251219002091,
          "passed": false
        }
      ]
    }
  ],
  "tta_integrated": false,
  "new_training_steps": 0
}
```

完整逐seed、控制arm及源valid排序结果见artifacts/cells.csv与summary.csv，未只挑最好切片。条件AUROC只含真正纠错与损伤；完整改写AUROC另外包含旧错→其他错。对候选池同分阈值不拆开，零改写始终可选。未更新校准温度或风险阈值。跨seed/扰动稳定只提供一致性证据，仍可能稳定地错误；源不变性不能证明目标不变性。sourcevalid各切片行重复用于诊断，不能当作独立重复。

复用9个v10模型的额外源监督预算依然属于方法成本；本轮新增推理5倍/跨seed3倍，不能只报无参数而隐去推理成本。另一候选池v9smooth是历史checkpoint固定预测，raw输出性能未因本实验提高，任何Rec@NAR1收益只是评分前沿诊断。没有访问WTT/AWF，没有额外目标标签训练或改动既有封存。

180个新扰动状态全量重载、45个原始logits精确一致、9模型锁定、360个终端评分状态独立混淆与tie-safe前沿核验通过；输入SHA与历史基线前沿精确一致，结果已封存。

结果解释补充（不修改预注册判决、输入、分数或参数）：

主方向+IAT视图的联合稳定性分数确实包含额外判别信号：纠错vs损伤AUROC在Version/USA从0.627/0.641提高到0.710/0.755，且Rec@NAR1分别比同分布打乱对照高1.513/2.865pp。不能将本轮解释为“稳定性完全无信息”。

但AUROC是全阈值排序能力，不等于低误改区域表现。同NAR1诊断纠错只从1.383%/3.168%提高到1.813%/3.223%，增幅+0.430/+0.056pp，两片均未达预定+.5pp。它仍分别比v9低5.702/1.971pp。完整改写AUROC为0.653/0.680，低于仅纠错vs损伤的AUROC；旧错改成其他错不能忽略。

在v9固定LoRA候选上，联合最差支持度反而使Rec@NAR1从7.515%/5.195%变为7.298%/5.188%。仅扰动支持度在USA升到5.654%（+.459pp），Version降到7.373%（−.142pp），不能单取USA作为成功。现有数据不能唯一归因为取min过于保守、视图能力不足或高分尾部错误集中；本轮只确立全局判别改善尚未转化为所需低风险收益。

主方向+IAT的四种扰动三seed平均源valid F1下降均<=2pp，通过预定源不变性门槛；这并不证明目标语义不变或逐样本输出不变。方向控制的tail98下降3.717pp，TAM控制的时间缩放下降2.524/3.256pp，说明这些探针的有效性与表示有关；均完整保留，未按目标结果删去。三个模型seed来自同一训练集，也不能视作独立语义真值。

处置：保留现有v9候选，不以联合评分替换主评分，不接入Teacher/TTA，不扩大矩阵。本轮为有真实判别信号但无满足风险目标的改进，后续若研究应明确面向低误改区的错误排序，而不能只优化总体AUROC或加大融合权重。
