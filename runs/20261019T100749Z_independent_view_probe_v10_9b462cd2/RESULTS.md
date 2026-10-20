# 独立视图能力与共错互补性 v10

完成9个独立纯源模型；主候选direction_order门槛：False。本轮只做能力探针，没有接入TTA。

所有结果为两个已观察目标切片的诊断开发；增加了源监督模型预算，不能据此宣称同预算优于原TTA。

|视图|源valid F1均值±SD|源train accuracy均值|
|---|---:|---:|
|direction_order|68.757 ± 0.518|95.634|
|direction_iat|80.818 ± 0.568|98.573|
|tam_cnn|78.461 ± 0.072|99.793|

下表均为三训练种子均值，单位%或pp；同NAR列来自query标签诊断阈值，不可部署。共错指冻结head、源原型及双bank投票指向同一错误类别的固定子集。

|切片|视图|输出|F1|ΔF1 pp|NAR|Recovery|共错纠正|共错仍同错|Rec@NAR1|较v9 pp|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|version_045|direction_order|standalone|27.996|-22.480|51.438|11.670|10.516|23.456|0.847|-6.668|
|version_045|direction_order|fixed_equal_fusion|48.218|-2.258|10.290|7.668|4.886|76.205|—|—|
|version_045|direction_iat|standalone|34.966|-15.510|39.138|12.716|10.768|32.489|1.383|-6.132|
|version_045|direction_iat|fixed_equal_fusion|48.065|-2.411|11.731|9.505|6.133|70.781|—|—|
|version_045|tam_cnn|standalone|32.268|-18.207|43.981|10.088|8.670|35.286|0.503|-7.012|
|version_045|tam_cnn|fixed_equal_fusion|48.020|-2.456|11.099|7.047|4.180|77.883|—|—|
|version_045|v9_pca|standalone|6.952|-43.524|88.319|3.790|3.986|10.824|—|—|
|network_USA|direction_order|standalone|31.522|-18.881|49.593|14.894|12.859|18.824|1.816|-3.379|
|network_USA|direction_order|fixed_equal_fusion|49.514|-0.889|9.839|10.138|5.860|73.549|—|—|
|network_USA|direction_iat|standalone|42.192|-8.211|33.580|20.959|16.912|26.516|3.168|-2.027|
|network_USA|direction_iat|fixed_equal_fusion|51.430|1.027|11.050|15.875|10.365|65.045|—|—|
|network_USA|tam_cnn|standalone|37.773|-12.630|37.144|14.968|12.924|27.229|1.310|-3.885|
|network_USA|tam_cnn|fixed_equal_fusion|49.370|-1.033|10.214|10.498|6.600|73.261|—|—|
|network_USA|v9_pca|standalone|6.008|-44.395|88.650|3.931|3.853|9.673|—|—|

预注册判决（主、次、对照分别报告，不互换）：

```json
[
  {
    "arm": "direction_order",
    "source_semantic_pass": true,
    "slices": [
      {
        "slice": "version_045",
        "common_wrong_pass": true,
        "ranking_auc_pass": false,
        "frontier_pass": false,
        "passed": false
      },
      {
        "slice": "network_USA",
        "common_wrong_pass": true,
        "ranking_auc_pass": true,
        "frontier_pass": false,
        "passed": false
      }
    ],
    "passed": false
  },
  {
    "arm": "direction_iat",
    "source_semantic_pass": true,
    "slices": [
      {
        "slice": "version_045",
        "common_wrong_pass": true,
        "ranking_auc_pass": false,
        "frontier_pass": false,
        "passed": false
      },
      {
        "slice": "network_USA",
        "common_wrong_pass": true,
        "ranking_auc_pass": false,
        "frontier_pass": false,
        "passed": false
      }
    ],
    "passed": false
  },
  {
    "arm": "tam_cnn",
    "source_semantic_pass": true,
    "slices": [
      {
        "slice": "version_045",
        "common_wrong_pass": false,
        "ranking_auc_pass": false,
        "frontier_pass": false,
        "passed": false
      },
      {
        "slice": "network_USA",
        "common_wrong_pass": true,
        "ranking_auc_pass": false,
        "frontier_pass": false,
        "passed": false
      }
    ],
    "passed": false
  }
]
```

条件AUROC只比较改对与改坏两组，排除旧错→其他错；完整分歧精度包含这类无效改写，详见cells.csv。源valid仅选checkpoint与温度；所有9训练完成之后才读取query评分。GroupNorm与独立逐样本变换不拟合query统计。固定等权概率融合不是风险保证。额外源训练与原主模型的不同训练日程/结构，意味着这是能力与互补性对照，而不是严格的同算力模型排名。

输出方向不同不等于获得正确的新证据；只有共同错误中的真实纠正与受控损伤前沿提升才支持后续整合。主候选未过门槛则不扩展；次候选通过也必须单独立项才接入适应。zero timestamp沿用缓存padding，无法恢复真实t=0包。方向IAT负时间差截零，比例见input_diagnostics.json。没有新的WTT/AWF访问，未修改既有封存。

9个checkpoint、源valid与source探针，以及36个reference/query状态完整独立重载通过；逐seed初始权重、训练行顺序、选模日程、独立混淆指标、冻结输入SHA核验通过。

解释补充（完成预注册评价之后加入；不改变方法、选模或门槛）：

本轮主候选与两个次级比较均未通过完整门槛。三个视图源valid F1为68.757/80.818/78.461%，强于v9粗PCA的19.105%，但它们的独立目标F1均低于冻结主模型。新增源训练不能被隐去作为“免费适应”。

最有互补迹象的方向+IAT视图，在Version/USA固定共错集合中真正纠正10.768%/16.912%；同时56.743%/56.571%变为其他错误。不同错误多于真正纠正，因此低同错率不能解释为成功解耦。其直接替换NAR为39.138%/33.580%；固定等权融合USA虽ΔF1+1.027pp，NAR仍11.050%，Version则ΔF1−2.411pp。不能把USA单点增幅当作方法成功。

对主分数view probability margin，方向+IAT的改对vs改坏AUROC为0.627/0.641；相同NAR<=1%的诊断纠错仅1.383%/3.168%，低于v9smooth的7.515%/5.195%。有一些可用的新信息，但当前表征与评分组合无法以相近风险利用它，不接入Teacher或TTA。

进一步仅用已保存sourcevalid logits作解释性复算（posthoc_source_ranking.py/json；没有训练、阈值选择或目标拟合）：三个视图同一主分数在源valid的纠错vs损伤AUROC也只有0.641/0.663/0.662。因而不能把失败完全归因为“源valid到目标的阈值迁移”；主评分在源端就只有有限判别力。AUROC排除旧错→其他错的样本，完整分歧精度更低，二者不可混称。未来候选应先检验每样本跨视图稳定性/不变性是否提供额外判别信号，再讨论如何利用，而非扩大融合权重或继续训练弱视图。

只读方向投影身份检查projected_identity_inspection.json显示：source/valid投影交集为0；Version与USA的query对source/valid/reference均无精确方向投影交集，query内投影亦无重复。该检查没有标签/拟合，不赋予独立确认资格。首次预检确定性池化失败及等价修复在repair_history.json，失败前没有模型训练。
