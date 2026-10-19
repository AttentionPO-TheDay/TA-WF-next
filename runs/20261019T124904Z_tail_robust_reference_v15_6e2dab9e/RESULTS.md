# A/B/C低风险候选清洗 v15

组合主臂ABC通过：False。全部A/B/C单独及组合均预注册；不以最佳分支替换ABC。固定v9候选，零神经网络训练/新前向，复用缓存。

## 终端解释补充

以下为全部拟合/评分/重载结束后的解释修订，未修改模型、特征、归一化、阈值、预测或主判据。修订前报告及原delivery manifest保留，最终清单以final_verification.json为准。

1. A/B/C全部及组合均完成，八组无一通过预注册主判据。与v12同实际损伤预算时，A的Version/USA Recovery变化为-0.0684/-0.1237pp，B为+0.0230/-0.1022pp，C为+0.0377/-0.0770pp，ABC为-0.0652/-0.1237pp（精确值见registered_decisions.json）。没有两片一致改进，不能凭NAR≤1%操作点的微小均值收益升级方法。
2. A的源valid融合pAUC相对ce提高+1.393/+1.481pp，但真实切片变化+0.055/−0.811pp；逐扰动留出均值也提高+0.698/+1.353pp。说明本轮“合成扰动组内/组间可学”的收益仍不能稳定迁移到真实低风险尾部。A是局部hard-negative pair代理，并不等于精确优化完整目标pAUC，其失败不能证伪所有pAUC优化算法。
3. 分支单独评分已与融合评分分开。A在sourcevalid/target的独立低风险pAUC都低于ce分支，因此不能把失败仅归咎于最终融合；但分支为残差信号，独立排序弱也不是信息完全无用的证明。本轮未据此搜索新融合系数。
4. C六状态均通过覆盖门槛，OOF域分类AUROC为0.524–0.677，raw ESS/N为0.484–0.936，reference支持比例全为1。30个域分类fold和全部源OOF权重重载复算误差<1e-12；故当前C效果弱不能解释为权重崩塌或未真正执行。但这只是三维边缘证据空间的覆盖，不能证明真实网站条件关系保持，不能否定更丰富特征或其他迁移方法。
5. 源valid阈值直接迁移时，A在Version的Recovery提高到3.823%，同时NAR提高到0.240%、W提高到365.333；v12分别为1.985%、0.134%、153.667。这是操作点变化，不是同风险清洗成功。USA同预算前沿同样未改善，不能只报更多纠错。
6. 新的描述性线索：固定2448源train clean样本，三seed frozen accuracy均100%，R均0；816源valid clean accuracy为87.13%–88.48%，R为8–16。说明源train元监督依赖人为损坏后才产生的纠错事件，与模型外真实错误存在明显构成差异。源train/valid输入内容SHA256交叉为0；该差异不是行号重合导致的数据泄漏。此观察建议后续优先检验骨干模型外元训练/更真实的错误监督，但不能单独认定骨干记忆是迁移失败的因果根源。

保留v12，不部署、不扩矩阵，不自动启动新骨干交叉拟合训练。当前结果仅Version045/USA已观察诊断开发证据，不能推广为全部时序漂移结论。

## 三seed诊断前沿

pAUC/.01为低风险区间平均候选TPR，FPR分母是全部frozen正确者；不是McClish标准化。阈值由query标签确定的前沿仅诊断，不部署。

|切片|方法|状态|pAUC/.01 %|Recovery@NAR1 %|W|W/R|
|---|---|---:|---:|---:|---:|---:|
|version_045|v9_margin|3|42.846|7.515|914.333|1.077|
|version_045|v12_primary|3|52.331|7.866|1000.667|1.128|
|version_045|v14_primary|3|52.404|7.949|979.333|1.094|
|version_045|ce|3|52.404|7.949|979.333|1.094|
|version_045|A|3|52.459|7.855|1004.333|1.138|
|version_045|B|3|52.355|7.958|982.000|1.096|
|version_045|AB|3|52.455|7.831|1010.667|1.148|
|version_045|C|3|52.368|7.949|976.667|1.091|
|version_045|AC|3|52.505|7.858|1006.667|1.140|
|version_045|BC|3|52.311|7.943|978.667|1.094|
|version_045|ABC|3|52.474|7.843|1010.667|1.147|
|network_USA|v9_margin|3|40.561|5.195|102.667|1.567|
|network_USA|v12_primary|3|60.123|6.254|118.667|1.532|
|network_USA|v14_primary|3|59.697|6.310|119.667|1.526|
|network_USA|ce|3|59.697|6.310|119.667|1.526|
|network_USA|A|3|58.886|6.131|118.333|1.557|
|network_USA|B|3|59.730|6.285|119.667|1.530|
|network_USA|AB|3|58.856|6.132|118.000|1.552|
|network_USA|C|3|59.697|6.310|119.667|1.526|
|network_USA|AC|3|58.963|6.157|118.333|1.553|
|network_USA|BC|3|59.799|6.310|119.667|1.526|
|network_USA|ABC|3|58.959|6.157|119.000|1.560|

## 同v12实际损伤预算与同纠错匹配

|切片|方法|同损伤Recovery增量pp|不下降seed|同纠错可行seed|同纠错D|同纠错W|
|---|---|---:|---:|---:|---:|---:|
|version_045|ce|0.038|2|2|NA|NA|
|version_045|A|-0.068|1|1|NA|NA|
|version_045|B|0.023|2|2|NA|NA|
|version_045|AB|-0.059|1|1|NA|NA|
|version_045|C|0.038|2|2|NA|NA|
|version_045|AC|-0.063|0|1|NA|NA|
|version_045|BC|0.017|2|2|NA|NA|
|version_045|ABC|-0.065|1|1|NA|NA|
|network_USA|ce|-0.077|0|1|NA|NA|
|network_USA|A|-0.124|1|1|NA|NA|
|network_USA|B|-0.102|0|1|NA|NA|
|network_USA|AB|-0.149|1|1|NA|NA|
|network_USA|C|-0.077|0|1|NA|NA|
|network_USA|AC|-0.124|1|1|NA|NA|
|network_USA|BC|-0.077|0|1|NA|NA|
|network_USA|ABC|-0.124|1|1|NA|NA|

## 源阈值直接迁移

阈值统一按源valid的R-D-.25W选取，未使用query标签；源校准不保证目标风险。低NAR也可能来自该效用主动选择较少改写。

|切片|方法|平均NAR %|最大NAR %|Recovery %|W|
|---|---|---:|---:|---:|---:|
|version_045|v9_margin|0.000|0.000|0.000|0.000|
|version_045|v12_primary|0.134|0.362|1.985|153.667|
|version_045|v14_primary|0.153|0.277|2.671|223.333|
|version_045|ce|0.153|0.277|2.671|223.333|
|version_045|A|0.240|0.431|3.823|365.333|
|version_045|B|0.153|0.277|2.674|223.000|
|version_045|AB|0.245|0.462|3.894|373.000|
|version_045|C|0.150|0.277|2.629|219.333|
|version_045|AC|0.243|0.431|3.883|372.000|
|version_045|BC|0.153|0.277|2.665|223.000|
|version_045|ABC|0.245|0.462|3.869|368.000|
|network_USA|v9_margin|0.000|0.000|0.000|0.000|
|network_USA|v12_primary|0.182|0.411|2.336|31.667|
|network_USA|v14_primary|0.136|0.274|2.207|27.000|
|network_USA|ce|0.136|0.274|2.207|27.000|
|network_USA|A|0.182|0.342|2.690|38.000|
|network_USA|B|0.159|0.274|2.255|28.333|
|network_USA|AB|0.182|0.342|2.664|38.000|
|network_USA|C|0.045|0.069|1.904|23.000|
|network_USA|AC|0.159|0.274|2.336|32.000|
|network_USA|BC|0.136|0.274|2.154|27.000|
|network_USA|ABC|0.182|0.342|2.260|31.333|

## reference覆盖与训练迁移检查

```json
[
  {
    "seed": 42,
    "slice": "version_045",
    "feasible": true,
    "source_changed": 1380,
    "reference_changed": 130,
    "raw_ess_fraction": 0.9051993303953255,
    "clipped_ess_fraction": 0.9052323325086642,
    "reference_support_fraction": 1.0,
    "domain_oof_auc": 0.5238182831661093,
    "clip_low_fraction": 0.0014492753623188406,
    "clip_high_fraction": 0.0,
    "models": 6
  },
  {
    "seed": 42,
    "slice": "network_USA",
    "feasible": true,
    "source_changed": 1116,
    "reference_changed": 103,
    "raw_ess_fraction": 0.9053852641880183,
    "clipped_ess_fraction": 0.9053852641880187,
    "reference_support_fraction": 1.0,
    "domain_oof_auc": 0.5784615652294951,
    "clip_low_fraction": 0.0,
    "clip_high_fraction": 0.0,
    "models": 6
  },
  {
    "seed": 100,
    "slice": "version_045",
    "feasible": true,
    "source_changed": 1123,
    "reference_changed": 122,
    "raw_ess_fraction": 0.4835041191647056,
    "clipped_ess_fraction": 0.5062362239810386,
    "reference_support_fraction": 1.0,
    "domain_oof_auc": 0.6771382275228823,
    "clip_low_fraction": 0.0,
    "clip_high_fraction": 0.003561887800534283,
    "models": 6
  },
  {
    "seed": 100,
    "slice": "network_USA",
    "feasible": true,
    "source_changed": 1028,
    "reference_changed": 98,
    "raw_ess_fraction": 0.8630010853850636,
    "clipped_ess_fraction": 0.8630010853850633,
    "reference_support_fraction": 1.0,
    "domain_oof_auc": 0.571478202175812,
    "clip_low_fraction": 0.0,
    "clip_high_fraction": 0.0,
    "models": 6
  },
  {
    "seed": 2026,
    "slice": "version_045",
    "feasible": true,
    "source_changed": 1100,
    "reference_changed": 127,
    "raw_ess_fraction": 0.935945169039501,
    "clipped_ess_fraction": 0.9359451690395004,
    "reference_support_fraction": 1.0,
    "domain_oof_auc": 0.52571224051539,
    "clip_low_fraction": 0.0,
    "clip_high_fraction": 0.0,
    "models": 6
  },
  {
    "seed": 2026,
    "slice": "network_USA",
    "feasible": true,
    "source_changed": 902,
    "reference_changed": 112,
    "raw_ess_fraction": 0.7787500507171509,
    "clipped_ess_fraction": 0.7787500507171506,
    "reference_support_fraction": 1.0,
    "domain_oof_auc": 0.620575308837504,
    "clip_low_fraction": 0.0,
    "clip_high_fraction": 0.0,
    "models": 6
  }
]
```

LOPO逐扰动/逐seed结果见artifacts/lopo.csv；sourcevalid与query的分支单独/融合结果见fusion_source.json和fusion_target.json。目标结果未用于挑臂或重训。原clean组几乎没有改写，可能无纠错候选；对应pAUC未定义，保留null，不填零。

## 判决

```json
{
  "primary": "ABC",
  "primary_passed": false,
  "single_method_tests": [
    "A",
    "B",
    "C"
  ],
  "all_arms": [
    {
      "arm": "ce",
      "passed": false,
      "slices": [
        {
          "slice": "version_045",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": 0.07338630370081756,
          "rec1_gain_pp": 0.08263576643177672,
          "same_damage_rec_gain_pp": 0.03768386496634385,
          "nondecreasing_same_damage_seeds": 2,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        },
        {
          "slice": "network_USA",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": -0.4259322888231831,
          "rec1_gain_pp": 0.055605812784673136,
          "same_damage_rec_gain_pp": -0.07698501193025703,
          "nondecreasing_same_damage_seeds": 0,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        }
      ]
    },
    {
      "arm": "A",
      "passed": false,
      "slices": [
        {
          "slice": "version_045",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": 0.12877540454718783,
          "rec1_gain_pp": -0.011400407226470899,
          "same_damage_rec_gain_pp": -0.06837853211777156,
          "nondecreasing_same_damage_seeds": 1,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        },
        {
          "slice": "network_USA",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": -1.2366647316635393,
          "rec1_gain_pp": -0.12365493984614409,
          "same_damage_rec_gain_pp": -0.12365493984614409,
          "nondecreasing_same_damage_seeds": 1,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        }
      ]
    },
    {
      "arm": "B",
      "passed": false,
      "slices": [
        {
          "slice": "version_045",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": 0.023970910999160377,
          "rec1_gain_pp": 0.09193689814520206,
          "same_damage_rec_gain_pp": 0.023036133634814215,
          "nondecreasing_same_damage_seeds": 2,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        },
        {
          "slice": "network_USA",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": -0.3930118477409761,
          "rec1_gain_pp": 0.03035328753214772,
          "same_damage_rec_gain_pp": -0.10223753718278175,
          "nondecreasing_same_damage_seeds": 0,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        }
      ]
    },
    {
      "arm": "AB",
      "passed": false,
      "slices": [
        {
          "slice": "version_045",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": 0.12473078029913154,
          "rec1_gain_pp": -0.03529069571879817,
          "same_damage_rec_gain_pp": -0.05918098421112544,
          "nondecreasing_same_damage_seeds": 1,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        },
        {
          "slice": "network_USA",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": -1.2663789892857102,
          "rec1_gain_pp": -0.12238930015568486,
          "same_damage_rec_gain_pp": -0.1489074650986702,
          "nondecreasing_same_damage_seeds": 1,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        }
      ]
    },
    {
      "arm": "C",
      "passed": false,
      "slices": [
        {
          "slice": "version_045",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": 0.036892862810256055,
          "rec1_gain_pp": 0.08263576643177672,
          "same_damage_rec_gain_pp": 0.03768386496634385,
          "nondecreasing_same_damage_seeds": 2,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        },
        {
          "slice": "network_USA",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": -0.4259322888231942,
          "rec1_gain_pp": 0.055605812784673136,
          "same_damage_rec_gain_pp": -0.07698501193025703,
          "nondecreasing_same_damage_seeds": 0,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        }
      ]
    },
    {
      "arm": "AC",
      "passed": false,
      "slices": [
        {
          "slice": "version_045",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": 0.1739988625991895,
          "rec1_gain_pp": -0.008739849619057716,
          "same_damage_rec_gain_pp": -0.06259807242167265,
          "nondecreasing_same_damage_seeds": 0,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        },
        {
          "slice": "network_USA",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": -1.1602371827189706,
          "rec1_gain_pp": -0.09713677490315806,
          "same_damage_rec_gain_pp": -0.12365493984614409,
          "nondecreasing_same_damage_seeds": 1,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        }
      ]
    },
    {
      "arm": "BC",
      "passed": false,
      "slices": [
        {
          "slice": "version_045",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": -0.019890391981891575,
          "rec1_gain_pp": 0.07710152405098897,
          "same_damage_rec_gain_pp": 0.017171614982811056,
          "nondecreasing_same_damage_seeds": 2,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        },
        {
          "slice": "network_USA",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": -0.3238780555847254,
          "rec1_gain_pp": 0.055605812784673136,
          "same_damage_rec_gain_pp": -0.07698501193025703,
          "nondecreasing_same_damage_seeds": 0,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        }
      ]
    },
    {
      "arm": "ABC",
      "passed": false,
      "slices": [
        {
          "slice": "version_045",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": 0.14294271508936696,
          "rec1_gain_pp": -0.023219463038778865,
          "same_damage_rec_gain_pp": -0.06521362077493431,
          "nondecreasing_same_damage_seeds": 1,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        },
        {
          "slice": "network_USA",
          "passed": false,
          "complete": true,
          "pauc_gain_pp": -1.1636104483546794,
          "rec1_gain_pp": -0.09713677490315806,
          "same_damage_rec_gain_pp": -0.12365493984614409,
          "nondecreasing_same_damage_seeds": 1,
          "same_correction_feasible": false,
          "same_correction_W_ratio": null,
          "same_correction_D_change": null
        }
      ]
    }
  ],
  "integrated": false,
  "target_role": "observed_diagnostic"
}
```

## 局限与完成检查

已有三源seed，共享query样本；六状态不是六个独立数据集。微小样本计数改变不能视为显著性证据。五扰动共享原trace，非独立增样；源骨干见过train，元训练不是骨干模型外预测。LOPO仅扰动组泛化诊断。C密度比依赖边缘覆盖和条件关系不变假设，非真实纠错概率或风险保证；OOF按trace分组只防域分类器与元训练内部泄漏，不能解除骨干记忆。

已观察Version045/USA沿用内容支持标签对应假设，非官方类别映射；不是独立确认。WTT/AWF关闭。此次目标表现不返调已冻结方法，未部署或扩大矩阵。完整重载审计见reload_audit.json；散列见freeze.json/source_fit_freeze.json/delivery_audit.json。
