# wide+BN 全局统计支路实验结果

12 项新训练和 3 项历史 wide+BN 基线均完成。监督脚本在训练结束后的候选汇总阶段因引用不存在的 `zero_add - baseline` 等比较键报错；已依据保存的 report、history、预测和冻结 hash 恢复汇总，没有重训。

| 条件 | 验证准确率均值 | Macro-F1 均值 | 固定三 seed 投票 | ≥90% |
|---|---:|---:|---:|---|
| baseline | 89.150% | 88.944% | 91.569% | False |
| zero_add | 87.712% | 87.531% | 90.588% | False |
| stats_add | 86.144% | 85.984% | 89.216% | False |
| zero_gate | 88.170% | 87.950% | 90.196% | False |
| stats_gate | 86.797% | 86.636% | 90.392% | False |

| 条件 | 21729 | 23407 | 22026 |
|---|---:|---:|---:|
| baseline | 89.020% | 89.412% | 89.020% |
| zero_add | 87.451% | 88.235% | 87.451% |
| stats_add | 86.863% | 86.471% | 85.098% |
| zero_gate | 87.843% | 90.000% | 86.667% |
| stats_gate | 87.255% | 86.667% | 86.471% |

## 比较结果

```json
{
  "stats_add - zero_add": {
    "accuracy_mean_delta_pp": -1.5686274509803904,
    "accuracy_per_seed_pp": [
      -0.588235294117645,
      -1.764705882352935,
      -2.352941176470591
    ],
    "macro_f1_mean_delta_pp": -1.5465199288728582,
    "pass": false
  },
  "stats_gate - zero_gate": {
    "accuracy_mean_delta_pp": -1.3725490196078418,
    "accuracy_per_seed_pp": [
      -0.588235294117645,
      -3.3333333333333326,
      -0.19607843137254832
    ],
    "macro_f1_mean_delta_pp": -1.3144027359713575,
    "pass": false
  },
  "stats_gate - stats_add": {
    "accuracy_mean_delta_pp": 0.6535947712418315,
    "accuracy_per_seed_pp": [
      0.39215686274509665,
      0.19607843137254832,
      1.3725490196078494
    ],
    "macro_f1_mean_delta_pp": 0.6514654553870198,
    "pass": false
  },
  "zero_add - baseline": {
    "accuracy_mean_delta_pp": -1.4379084967320246,
    "accuracy_per_seed_pp": [
      -1.5686274509803866,
      -1.176470588235301,
      -1.5686274509803866
    ],
    "macro_f1_mean_delta_pp": -1.4128373442098976,
    "pass": false
  },
  "stats_add - baseline": {
    "accuracy_mean_delta_pp": -3.0065359477124147,
    "accuracy_per_seed_pp": [
      -2.1568627450980316,
      -2.941176470588236,
      -3.9215686274509776
    ],
    "macro_f1_mean_delta_pp": -2.959357273082756,
    "pass": false
  },
  "zero_gate - baseline": {
    "accuracy_mean_delta_pp": -0.9803921568627416,
    "accuracy_per_seed_pp": [
      -1.17647058823529,
      0.588235294117645,
      -2.35294117647058
    ],
    "macro_f1_mean_delta_pp": -0.9934890817243783,
    "pass": false
  },
  "stats_gate - baseline": {
    "accuracy_mean_delta_pp": -2.352941176470584,
    "accuracy_per_seed_pp": [
      -1.764705882352935,
      -2.7450980392156876,
      -2.549019607843128
    ],
    "macro_f1_mean_delta_pp": -2.3078918176957357,
    "pass": false
  }
}
```

## 核验

冻结输入 hash、60 组 source/valid best/last 预测指标、15 组 history 与选模步数、15 组初始化状态和 15 条 index stream 均已核验；另对 30 组保存 logits/predictions 复算，并重新加载 15 个最佳 checkpoint 在 source/valid 上完成 30 组推理复核。12 项新任务均完成 12800 步，valid 只用于预定 checkpoint 选择与评价。

真实统计支路相对 zero 控制下降：`stats_add` 低于 `zero_add`，`stats_gate` 低于 `zero_gate`；四个新条件均低于历史 wide+BN 89.150%，没有统计支路候选通过门槛。固定三 seed 投票的历史 baseline 仍为 91.569%，新条件投票分别为 `zero_add=90.588%`, `stats_add=89.216%`, `zero_gate=90.196%`, `stats_gate=90.392%`。

本实验说明在当前 TAM 输入、遮挡规则和分类配方下，增加全局统计残差会损害源期 valid 泛化，不能据此否定所有统计特征或 RF 的优势。未来/WTT-Time/AWF/适应关闭；该 valid 已参与多轮开发，不是独立确认。
