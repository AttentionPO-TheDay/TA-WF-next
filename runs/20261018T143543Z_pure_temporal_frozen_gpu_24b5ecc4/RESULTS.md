# 纯净源域 TemporalDrift 冻结退化基线

三份纯净源域权重在固定独立 query 上完成9个冻结推断单元。仅为已观察开发证据。

| Day | Mean Accuracy ± SD | Mean Macro-F1 ± SD |
|---|---:|---:|
| day14 | 87.560 ± 0.176% | 87.531 ± 0.154% |
| day90 | 73.209 ± 0.554% | 71.174 ± 0.605% |
| day270 | 62.399 ± 0.582% | 59.230 ± 0.689% |

逐seed指标及预测见 artifacts/scores.json 和 pred_*.npz；输入/代码散列见 artifacts/freeze.json。无TTA、目标标签训练或外部评分。

独立审计 `artifacts/audit.json` 通过：9/9 预测文件、26个冻结输入/代码散列、逐样本argmax与固定102类指标均一致。三个日期的reference均为1024条，query分别为21579、27575、18911条，且与纯净源/valid及reference精确输入隔离。前次中断CPU run留下的两份Day14预测只作事后跨设备检查：seed42有2/21579条argmax不同，seed100为0条；CPU/GPU logits最大绝对差分别约0.0068/0.0083。正式结果全由本次GPU完整重跑产生。
