# 实验结果

已完成：三 seed 只读 hybrid checkpoint、source2040/valid510、full/off/bias_only 固定推理干预；无训练，无 GPU。主推理 9.77 秒，CPU 单线程。全部结果为已观察 TemporalDrift 的开发诊断。

## 核心结果

| 角色 | 固定条件 | accuracy 均值 | macro-F1 均值 | 相对 full 预测变化率 |
|---|---|---:|---:|---:|
| source | full | 39.15% | 36.95% | 0 |
| source | off（整个连续输出归零） | 17.75% | 14.12% | 67.21% |
| source | bias_only（保留偏置） | 35.96% | 33.72% | 21.03% |
| valid | full | 22.35% | 19.63% | 0 |
| valid | off | 11.70% | 7.94% | 67.39% |
| valid | bias_only | 19.74% | 16.98% | 22.42% |

valid 上 off 的 F1 下降 11.70 pp；bias_only 的 F1 下降 2.66 pp。两项干预均在三 seed 降低 F1，source 也三 seed 降低。删除输入相关部分并未改善泛化，因此这次检查不支持“训练好的 hybrid 中，连续长度只是有害干扰，关掉即可改善”的简单解释。

| seed | full valid F1 | off valid F1 | bias_only valid F1 | bucket RMS | continuous RMS | bias RMS | variable RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1729 | 19.21% | 4.70% | 15.45% | 1.0311 | 0.7251 | 0.7229 | 0.0992 |
| 3407 | 22.46% | 14.14% | 21.59% | 1.1462 | 0.5603 | 0.5602 | 0.0715 |
| 2026 | 17.23% | 4.97% | 13.88% | 0.9004 | 0.6591 | 0.6121 | 0.1007 |

RMS 仅统计非 padding token，每个分支按自己的元素数归一化。continuous=bias+variable，因交叉项存在，RMS 不能相加，也不能直接当作方差解释率。输入相关 variable 尺度仅为 bucket 的约 6.2–11.2%；整个 continuous 尺度为 bucket 的约 48.9–73.2%，且 bias 尺度远大于 variable。没有出现连续分支整体幅度压过 bucket 的现象，但小幅变量也足以改变约22%的valid预测，不能凭尺度断言信息无用。

## 解释边界

1. 当前 checkpoint 已在两分支共同存在时训练并选择；突然删除分支会产生训练外激活分布。off 尤其混淆了偏置移除与精确长度信息移除，不是重新训练 bucket-only 的替代品。
2. bias_only 比 off 温和，full 又优于 bias_only，支持这些固定模型使用了输入相关长度分量；不证明该分量提供 bucket 基线之上的独立信息，亦不能证明门控一定有效。
3. 先前“直接相加干扰导致 F1 下降”仍是未证实假设。现在更应通过同预算独立训练检验融合/优化，而非据此删掉精确分支。本轮不据valid搜索干预系数。
4. 无未来评分、无漂移机制评价、无训练期蒸馏验证；源域和valid均为此前已观察样本，无独立确认资格主张。

## 核验与来源

- 显式只读复用 `20260922T062732Z_cpu_token_typed_encoding_df081d9e/train.py` 的 TokenNet、该run hybrid三个checkpoint与source/valid缓存。未加载旧项目checkpoint或代码，只使用旧虚拟环境依赖。
- full 三 seed 的全部 1530 条 valid 预测逐条精确复现历史保存结果；源/验证方向hash隔离通过。
- 18 指标、22950 条预测由独立 `verify.py` 逐类重算；输入/代码/配置/output SHA256 封存检查通过。详见 artifacts/verification.json。
- 输入/代码版本见 artifacts/input_seal.json，结果封存见 artifacts/output_seal.json；统计全量结果见 artifacts/metrics.json、scales.json，预测见 predictions.npz。
