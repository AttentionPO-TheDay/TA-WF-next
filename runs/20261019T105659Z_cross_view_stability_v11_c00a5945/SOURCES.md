# 显式来源
- model.py逐字复制v10最小独立CNN及输入features；构造/权重与v10严格匹配，无sys.path或旧项目隐式导入。
- v10: 20261019T100749Z_independent_view_probe_v10_9b462cd2，9checkpoint、3温度汇总、45原始logit状态。额外source监督预算来自该轮，本轮仅重用。
- v9: 20261019T094400Z_progressive_view_v9_9ee04ba2，冻结基线、smooth候选、sourcevalid对照及前沿，历史复用而非新重复。
- source和诊断缓存分别来自pure_source_backbone和multifactor_diagnostic_v1；配置经configs/datasets.json解析数据根并核对已有SHA与行角色。
- runtime/perturb/worker/analysis/controller为本轮新实现。基于概率差的鲁棒排序不是已验证的误改概率，稳健性假设须由源valid诊断约束。
