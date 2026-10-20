# 实现来源

模型及递归依赖显式复制自本项目纯净源训练run；不导入旧项目或其他run运行时。检查点沿用该run三seed纯净源权重，不训练骨干。新runner通过classifier的前向pre-hook取得原模型实际使用的特征，不另写近似前向。

{
  "model.py": {
    "source": "runs/20261018T125337Z_pure_source_backbone_c1d249f6/model.py",
    "sha256": "b6f59add5040dbb9d80f2469cac46412b37c87a2dbd66c0df01fc00e24b67212"
  },
  "base_model.py": {
    "source": "runs/20261018T125337Z_pure_source_backbone_c1d249f6/base_model.py",
    "sha256": "2997108ffa67b71556c54a4df2fbba0b5861926820c6de44e08be9b8d55de289"
  },
  "candidate_base.py": {
    "source": "runs/20261018T125337Z_pure_source_backbone_c1d249f6/candidate_base.py",
    "sha256": "7b9687ad96e9fc4e476986aff6b448519f70e77d112b053ace5796b0a0dd4c65"
  },
  "encoder_base.py": {
    "source": "runs/20261018T125337Z_pure_source_backbone_c1d249f6/encoder_base.py",
    "sha256": "baf3c12794d850b68261ef70131c818ac7be25333d766127725f3f040a114c5a"
  },
  "framework_base.py": {
    "source": "runs/20261018T125337Z_pure_source_backbone_c1d249f6/framework_base.py",
    "sha256": "e530ca3f369cb729f6f9b802bff03fb18ffcec90ebb5ad58db04a4b077a1a148"
  },
  "generator_base.py": {
    "source": "runs/20261018T125337Z_pure_source_backbone_c1d249f6/generator_base.py",
    "sha256": "5b55eee433f26f7ecade39b0a1f26862819aa94c594885bf09f696bd6ad100d4"
  },
  "input_base.py": {
    "source": "runs/20261018T125337Z_pure_source_backbone_c1d249f6/input_base.py",
    "sha256": "1bc1ca5b7aa2b414793171907efba8d5aa5ced701bb2e851d55d511288aaeb62"
  },
  "rf_native.py": {
    "source": "runs/20261018T125337Z_pure_source_backbone_c1d249f6/rf_native.py",
    "sha256": "8ffd968ba4a750fb66edda53e8146ed367b9a62e1e1330416b3eb600b4060617"
  },
  "view_base.py": {
    "source": "runs/20261018T125337Z_pure_source_backbone_c1d249f6/view_base.py",
    "sha256": "aa41abb7d1eec425dc367b3b6a4e85fdfa60d35bcdbad6123754f2a63a0c44dd"
  }
}
