# 来源

- 代码骨架与模型显式复制自 runs/20261014T084100Z_tent_fixedstats_ln_34b46eca；本run只增加逐样本熵区间门控。
- TENT纯熵实现来源 DequanWang/tent commit e9e926a668d85244c66a6d5c006efbd2b82e83e8；官方代码及MIT许可随run保留。
- 阈值0.20/0.40 nats来自前一run无标签support entropy尺度的取整规则，未使用本run query反馈。
