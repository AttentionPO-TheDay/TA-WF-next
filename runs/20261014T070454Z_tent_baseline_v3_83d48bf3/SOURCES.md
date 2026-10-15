# 来源与改动

- 官方TENT：https://github.com/DequanWang/tent ，commit `e9e926a668d85244c66a6d5c006efbd2b82e83e8`。保存official_tent.py、TENT_LICENSE、TENT_README.md。
- tent_bn1d.py仅将3处nn.BatchNorm2d判断扩展至BN1d/BN2d；不新增LayerNorm更新。
- 模型及数据helper显式复制本项目run `20261014T003143Z_tta_subspace_restriction_f86ece7f`；不导入旧项目。
- 官方单输入接口通过InputAdapter接收(timestamp,tam)元组；使用本项目forward_with_features得到logits。support-only末步评分、eval dropout与官方在线评价的差异在PLAN/config声明。
- Soft Pacing公共代码已撤下；历史run原样保留。清洁参考仍为no_pseudo_fixed（73.504%/3.963%），没有新增锚定或改变其边缘项。仅回滚公共模块不重写历史结果。
