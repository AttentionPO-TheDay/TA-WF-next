# 实验结果

完成共享stored-order方向run生成器原型，不训练、不预测、不性能评分。实现位于src/ta_wf_next/burst_tokens.py，定义与限制见同目录BURST_TOKENS.md。

精确view保留run方向/数量，可恢复有效方向前缀；coarse view独立输出log2长度bin、前后bin差和边界标记。没有精确长度旁路、时间字段、类别或漂移原因输入；没有排序、负时间截断或学习参数。粗化只是固定工程候选，不称创新或抗漂移结论。

## 验证

- 新增12个unittest通过：长度0–9共1023条二进制方向序列穷举可逆、padding/内部零拒绝、非法值、空/短trace、截断、预算外禁止读取、timestamp符号路径、bin边界及粗化碰撞。
- 全项目22项：21通过、1可选旧源码迁移测试跳过。包含已有模型的合成梯度测试，但没有训练循环、optimizer更新或真实数据模型前向。
- 真实数据固定等距128行/文件，Version048/train、Network/train、Behavior/subpage共384行：round-trip错误0，signed timestamp与direction接口差异0，bin范围断言全部通过。只消费各行前5000，未读y/URL，未载checkpoint。
- artifacts/smoke.json保存样本索引、观测数/token数、结束原因、精确/粗化例子及实现/测试/检查脚本SHA256。

## 边界

精确还原针对有效方向，不包括原始timestamp、padding和预算外数据。首末标记表示可能删失，不判断真实网络burst边界。内部零检查仅限预算内。None邻居表示缺失，不是数值0；模型接入时需mask。

coarse-only不能同时喂精确runs或observed_count。固定对数bin会合并4和7，却分开7和8，不能承诺任意扰动稳定。相同接口在三场景工作不等于跨场景性能泛化。

没有新增embedding/模型adapter/collator，没有接入现有checkpoint或训练任务。下一步应冻结开发/留出划分和最小表示对照，再决定如何训练；不自动扩大到性能实验。WTT-Time/AWF未打开，原始数据未修改。
