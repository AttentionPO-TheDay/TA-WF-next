# Stored-order burst tokens v0

实现：burst_tokens.py；无时间属性、可学习参数、类别标签或条件路由。不代表ET-BERT字节token、TLS record或网页资源。

`encode_directions(values, budget=5000, input_kind='direction')` 接受单条序列。`signed_timestamp`模式仅使用符号，不排序时间。direction模式只接受-1/0/+1。最多消费budget项，不检查budget外任何值；因此不能发现预算外的内部零或推断下一观测方向。预算内非有限值、padding后非零拒绝。

输出Encoding用于审计：精确runs、有效观测数、预算和结束原因。`decode()`只恢复有效方向前缀，不恢复时间、padding长度或预算外内容。`coarse()`独立输出方向、floor(log2 count)、邻居bin差及边界标记。邻居缺失用None，未来模型接入需显式mask，不以0混淆相等关系。

粗化bin k覆盖[2^k,2^(k+1)-1]，不是训练选择的词表或最佳量化。所有场景共用规则，序列位置由token排列保留。长度4和7会被合并，而7和8会跨bin；不承诺对所有小扰动连续或不变。邻居bin差完全由粗化长度派生，不增加信息，只显式表达关系。模型如宣称coarse-only，只能使用coarse返回值，不能同时读Encoding.observed_count或精确runs，否则重新引入细节旁路。

首末run的boundary标记始终表示“触及可见前缀边界，真实延续未知”；不是断言run确实被截断。padding结束也不能证明网页物理传输结束。单run两个标记都为true，空输入无token。右边界token保留并标记，不默默删除。

示例：+4,-6,+2,-3对应长度bin 2,2,1,1。精确RLE可逆；粗化主动舍弃精确长度。token数量、方向、粗化规模及次序仍然保留，不应宣称所有时间敏感信息已被过滤。

目前没有embedding层、batch padding/collator、模型adapter或训练目标。现有DF/VarCNN checkpoint不能直接消费这些token；这只是生成器原型。后续须冻结数据角色和对照，区分输入表示收益与模型/预算变化。时间可选分支尚未实现。
