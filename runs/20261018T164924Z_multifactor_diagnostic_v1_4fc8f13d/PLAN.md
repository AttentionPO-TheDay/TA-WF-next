# 多因素诊断及原型锚定对齐 v1

严格执行 DIAGNOSTIC_PROTOCOL.md 与 config.json。三seed、十诊断切片、七条件共210单元；source-only预检→逐行清单与TAM隔离→数据freeze→GPU0/2队列→全部重载审计→三表/雷达/两组共享坐标t-SNE。外部v4只引用，不训练新底座、不重跑外部、不按query选模。

本方法因已读AWF/WTT反馈提出，属于post-external-feedback开发版本；源码及输入散列在 artifacts/freeze.json，参考实现和旧结果来源见 SOURCES.md。资源wall3h/6GPUh/output8GiB，单元900s，失败立即停止并保留。细节与正负判据见规范。
