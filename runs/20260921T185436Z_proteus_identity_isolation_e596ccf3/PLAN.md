# 20260921T185436Z_proteus_identity_isolation_e596ccf3

问题：Version整行身份恢复及Network/Behavior内容URL隔离、旧使用记录审计；无训练评分

状态：frozen audit-only。2026-09-22 用户授权接续审计。

范围：仅 configs/datasets.json data_root 中 VersionDrift 045/046/047/048 的 train/valid/drift，NetworkDrift train/valid/SG/JP/USA/DE/UK，BehaviorDrift train/valid/test/subpage。显式读取 X/y 与唯一 URL 字段进行隔离审计，标签不进入模型或选参；记录这是结构/数据使用暴露。Temporal、OpenWorld、Defense、WTT、AWF 不读取。

方法：完整float64行 SHA256 索引候选，并对候选逐字节比较验证；不以标签作为匹配前提，同时检查异标签冲突。Version 以各目录 train/valid 的目录版本为协议支持的锚点，恢复唯一/多版本/未匹配角色，输出逐行匹配manifest；不是独立采集provenance认证。Network/Behavior 检查全行及前5000方向输入相等（候选精确比较），输出文件内和跨文件重复。URL只检查精确字符串重复与标签冲突，不擅自规范化路径或合并站点。

预算：单进程 CPU、只读memmap、最多23个预定文件，约数十GB顺序读取；不训练/评分/新split/新token。数值检查仅有限抽样，不声称全量质量通过。缺文件、对象数组或压缩X不兼容则停止；输出只写本run。既往暴露用限定目录文本检索，未搜到不等于从未使用。完成报告/完整性检查后停止。
