# P0 重复隔离与 split 边界审计 v2

用户授权：处理两项 P0；不授权训练或删除原数据。本轮全部 CPU 只读，原角色不变；导出原文件/行号对应的隔离候选，不生成新的随机 split。

范围：Shusterman 的全部 142 HDF5（完整性审计会读取原 WTT future 日期结构/向量/网站标签；仅判断重复，不评分、不选模，不自动撤销或赋予确认资格）。核心 NPZ：Temporal/Version/Network/Behavior 所有原始 X/y 共30文件，跳过 tam_*。AWF/Defense/OpenWorld 和模型 checkpoint 不读取。

流程：先复核历史审计（20260921 identity、旧 Shusterman/WTT 报告），历史结果显式引用不计本轮独立重复。借旧 venv 的 numpy/h5py，不导入旧训练代码。补充 configs/datasets.json 中三个缺失目录，以其定位数据。

HDF5：验证有限整数且 int32 范围；跨 dtype 统一 little-endian int32 hash；全量保留组成员，区分 all-zero 与非零有效 trace。每个重复组以逐向量字节比较验证 hash 命中。报告所有文件对的非零交叉，不仅相对 first occurrence 的分类计数。all-zero/无效值隔离；辅助 static_webpages_network 不进入主池；非零冲突标签整组隔离；原 test 优先保护，匹配 test 的 train 排除候选。既有 WTT split/indices 不改。

NPZ：未压缩 X.npy 只读 memmap；labels 禁用 pickle、校验有限整数。raw10000 float64 字节、first5000 float32 字节、first5000 direction int8 三视图全量 hash，每次 hash 命中逐字节复核。raw 与截断重复分别报告；不把跨目录标签数字相等当身份映射。原非train角色全部保护；匹配非train的train、重复train副本导出候选排除；有标签分歧的组隔离候选。方向相同不能反推同一 session，也不能改写原语义标签。

预算：单 CPU 进程，数值库2线程，1800s，256行/块；输出上限2GB。数据变更、schema/标签错误或超时即停止，不静默跳文件再宣称完成。

检查：真实SG首尾行 memmap 与 np.load 一致；三视图规则和跨dtype canonical synthetic checks；全文件计数核验、组CSV与数组对照、隔离候选回查、源size/mtime/inode前后一致；最终脚本/config/输出hash记录。

修订原因：初版存在syntax、NPY header偏移、错误地要求时间戳为整数、HDF5跨dtype未统一和有限组示例遗漏风险；初版结果不具有效裁决资格，保留为attempt_v1。修订全部为结构读取/重复规则实现修正，无性能反馈或调参。后台shell的短命进程未证明运行，实际采用受监督PTY任务。
