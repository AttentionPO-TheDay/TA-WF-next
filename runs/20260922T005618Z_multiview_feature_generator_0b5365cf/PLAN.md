# 多视角流量表示生成器

日期：2026-09-22。目标是把现有可信信息分别暴露为可审计视角，而非假定 burst 是唯一特征。用户授权实现与CPU验证，不授权训练评分。

输入：方向、Temporal/Proteus signed timestamp、若以后提供 signed size。统一观察预算5000，保留stored order。视角：packet_direction、exact/coarse runs、固定局部方向窗口、同方向时间间隔/run内跨度（仅signed timestamp，负值标None）、signed size的绝对值/run totals（不由timestamp伪造）。

规则：零是padding；非有限值、padding后非零拒绝；不排序、不跨方向修复、不计算rate；缺失模态为None，select显式拒绝。coarse-only不得读取exact runs/observed_count旁路。边界标记是可能删失，不是资源边界。

验证：合成测试；Version048/train、Network/train、Behavior/subpage各128条固定等距行，读取X前5000，比较signed_timestamp与sign方向结构，验证多视角字段。标签/URL不读，未训练评分，不读取WTT/AWF。结果仅接口和边界正确性，不证明判别性、抗漂移或泛化。代码和结果归本run。
