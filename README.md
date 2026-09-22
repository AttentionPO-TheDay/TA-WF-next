# WF Research — 新研究工作区

本目录是独立新项目。旧研究档案保留在 `/home/rbf/TA-WF`；共享数据位于 `/mnt/data2/ren/datasets`。

日常只看 [当前进度](STATUS.md)、[实验索引](EXPERIMENTS.csv) 和 [研究协议](PROTOCOL.md)。旧工作只需先读 [交接摘要](HANDOFF.md)，需要查证时再打开原始证据。

目录约定：

```text
configs/datasets.json   数据路径与字段语义
src/ta_wf_next/models/  DF、VarCNN、VarCNNDirection 基础模型
scripts/experiment.py  统一创建实验和更新索引
tests/                 后续新增代码的测试
runs/<实验ID>/         一次实验的计划、配置、日志、结果、checkpoint
STATUS.md              唯一的全局当前进度
EXPERIMENTS.csv         唯一的全局实验索引
PROTOCOL.md            新研究问题、信息边界和选择规则
HANDOFF.md             旧研究的必要结论与证据入口
```

开始使用（只需系统 Python，不导入旧项目）：

```bash
cd /home/rbf/TA-WF-next
python3 scripts/experiment.py check
python3 scripts/experiment.py new --name baseline --question '填写本次具体问题'
python3 scripts/experiment.py status --id <返回的实验ID> --state completed --summary '填写结论'
```

创建实验只创建记录，不启动训练。正式运行前补齐该实验的 PLAN.md 和 config.json；未冻结的新方向和资源预算不由目录初始化自动确定。checkpoint 默认只保留 `best.pt` 和 `last.pt`，中间版本仅在有明确用途时保留；已生成的证据不得自动清理。全局不再维护第二套 logs/outputs/checkpoints 目录。

本次没有复制旧 checkpoint、实验配置、训练脚本或 Python 虚拟环境。未来依赖随新代码实际需要锁定；当前管理入口仅使用 Python 标准库。

## 基础 WF 模型

已最小迁入 DF 和 VarCNN，并提供明确命名的方向单分支 `VarCNNDirection`。来源、输入限制与改动见 [模型说明](src/ta_wf_next/models/SOURCES.md)。数据适配器与 screening 训练评价入口已实现；既有实验结果以各 run 的 RESULTS.md 为准。

在安装有 PyTorch 的独立 Python 环境中，可执行 `pip install -e . --no-build-isolation` 安装本地包，或直接设置 `PYTHONPATH=src` 使用：

```python
import torch
from ta_wf_next.models import DF, VarCNNDirection

model = DF(num_classes=100).eval()
x = torch.randn(2, 1, 5000)
with torch.no_grad():
    logits, features = model(x)
    local_features = model.forward_local(x)
```

合成输入单元检查：`PYTHONPATH=src python -m unittest discover -s tests -v`。项目包仅依赖 PyTorch，管理脚本仍仅依赖标准库。

## 多视角生成器与当前候选

[多视角接口](src/ta_wf_next/TRAFFIC_VIEWS.md)包含方向、run、局部窗口、显式可选的受限时间和真实size视角。用户提出的后续候选是“分级释放特征token”；尚未实现选择机制或证明抗漂移收益。具体状态见STATUS.md。

## Git 与本地研究产物

版本控制保留源码、测试、实验脚本、配置、计划、结果报告、小型JSON摘要及EXPERIMENTS.csv。环境文件、锁、缓存、数据数组、抓包、模型权重、runs中的artifacts/checkpoints/logs及生成CSV不提交。忽略不等于删除，本地证据仍保留。

Git不是完整研究归档：逐样本证据、预测和模型需要单独备份；克隆仓库不会恢复这些产物。报告中的服务器绝对路径是本地证据入口，不保证外部可访问。当前数据配置也是服务器路径，换环境须显式配置；部分归档脚本保留旧环境依赖，不应当作可移植生产入口。
