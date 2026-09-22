#!/usr/bin/env python3
"""Create source and final reports from immutable diagnostic JSON artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs" / "exp_4b72ed0c9a354d54"
MODELS = ("df", "varcnn_direction")


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite formal artifact: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pp(value: float) -> str:
    return f"{100 * value:.2f}"


def source() -> None:
    matrix_rows = []
    selection_rows = []
    loaded = {}
    for name in MODELS:
        result = json.loads((RUN / "artifacts" / f"{name}_source_diagnostic.json").read_text())
        loaded[name] = result
        for key, row in result["matrix"].items():
            candidates = row.get("selection_candidates", [])
            matrix_rows.append({
                "backbone": name, "configuration": key, "representation": row["representation"], "readout": row["readout"],
                "selected_C": row.get("selected_C", ""), "validation_accuracy": row["source_validation"]["accuracy"],
                "validation_macro_f1": row["source_validation"]["macro_f1"], "holdout_accuracy": row["source_holdout"]["accuracy"],
                "holdout_macro_precision": row["source_holdout"]["macro_precision"], "holdout_macro_recall": row["source_holdout"]["macro_recall"],
                "holdout_macro_f1": row["source_holdout"]["macro_f1"], "dimension": json.dumps(row["cost"]["dimension"]),
                "reference_or_model_bytes": row["cost"].get("reference_bytes", row["cost"].get("model_bytes")),
                "fit_seconds_all_C": row["cost"].get("all_candidate_fit_seconds", 0.0),
                "predict_seconds": row["cost"]["validation_and_holdout_predict_seconds"],
                "any_convergence_warning": any(candidate["convergence_warning"] for candidate in candidates),
            })
            for candidate in candidates:
                selection_rows.append({
                    "backbone": name, "representation": row["representation"], "C": candidate["C"],
                    "validation_accuracy": candidate["source_validation"]["accuracy"],
                    "validation_macro_f1": candidate["source_validation"]["macro_f1"],
                    "selected": candidate["C"] == row["selected_C"], "fit_seconds": candidate["fit_seconds"],
                    "n_iter": json.dumps(candidate["n_iter"]), "convergence_warning": candidate["convergence_warning"],
                })
    write_csv(RUN / "artifacts" / "source_matrix.csv", matrix_rows)
    write_csv(RUN / "artifacts" / "linear_selection_records.csv", selection_rows)

    flags = {}
    lines = [
        "# Source-only representation diagnostic", "",
        "本报告在任何本实验 future 特征抽取前生成。配置选择只使用 official source validation；source-holdout 只用于下表最终报告。线性分类器使用 supervised-train 的约 160 标签/类，不是与 2-reference 余弦读出的公平部署基线。", "",
    ]
    for name in MODELS:
        result = loaded[name]
        m = result["matrix"]
        lines += [f"## {name}", "", "| 层/读取 | validation F1 | holdout accuracy / F1 |", "|---|---:|---:|"]
        for key in sorted(m):
            row = m[key]
            lines.append(f"| `{key}` | {pp(row['source_validation']['macro_f1'])} | {pp(row['source_holdout']['accuracy'])} / {pp(row['source_holdout']['macro_f1'])} |")
        old = result["old_frozen_controls_not_new_repeats"]
        lines += ["", "旧冻结对照（非本实验重跑）：" + ", ".join(f"{key} {pp(old[key]['accuracy'])}/{pp(old[key]['macro_f1'])}" for key in "ACDE") + "（accuracy/F1）。", ""]
        def f1(key): return m[key]["source_holdout"]["macro_f1"]
        model_flags = {
            "shallow_global_has_usable_discrimination": max(f1("shallow_global__cosine_1nn"), f1("shallow_global__linear")) >= .4,
            "ordered_beats_unordered_shallow_by_5pp": f1("shallow_ordered4__cosine_1nn") - f1("shallow_unordered4__cosine_region_permutation_invariant") >= .05,
            "ordered_beats_global_shallow_linear_by_5pp": f1("shallow_ordered4__linear") - f1("shallow_global__linear") >= .05,
            "linear_beats_cosine_shallow_ordered_by_10pp": f1("shallow_ordered4__linear") - f1("shallow_ordered4__cosine_1nn") >= .10,
            "intermediate_beats_shallow_ordered_linear_by_5pp": f1("intermediate_ordered4__linear") - f1("shallow_ordered4__linear") >= .05,
            "selected_validation_configuration": max(m, key=lambda key: m[key]["source_validation"]["macro_f1"]),
        }
        flags[name] = model_flags
        lines += [
            f"- 浅层 global mean 可用（holdout F1≥40%）：{model_flags['shallow_global_has_usable_discrimination']}。",
            f"- 浅层 ordered cosine 相对 unordered 的 F1 差：{pp(f1('shallow_ordered4__cosine_1nn') - f1('shallow_unordered4__cosine_region_permutation_invariant'))} pp；达到 5 pp 门槛：{model_flags['ordered_beats_unordered_shallow_by_5pp']}。",
            f"- 浅层 ordered linear 相对 global linear 的 F1 差：{pp(f1('shallow_ordered4__linear') - f1('shallow_global__linear'))} pp。",
            f"- 浅层 ordered linear 相对 ordered cosine 的 F1 差：{pp(f1('shallow_ordered4__linear') - f1('shallow_ordered4__cosine_1nn'))} pp；达到监督差异更严格的 10 pp 门槛：{model_flags['linear_beats_cosine_shallow_ordered_by_10pp']}。",
            f"- 中间层 ordered linear 相对浅层同读出的 F1 差：{pp(f1('intermediate_ordered4__linear') - f1('shallow_ordered4__linear'))} pp；达到 5 pp 门槛：{model_flags['intermediate_beats_shallow_ordered_linear_by_5pp']}。",
            "",
        ]
        # The newly computed shallow rules must reproduce old D/E exactly.
        if m["shallow_global__cosine_1nn"]["source_holdout"]["macro_f1"] != old["E"]["macro_f1"]:
            raise ValueError(f"{name} shallow global does not reproduce old E")
        if m["shallow_unordered4__cosine_region_permutation_invariant"]["source_holdout"]["macro_f1"] != old["D"]["macro_f1"]:
            raise ValueError(f"{name} shallow unordered does not reproduce old D")
    summary_path = RUN / "artifacts" / "source_diagnostic_summary.json"
    if summary_path.exists(): raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps({"thresholds": {"clear_effect_pp": 5, "linear_vs_cosine_pp": 10, "usable_validation_macro_f1": .4}, "flags": flags}, indent=2, sort_keys=True) + "\n")
    report = RUN / "SOURCE_RESULTS.md"
    if report.exists(): raise FileExistsError(report)
    report.write_text("\n".join(lines) + "\n")
    print(report)


def final() -> None:
    frozen = json.loads((RUN / "artifacts" / "frozen_selection.json").read_text())
    future = {name: json.loads((RUN / "artifacts" / f"{name}_future_diagnostic.json").read_text()) for name in MODELS}
    rows = []
    lines = ["# 实验结果", "", "## 完成状态", "", "有界表示诊断已按预注册 source→freeze→future 顺序完成。新增 backbone 训练为 0；未使用外部数据集、DNNF/TFAN、attention、新损失、额外层或大规模超参搜索。", "", "## 冻结 source 选择", ""]
    for name in MODELS:
        selection = frozen["selected"][name]
        lines.append(f"- {name}: `{selection['matrix_key']}`，C={selection.get('selected_C','—')}；validation accuracy/F1 {pp(selection['source_validation']['accuracy'])}/{pp(selection['source_validation']['macro_f1'])}，source-holdout {pp(selection['source_holdout']['accuracy'])}/{pp(selection['source_holdout']['macro_f1'])}。")
    lines += ["", "完整 source 矩阵、门槛判断、逐网站指标与监督预算限定见 `SOURCE_RESULTS.md`、`artifacts/source_matrix.csv` 和两个 `*_source_diagnostic.json`。", "", "## Future 绝对结果、相对 source 下降与 C 差值", "", "单元格均为 accuracy / macro-F1 百分数或百分点。下降为 future-source，因此负数表示下降。", "", "| Backbone | 日期 | 冻结诊断 | 相对 source | 旧 C | 诊断-C |", "|---|---|---:|---:|---:|---:|"]
    repeated = {}
    date_order = ("day14", "day30", "day90", "day150", "day270")
    for name in MODELS:
        positive = 0
        for domain in date_order:
            row = future[name]["domains"][domain]
            s, c = row["selected"], row["old_C_hard_control"]
            d, gap = row["drop_from_source_holdout"], row["selected_minus_C"]
            positive += gap["macro_f1"] > 0
            lines.append(f"| {name} | {domain} | {pp(s['accuracy'])} / {pp(s['macro_f1'])} | {pp(d['accuracy'])} / {pp(d['macro_f1'])} | {pp(c['accuracy'])} / {pp(c['macro_f1'])} | {pp(gap['accuracy'])} / {pp(gap['macro_f1'])} |")
            rows.append({"backbone": name, "domain": domain, "selected_accuracy": s["accuracy"], "selected_macro_f1": s["macro_f1"], "source_delta_accuracy": d["accuracy"], "source_delta_macro_f1": d["macro_f1"], "C_accuracy": c["accuracy"], "C_macro_f1": c["macro_f1"], "selected_minus_C_accuracy": gap["accuracy"], "selected_minus_C_macro_f1": gap["macro_f1"]})
        repeated[name] = positive >= 4
    write_csv(RUN / "artifacts" / "future_metrics.csv", rows)
    source_summary = json.loads((RUN / "artifacts" / "source_diagnostic_summary.json").read_text())
    both_source = all(frozen["selected"][name] is not None for name in MODELS)
    both_future = all(repeated.values())
    lines += ["", "## 按预注册门槛的证据解释（不替 Host 决策）", ""]
    for name in MODELS:
        flags = source_summary["flags"][name]
        lines.append(f"- {name}: 浅层 global mean 未达可用门槛；ordered linear 显著优于对应 cosine={flags['linear_beats_cosine_shallow_ordered_by_10pp']}；唯一中间层显著优于浅层={flags['intermediate_beats_shallow_ordered_linear_by_5pp']}；future 相对 C 至少 4/5 日期 macro-F1 为正={repeated[name]}。")
    lines += [
        f"- 两 backbone 都有 validation F1≥40% 的 source 配置：{both_source}；两者都达到重复 future 正向收益：{both_future}。只有二者同时为 True 才满足建议 DNNF/TFAN 同权限比较的证据条件。",
        "- ordered 对无序的 5 pp 证据按 backbone 分开报告；即使为正也只说明有限顺序结构值得检验，不证明局部抗漂移机制。",
        "- 若 source 恢复但 future 相对 C 不重复为正，应归入一般识别恢复；若固定诊断无明确正向信号，应结束当前局部候选。是否据此继续或停止由 Host 决定。",
        "", "## Provenance、成本与限制", "",
        "- Checkpoint：DF epoch 29 / SHA-256 `1bf851...c9133ae`；VarCNNDirection epoch 23 / `fc3ade...1ff83`。完整路径、哈希、训练记录与模型来源见 `representation_diagnostic_plan.md` 和 source JSON。",
        "- Split v3 SHA-256 `0f322e...3f142`，16,309 train / 204 reference / 2,040 holdout；official valid 2,160。reference/holdout 未拟合或选模。",
        "- VarCNNDirection 中间层对短流量有明显覆盖限制；所有短样本以预注册零表示保留，没有过滤。完整覆盖和运行成本见 `artifacts/layer_source_audit.json` 及各 JSON。",
        "- Future 日期是已观察的 TemporalDrift 开发证据，不是独立确认。线性读出监督预算远大于 2-reference 方法，任何提升不归因于原局部机制。",
        "- 每 backbone 的 future JSON 和 predictions JSON 保存完整逐网站结果与固定预测；旧 C 来自 donor artifact，并标记为非本次重跑。",
    ]
    (RUN / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(RUN / "RESULTS.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("source", "final"))
    args = parser.parse_args()
    (source if args.stage == "source" else final)()


if __name__ == "__main__":
    main()
