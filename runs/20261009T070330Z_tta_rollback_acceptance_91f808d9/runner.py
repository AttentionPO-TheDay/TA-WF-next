"""Unlabeled entropy-adapter acceptance and rollback experiment."""
import copy
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
R = Path(__file__).resolve().parent
ROOT = R.parents[1]
C = json.loads((R / "config.json").read_text())
sys.path.insert(0, str(R))
from model import MultiViewModel

SEEDS = C["seeds"]
ROLES = C["roles"]
CONDITIONS = C["conditions"]
DEVICE = torch.device("cuda:0")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(name, value):
    path = R / "artifacts" / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def event(stage, **payload):
    with (R / "artifacts/events.jsonl").open("a") as handle:
        handle.write(json.dumps({"utc": time.time(), "stage": stage, **payload}, ensure_ascii=False) + "\n")
    print(stage, json.dumps(payload, ensure_ascii=False), flush=True)


def registry(state, summary):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/experiment.py"), "status", "--id", R.name, "--state", state, "--summary", summary],
        check=True,
    )


def metric(labels, predictions):
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(102), where=denominator > 0)
    return {"accuracy": float((labels == predictions).mean()), "macro_f1": float(f1.mean())}


def fp_rows(values):
    return [hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest() for row in values]


def reconstruct(values):
    output = np.zeros((len(values), 2, 1800), np.float32)
    for index, row in enumerate(values):
        nonzero = row[row != 0]
        bins = np.minimum(np.floor(np.abs(nonzero).astype(np.float64) * 1799 / 80), 1799).astype(np.int64)
        channels = (nonzero < 0).astype(np.int64)
        output[index] = np.bincount(bins + 1800 * channels, minlength=3600).reshape(2, 1800)
    return output


def load_prepared(path, role=None):
    data = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    if "timestamps" not in data:
        data = data[role or "source"]
    timestamps = data["timestamps"].numpy().astype(np.float32, copy=False)
    tam = data["tam"].numpy().astype(np.float32, copy=False)
    labels = data["labels"].numpy().astype(np.int64, copy=False)
    rows = data["rows"].numpy().astype(np.int64, copy=False)
    assert timestamps.ndim == 2 and timestamps.shape[1] == 5000
    assert tam.shape == (len(timestamps), 2, 1800)
    assert labels.shape == rows.shape == (len(timestamps),)
    assert np.isfinite(timestamps).all() and np.isfinite(tam).all()
    assert (timestamps != 0).any(1).all()
    assert np.array_equal(reconstruct(timestamps), tam)
    return {"timestamps": timestamps, "tam": tam, "labels": labels, "rows": rows}


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(C["adapter"]["input_dim"])
        self.down = nn.Linear(C["adapter"]["input_dim"], C["adapter"]["bottleneck"])
        self.act = nn.GELU()
        self.up = nn.Linear(C["adapter"]["bottleneck"], C["adapter"]["input_dim"])
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values):
        return self.up(self.act(self.down(self.norm(values))))


def load_model(seed):
    checkpoint = ROOT / C["checkpoint_run"] / "checkpoints" / f"wide_bn_{seed}_best.pt"
    model = MultiViewModel("wide_bn").to(DEVICE)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, checkpoint


def forward_logits(model, timestamps, tam, adapter=None):
    generated = model.generator(timestamps, tam)
    packet = generated["packet"]
    temporal = generated["time"]
    packet_valid = generated["packet_mask"]
    time_valid = generated["time_mask"]
    if adapter is not None:
        temporal = temporal + adapter(temporal)
    packet = model.packet_projection(packet) + model.packet_positions + model.view_types[0]
    temporal = model.time_projection(temporal) + model.time_positions + model.view_types[1]
    valid = torch.cat([packet_valid, time_valid], 1)
    padding = ~valid
    encoded = torch.cat([packet, temporal], 1).masked_fill(padding[..., None], 0)
    for layer in model.head:
        encoded = layer(encoded, src_key_padding_mask=padding)
        encoded = encoded.masked_fill(padding[..., None], 0)
    encoded = model.final_norm(encoded).masked_fill(padding[..., None], 0)
    packet_encoded = encoded[:, :100]
    time_encoded = encoded[:, 100:]
    packet_mean = (packet_encoded * packet_valid[..., None]).sum(1) / packet_valid.sum(1).clamp_min(1)[..., None]
    time_mean = (time_encoded * time_valid[..., None]).sum(1) / time_valid.sum(1).clamp_min(1)[..., None]
    return model.classifier(torch.cat([packet_mean, time_mean], -1))


def prediction_entropy(logits):
    probabilities = torch.softmax(logits, 1)
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)


@torch.no_grad()
def predict(model, timestamps, tam, adapter=None):
    outputs = []
    batch_size = C["eval_batch_size"]
    for start in range(0, len(timestamps), batch_size):
        xb = torch.from_numpy(timestamps[start : start + batch_size]).to(DEVICE)
        tb = torch.from_numpy(tam[start : start + batch_size]).to(DEVICE)
        outputs.append(forward_logits(model, xb, tb, adapter).cpu())
    return torch.cat(outputs).numpy()


@torch.no_grad()
def support_monitor(model, timestamps, tam, adapter):
    entropy_sum = 0.0
    high_confidence = 0
    batch_size = C["eval_batch_size"]
    threshold = C["adaptation"]["confidence_threshold"]
    for start in range(0, len(timestamps), batch_size):
        xb = torch.from_numpy(timestamps[start : start + batch_size]).to(DEVICE)
        tb = torch.from_numpy(tam[start : start + batch_size]).to(DEVICE)
        logits = forward_logits(model, xb, tb, adapter)
        entropy_sum += float(prediction_entropy(logits).double().sum().cpu())
        high_confidence += int((torch.softmax(logits, 1).amax(1) >= threshold).sum().cpu())
    return {"entropy": entropy_sum / len(timestamps), "high_confidence_rate": high_confidence / len(timestamps)}


@torch.no_grad()
def relative_feature_change(model, timestamps, tam, adapter):
    delta_square = 0.0
    raw_square = 0.0
    batch_size = C["eval_batch_size"]
    for start in range(0, len(timestamps), batch_size):
        xb = torch.from_numpy(timestamps[start : start + batch_size]).to(DEVICE)
        tb = torch.from_numpy(tam[start : start + batch_size]).to(DEVICE)
        raw = model.generator(xb, tb)["time"]
        delta = adapter(raw)
        delta_square += float(delta.double().square().sum().cpu())
        raw_square += float(raw.double().square().sum().cpu())
    return math.sqrt(delta_square / max(raw_square, 1e-12))


def clone_state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def adapt(model, initial_state, timestamps, tam, seed, role, condition):
    del seed, role
    adapter = Adapter().to(DEVICE)
    adapter.load_state_dict(initial_state)
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=C["adaptation"]["learning_rate"], weight_decay=C["adaptation"]["weight_decay"])
    initial_adapter_state = clone_state(adapter)
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    initial = support_monitor(model, timestamps, tam, adapter)
    accepted_state = dict(initial)
    attempted_entropy = []
    accepted_entropy = [initial["entropy"]]
    attempted_confidence = []
    accepted_confidence = [initial["high_confidence_rate"]]
    acceptance = []
    update_losses = []
    batch_size = C["adaptation"]["batch_size"]
    minimum_improvement = C["adaptation"]["entropy_minimum_improvement"]
    confidence_tolerance = C["adaptation"]["confidence_tolerance"]

    for step in range(C["adaptation"]["steps"]):
        before_adapter = clone_state(adapter) if condition == "stepwise_rollback" else None
        before_optimizer = copy.deepcopy(optimizer.state_dict()) if condition == "stepwise_rollback" else None
        indices = (np.arange(batch_size) + step * batch_size) % len(timestamps)
        xb = torch.from_numpy(timestamps[indices]).to(DEVICE)
        tb = torch.from_numpy(tam[indices]).to(DEVICE)
        with torch.enable_grad():
            adapter.train()
            logits = forward_logits(model, xb, tb, adapter)
            loss = prediction_entropy(logits).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in adapter.parameters())
            optimizer.step()
        update_losses.append(float(loss.detach().cpu()))
        candidate = support_monitor(model, timestamps, tam, adapter)
        attempted_entropy.append(candidate["entropy"])
        attempted_confidence.append(candidate["high_confidence_rate"])

        if condition == "stepwise_rollback":
            accept = candidate["entropy"] <= accepted_state["entropy"] - minimum_improvement and candidate["high_confidence_rate"] + confidence_tolerance >= accepted_state["high_confidence_rate"]
            if accept:
                accepted_state = dict(candidate)
            else:
                adapter.load_state_dict(before_adapter)
                optimizer.load_state_dict(before_optimizer)
        else:
            accept = True
            accepted_state = dict(candidate)
        acceptance.append(bool(accept))
        accepted_entropy.append(accepted_state["entropy"])
        accepted_confidence.append(accepted_state["high_confidence_rate"])

    episode_final_accepted = True
    if condition == "episode_rollback":
        episode_final_accepted = accepted_state["entropy"] <= initial["entropy"] - minimum_improvement
        if not episode_final_accepted:
            adapter.load_state_dict(initial_adapter_state)
            optimizer.load_state_dict(initial_optimizer_state)
            accepted_state = dict(initial)

    adapter.eval()
    final = support_monitor(model, timestamps, tam, adapter)
    assert abs(final["entropy"] - accepted_state["entropy"]) <= 1e-7
    assert abs(final["high_confidence_rate"] - accepted_state["high_confidence_rate"]) <= 1e-12
    if condition == "stepwise_rollback":
        accepted_steps = sum(acceptance)
    elif condition == "episode_rollback":
        accepted_steps = C["adaptation"]["steps"] if episode_final_accepted else 0
    else:
        accepted_steps = C["adaptation"]["steps"]
    state = {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
    report = {
        "attempted_steps": C["adaptation"]["steps"],
        "accepted_steps": int(accepted_steps),
        "acceptance_history": acceptance,
        "episode_final_accepted": bool(episode_final_accepted),
        "support_entropy_before": initial["entropy"],
        "support_entropy_after": final["entropy"],
        "support_high_confidence_before": initial["high_confidence_rate"],
        "support_high_confidence_after": final["high_confidence_rate"],
        "attempted_entropy_history": attempted_entropy,
        "accepted_entropy_history": accepted_entropy,
        "attempted_high_confidence_history": attempted_confidence,
        "accepted_high_confidence_history": accepted_confidence,
        "loss_history": update_losses,
        "relative_feature_change": relative_feature_change(model, timestamps, tam, adapter),
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
    }
    return adapter, state, report


def split_targets(source, role, target):
    source_fingerprints = set(fp_rows(source["timestamps"]))
    fingerprints = fp_rows(target["timestamps"])
    first = {}
    source_overlap = 0
    for index, fingerprint in enumerate(fingerprints):
        if fingerprint in source_fingerprints:
            source_overlap += 1
            continue
        first.setdefault(fingerprint, index)
    keep = np.array(sorted(first.values()), dtype=np.int64)
    keys = [fingerprints[index] for index in keep]
    order = np.argsort([hashlib.sha256(f"{role}|{fingerprint}|20261009".encode()).hexdigest() for fingerprint in keys])
    keep = keep[order]
    support = keep[: C["split"]["support_rows"]]
    query = keep[C["split"]["support_rows"] :]
    assert len(set(fingerprints[index] for index in support)) == len(support)
    assert set(fingerprints[index] for index in support).isdisjoint(set(fingerprints[index] for index in query))
    assert len(query) > 0
    return support, query, {
        "raw_rows": len(target["timestamps"]),
        "source_overlap_excluded": source_overlap,
        "unique_after_source_exclusion": len(keep),
        "support_rows": len(support),
        "query_rows": len(query),
        "query_classes": int(len(np.unique(target["labels"][query]))),
        "query_fp_unique": len(set(fingerprints[index] for index in query)),
    }


def cell_metrics(labels, frozen_predictions, predictions):
    frozen = metric(labels, frozen_predictions)
    adapted = metric(labels, predictions)
    frozen_correct = frozen_predictions == labels
    frozen_wrong = ~frozen_correct
    return {
        **adapted,
        "base_accuracy": frozen["accuracy"],
        "base_macro_f1": frozen["macro_f1"],
        "negative_adaptation_rate": float(((frozen_correct) & (predictions != labels)).sum() / max(1, frozen_correct.sum())),
        "recovery_rate": float(((frozen_wrong) & (predictions == labels)).sum() / max(1, frozen_wrong.sum())),
        "changed_rate": float((predictions != frozen_predictions).mean()),
    }


def main():
    started = time.time()
    torch.set_num_threads(C["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert C["status"] == "frozen" and C["id"] == R.name
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("pipeline time budget exhausted")))
    signal.alarm(C["budget"]["max_wall_seconds"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(C["physical_gpu"])
    assert torch.cuda.is_available()

    source_container = load_prepared(C["prepared"]["source_valid"], "source")
    source_valid = load_prepared(C["prepared"]["source_valid"], "valid")
    source_all = {key: np.concatenate([source_container[key], source_valid[key]], axis=0) for key in source_container}
    targets = {role: load_prepared(C["prepared"][role]) for role in ROLES}
    comparison = ROOT / C["comparison_run"]
    splits = {}
    split_report = {}
    for role, target in targets.items():
        support, query, report = split_targets(source_all, role, target)
        splits[role] = (support, query)
        split_report[role] = report
        with np.load(comparison / "artifacts" / f"split_{role}.npz") as previous:
            assert np.array_equal(target["rows"][support], previous["support"])
            assert np.array_equal(target["rows"][query], previous["query"])
            assert np.array_equal(support, previous["support_indices"])
            assert np.array_equal(query, previous["query_indices"])
        np.savez_compressed(R / "artifacts" / f"split_{role}.npz", support=target["rows"][support], query=target["rows"][query], support_indices=support, query_indices=query)
    save_json("split_manifest.json", split_report)

    files = list(R.glob("*.py")) + [R / "config.json", R / "PLAN.md", R / "SOURCES.md", ROOT / "configs/datasets.json"]
    files += [ROOT / path for path in C["prepared"].values()]
    files += [ROOT / C["checkpoint_run"] / "checkpoints" / f"wide_bn_{seed}_best.pt" for seed in SEEDS]
    files += [comparison / "artifacts" / f"split_{role}.npz" for role in ROLES]
    files += [comparison / "artifacts" / f"pred_{seed}_{role}_full_global.npz" for seed in SEEDS for role in ROLES]
    save_json("freeze.json", {str(path.relative_to(ROOT)): sha(path) for path in files})
    event("preflight", gpu=torch.cuda.get_device_name(0), split=split_report)
    registry("running", "GPU1执行中；3日期×3seed×global/逐步回退/整体回退，query隔离与10步预算已冻结。")

    all_rows = []
    for seed in SEEDS:
        model, _ = load_model(seed)
        frozen_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.manual_seed(seed + 91000)
        initial_adapter = Adapter()
        initial_state = {key: value.detach().cpu().clone() for key, value in initial_adapter.state_dict().items()}
        with torch.no_grad():
            xb = torch.from_numpy(source_container["timestamps"][:8]).to(DEVICE)
            tb = torch.from_numpy(source_container["tam"][:8]).to(DEVICE)
            assert torch.allclose(forward_logits(model, xb, tb), model(xb, tb), atol=1e-5, rtol=1e-5)

        for role in ROLES:
            target = targets[role]
            support, query = splits[role]
            support_timestamps, support_tam = target["timestamps"][support], target["tam"][support]
            query_timestamps, query_tam = target["timestamps"][query], target["tam"][query]
            query_labels = target["labels"][query]
            frozen_logits = predict(model, query_timestamps, query_tam)
            frozen_predictions = frozen_logits.argmax(1)
            with np.load(comparison / "artifacts" / f"pred_{seed}_{role}_frozen.npz") as previous:
                assert np.array_equal(target["rows"][query], previous["rows"])
                assert np.array_equal(frozen_predictions, previous["predictions"])
            np.savez_compressed(R / "artifacts" / f"pred_{seed}_{role}_frozen.npz", rows=target["rows"][query], labels=query_labels, predictions=frozen_predictions, logits=frozen_logits)
            all_rows.append({"seed": seed, "role": role, "condition": "frozen", "steps": 0, "rows": len(query_labels), **metric(query_labels, frozen_predictions), "negative_adaptation_rate": 0.0, "recovery_rate": 0.0, "changed_rate": 0.0})

            for condition in [value for value in CONDITIONS if value != "frozen"]:
                adapter, adapter_state, adaptation_report = adapt(model, initial_state, support_timestamps, support_tam, seed, role, condition)
                logits = predict(model, query_timestamps, query_tam, adapter)
                predictions = logits.argmax(1)
                for key, value in model.state_dict().items():
                    assert torch.equal(value.cpu(), frozen_model_state[key]), key
                assert all(parameter.grad is None for parameter in model.parameters())
                assert np.isfinite(logits).all()
                if condition == "global":
                    with np.load(comparison / "artifacts" / f"pred_{seed}_{role}_full_global.npz") as previous:
                        assert np.array_equal(predictions, previous["predictions"])
                        assert np.allclose(logits, previous["logits"], atol=1e-6, rtol=1e-6)
                all_rows.append({"seed": seed, "role": role, "condition": condition, "steps": C["adaptation"]["steps"], "rows": len(query_labels), **cell_metrics(query_labels, frozen_predictions, predictions), **adaptation_report})
                torch.save({"state_dict": adapter_state, "seed": seed, "role": role, "condition": condition}, R / "checkpoints" / f"adapter_{seed}_{role}_{condition}.pt")
                np.savez_compressed(R / "artifacts" / f"pred_{seed}_{role}_{condition}.npz", rows=target["rows"][query], labels=query_labels, predictions=predictions, logits=logits)
                del adapter
            save_json("scores.json", all_rows)
            event("cell_completed", seed=seed, role=role)
        del model
        torch.cuda.empty_cache()

    save_json("scores.json", all_rows)
    registry("running", "全部36组预测和27个适配器任务完成；正在独立复核指标、split、冻结hash和checkpoint。")
    subprocess.run([sys.executable, str(R / "postrun_audit.py")], check=True)

    def mean(condition, key):
        return float(np.mean([row[key] for row in all_rows if row["condition"] == condition]))

    global_rows = {(row["seed"], row["role"]): row for row in all_rows if row["condition"] == "global"}

    def rollback_verdict(condition):
        rows = [row for row in all_rows if row["condition"] == condition]
        macro_delta = (mean(condition, "macro_f1") - mean("global", "macro_f1")) * 100
        negative_delta = (mean(condition, "negative_adaptation_rate") - mean("global", "negative_adaptation_rate")) * 100
        lower_negative_cells = sum(row["negative_adaptation_rate"] < global_rows[(row["seed"], row["role"])]["negative_adaptation_rate"] for row in rows)
        return {"macro_f1_minus_global_mean_pp": macro_delta, "negative_adaptation_minus_global_mean_pp": negative_delta, "cells_with_lower_negative_adaptation": lower_negative_cells, "pass": macro_delta >= 0 and negative_delta <= -0.2 and lower_negative_cells >= 6}

    verdict = {"primary_stepwise": rollback_verdict("stepwise_rollback"), "secondary_episode": rollback_verdict("episode_rollback")}
    verdict["pass"] = verdict["primary_stepwise"]["pass"]
    save_json("verdict.json", verdict)
    lines = ["# 无标签更新接受与回退实验结果", "", "全部 3 个训练 seed × 3 个开发日期完成；support 和 query 精确隔离，目标标签只在预测固定后用于评分。global 条件逐单元精确复现上一轮 full_global 预测。", "", "| 条件 | Macro-F1均值 | Accuracy均值 | 负适应率均值 | 纠错率均值 | 接受步数均值 | 特征相对改变量 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        accepted_steps = 0.0 if condition == "frozen" else mean(condition, "accepted_steps")
        change = 0.0 if condition == "frozen" else mean(condition, "relative_feature_change")
        lines.append(f"| {condition} | {mean(condition, 'macro_f1') * 100:.3f}% | {mean(condition, 'accuracy') * 100:.3f}% | {mean(condition, 'negative_adaptation_rate') * 100:.3f}% | {mean(condition, 'recovery_rate') * 100:.3f}% | {accepted_steps:.2f} | {change:.6f} |")
    lines += ["", "裁决 JSON：", "```json", json.dumps(verdict, ensure_ascii=False, indent=2), "```", "", "逐单元结果及接受轨迹见 `artifacts/scores.json`；完整 logits、适配器 checkpoint、split、冻结 hash 和独立审计均已保留。", "", "本轮是 TemporalDrift 已观察数据上的开发筛查；没有重新训练生成器或 backbone，也未访问 WTT-Time、AWF 或其他漂移因素数据。"]
    (R / "RESULTS.md").write_text("\n".join(lines) + "\n")
    registry("completed", f"完成且审计通过；stepwise相对global Macro-F1 {verdict['primary_stepwise']['macro_f1_minus_global_mean_pp']:.3f}pp，负适应率差 {verdict['primary_stepwise']['negative_adaptation_minus_global_mean_pp']:.3f}pp，主裁决={verdict['pass']}。")
    event("completed", elapsed_seconds=time.time() - started, verdict=verdict)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save_json("failure.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        registry("failed", f"执行失败：{type(error).__name__}: {error}；产物保留。")
        raise
