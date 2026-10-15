"""Proteus mechanism-faithful port on the current wide+BN model."""
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
R = Path(__file__).resolve().parent
ROOT = R.parents[1]
C = json.loads((R / "config.json").read_text())
sys.path.insert(0, str(R))
from model import MultiViewModel

SEEDS = C["seeds"]
ROLES = C["roles"]
DEVICE = torch.device("cuda:0")
COMPARISON = ROOT / C["comparison_run"]


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
    subprocess.run([sys.executable, str(ROOT / "scripts/experiment.py"), "status", "--id", R.name, "--state", state, "--summary", summary], check=True)


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


def load_model(seed):
    checkpoint = ROOT / C["base_checkpoint_run"] / "checkpoints" / f"wide_bn_{seed}_best.pt"
    model = MultiViewModel("wide_bn").to(DEVICE)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def forward_with_features(model, timestamps, tam):
    generated = model.generator(timestamps, tam)
    packet = generated["packet"]
    temporal = generated["time"]
    packet_valid = generated["packet_mask"]
    time_valid = generated["time_mask"]
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
    features = torch.cat([packet_mean, time_mean], -1)
    return model.classifier(features), features


def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def compute_gaussian_kernel(source, target):
    sample_count = int(source.size(0)) + int(target.size(0))
    combined = torch.cat([source, target], dim=0)
    l2_distance = ((combined.unsqueeze(0) - combined.unsqueeze(1)) ** 2).sum(2)
    bandwidth = torch.sum(l2_distance) / (sample_count ** 2 - sample_count)
    bandwidth = torch.clamp(bandwidth, min=1e-5)
    return torch.exp(-l2_distance / (bandwidth + 1e-5))


def calculate_mmd_loss(source_features, target_features):
    batch_size = min(source_features.size(0), target_features.size(0))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]
    kernels = compute_gaussian_kernel(source_features, target_features)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)


@torch.no_grad()
def predict(model, timestamps, tam, batch_size=128):
    outputs = []
    for start in range(0, len(timestamps), batch_size):
        xb = torch.from_numpy(timestamps[start : start + batch_size]).to(DEVICE)
        tb = torch.from_numpy(tam[start : start + batch_size]).to(DEVICE)
        outputs.append(forward_with_features(model, xb, tb)[0].cpu())
    return torch.cat(outputs).numpy()


@torch.no_grad()
def gmm_pseudo_labels(model, timestamps, tam):
    model.eval()
    entropies = []
    predictions = []
    for start in range(0, len(timestamps), C["eval_batch_size"]):
        xb = torch.from_numpy(timestamps[start : start + C["eval_batch_size"]]).to(DEVICE)
        tb = torch.from_numpy(tam[start : start + C["eval_batch_size"]]).to(DEVICE)
        logits, _ = forward_with_features(model, xb, tb)
        entropies.append(entropy(logits).cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
    raw_entropy = np.concatenate(entropies).astype(np.float64)
    pseudo_labels = np.concatenate(predictions).astype(np.int64)
    minimum = raw_entropy.min()
    span = raw_entropy.max() - minimum
    if not np.isfinite(span) or span <= 0:
        raise FloatingPointError("GMM entropy range is zero or non-finite")
    normalized = ((raw_entropy - minimum) / span).reshape(-1, 1)
    gmm = GaussianMixture(n_components=C["adaptation"]["gmm_components"], tol=C["adaptation"]["gmm_tol"])
    gmm.fit(normalized)
    probabilities = gmm.predict_proba(normalized)
    low_uncertainty_index = np.argmin(gmm.means_.flatten())
    clean_probabilities = probabilities[:, low_uncertainty_index]
    threshold = C["adaptation"]["gmm_threshold"]
    selected = clean_probabilities >= threshold
    if int(selected.sum()) < C["adaptation"]["target_batch_size"]:
        raise RuntimeError(f"pseudo-label coverage {int(selected.sum())} is below one batch")
    return {
        "probabilities": clean_probabilities,
        "predictions": pseudo_labels,
        "selected": selected,
        "gmm_means": gmm.means_.flatten().tolist(),
        "entropy_min": float(minimum),
        "entropy_max": float(raw_entropy.max()),
        "coverage": float(selected.mean()),
    }


def adapt_proteus(model, source, support, seed, role):
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    source_dataset = TensorDataset(torch.from_numpy(source["timestamps"]), torch.from_numpy(source["tam"]), torch.from_numpy(source["labels"]))
    support_inputs = torch.from_numpy(support["timestamps"])
    support_tam = torch.from_numpy(support["tam"])
    support_dataset = TensorDataset(support_inputs, support_tam, torch.zeros(len(support_inputs), dtype=torch.long))
    source_loader = DataLoader(source_dataset, batch_size=C["adaptation"]["source_batch_size"], shuffle=True, drop_last=True, num_workers=0)
    support_loader = DataLoader(support_dataset, batch_size=C["adaptation"]["target_batch_size"], shuffle=True, drop_last=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=C["adaptation"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    source_iterator = iter(source_loader)
    pseudo_loader = None
    pseudo_iterator = None
    history = []
    gmm_history = []
    model.train()

    for epoch in range(C["adaptation"]["epochs"]):
        if epoch % C["adaptation"]["gmm_refresh_every_epochs"] == 0:
            gmm_report = gmm_pseudo_labels(model, support["timestamps"], support["tam"])
            gmm_history.append({"epoch": epoch, **{key: value for key, value in gmm_report.items() if key != "probabilities" and key != "predictions" and key != "selected"}})
            pseudo_dataset = TensorDataset(support_inputs[gmm_report["selected"]], support_tam[gmm_report["selected"]], torch.from_numpy(gmm_report["predictions"][gmm_report["selected"]]))
            pseudo_loader = DataLoader(pseudo_dataset, batch_size=C["adaptation"]["target_batch_size"], shuffle=True, drop_last=True, num_workers=0)
            pseudo_iterator = iter(pseudo_loader)
        model.train()
        origin_loss_sum = 0.0
        mmd_loss_sum = 0.0
        entropy_loss_sum = 0.0
        pseudo_loss_sum = 0.0
        total_samples = 0
        for adapt_batch in support_loader:
            try:
                origin_batch = next(source_iterator)
            except StopIteration:
                source_iterator = iter(source_loader)
                origin_batch = next(source_iterator)
            try:
                pseudo_batch = next(pseudo_iterator)
            except StopIteration:
                pseudo_iterator = iter(pseudo_loader)
                pseudo_batch = next(pseudo_iterator)
            adapt_x, adapt_t, _ = (value.to(DEVICE) for value in adapt_batch)
            origin_x, origin_t, origin_y = (value.to(DEVICE) for value in origin_batch)
            pseudo_x, pseudo_t, pseudo_y = (value.to(DEVICE) for value in pseudo_batch)
            optimizer.zero_grad(set_to_none=True)
            origin_outputs, origin_features = forward_with_features(model, origin_x, origin_t)
            adapt_outputs, adapt_features = forward_with_features(model, adapt_x, adapt_t)
            pseudo_outputs, _ = forward_with_features(model, pseudo_x, pseudo_t)
            softmax_out = F.softmax(adapt_outputs, dim=-1)
            mean_softmax = softmax_out.mean(dim=0)
            classification_loss = criterion(origin_outputs, origin_y)
            pseudo_loss = criterion(pseudo_outputs, pseudo_y)
            entropy_loss = entropy(adapt_outputs).mean(0) + torch.sum(mean_softmax * torch.log(mean_softmax + 1e-5))
            mmd_loss = calculate_mmd_loss(origin_features, adapt_features)
            total_loss = classification_loss + pseudo_loss + entropy_loss + mmd_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite Proteus loss at epoch {epoch}")
            total_loss.backward()
            optimizer.step()
            origin_loss_sum += float(classification_loss.detach().cpu()) * len(origin_x)
            mmd_loss_sum += float(mmd_loss.detach().cpu()) * len(origin_x)
            entropy_loss_sum += float(entropy_loss.detach().cpu()) * len(origin_x)
            pseudo_loss_sum += float(pseudo_loss.detach().cpu()) * len(origin_x)
            total_samples += len(origin_x)
        history.append({"epoch": epoch, "source_ce": origin_loss_sum / total_samples, "mmd": mmd_loss_sum / total_samples, "entropy_diversity": entropy_loss_sum / total_samples, "pseudo_ce": pseudo_loss_sum / total_samples, "pseudo_coverage": gmm_history[-1]["coverage"]})
        if epoch % 10 == 0 or epoch == C["adaptation"]["epochs"] - 1:
            event("epoch_completed", seed=seed, role=role, epoch=epoch, pseudo_coverage=gmm_history[-1]["coverage"])

    model.eval()
    return {"loss_history": history, "gmm_history": gmm_history, "final_pseudo_coverage": gmm_history[-1]["coverage"], "parameter_norm": math.sqrt(sum(float(value.detach().double().square().sum().cpu()) for value in model.parameters()))}


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
    support = keep[: C["adaptation"]["target_support_rows"]]
    query = keep[C["adaptation"]["target_support_rows"] :]
    assert len(set(fingerprints[index] for index in support)) == len(support)
    assert set(fingerprints[index] for index in support).isdisjoint(set(fingerprints[index] for index in query))
    return support, query, {"raw_rows": len(target["timestamps"]), "source_overlap_excluded": source_overlap, "unique_after_source_exclusion": len(keep), "support_rows": len(support), "query_rows": len(query), "query_classes": int(len(np.unique(target["labels"][query])))}


def cell_metrics(labels, frozen_predictions, predictions):
    frozen = metric(labels, frozen_predictions)
    adapted = metric(labels, predictions)
    frozen_correct = frozen_predictions == labels
    frozen_wrong = ~frozen_correct
    return {**adapted, "base_accuracy": frozen["accuracy"], "base_macro_f1": frozen["macro_f1"], "negative_adaptation_rate": float((frozen_correct & (predictions != labels)).sum() / max(1, frozen_correct.sum())), "recovery_rate": float((frozen_wrong & (predictions == labels)).sum() / max(1, frozen_wrong.sum())), "changed_rate": float((predictions != frozen_predictions).mean())}


def main():
    started = time.time()
    torch.set_num_threads(C["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert C["status"] == "frozen" and C["id"] == R.name
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("pipeline time budget exhausted")))
    signal.alarm(C["budget"]["max_wall_seconds_per_task"] * len(SEEDS) * len(ROLES))
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(C["physical_gpu"]) and torch.cuda.is_available()

    source_train = load_prepared(C["prepared"]["source_valid"], "source")
    source_valid = load_prepared(C["prepared"]["source_valid"], "valid")
    source_all = {key: np.concatenate([source_train[key], source_valid[key]], axis=0) for key in source_train}
    targets = {role: load_prepared(C["prepared"][role]) for role in ROLES}
    splits = {}
    split_report = {}
    for role, target in targets.items():
        support, query, report = split_targets(source_all, role, target)
        splits[role] = (support, query)
        split_report[role] = report
        with np.load(COMPARISON / "artifacts" / f"split_{role}.npz") as previous:
            assert np.array_equal(target["rows"][support], previous["support"])
            assert np.array_equal(target["rows"][query], previous["query"])
            assert np.array_equal(support, previous["support_indices"])
            assert np.array_equal(query, previous["query_indices"])
        np.savez_compressed(R / "artifacts" / f"split_{role}.npz", support=target["rows"][support], query=target["rows"][query], support_indices=support, query_indices=query)
    save_json("split_manifest.json", split_report)

    previous_scores = json.loads((COMPARISON / "artifacts/scores.json").read_text())
    all_rows = []
    for old_condition, new_condition in [("frozen", "frozen"), ("global", "global_entropy_adapter"), ("stepwise_rollback", "stepwise_rollback")]:
        for old in [row for row in previous_scores if row["condition"] == old_condition]:
            row = dict(old)
            row["condition"] = new_condition
            all_rows.append(row)
            source_path = COMPARISON / "artifacts" / f"pred_{old['seed']}_{old['role']}_{old_condition}.npz"
            shutil.copy2(source_path, R / "artifacts" / f"pred_{old['seed']}_{old['role']}_{new_condition}.npz")

    files = list(R.glob("*.py")) + [R / "config.json", R / "PLAN.md", R / "SOURCES.md", ROOT / "configs/datasets.json"]
    files += [ROOT / path for path in C["prepared"].values()]
    files += [ROOT / C["base_checkpoint_run"] / "checkpoints" / f"wide_bn_{seed}_best.pt" for seed in SEEDS]
    files += [COMPARISON / "artifacts" / f"split_{role}.npz" for role in ROLES]
    files += [COMPARISON / "artifacts" / "scores.json"]
    save_json("freeze.json", {str(path.relative_to(ROOT)): sha(path) for path in files})
    event("preflight", gpu=torch.cuda.get_device_name(0), split=split_report, historical_rows=len(all_rows))
    registry("running", "GPU1执行中；3日期×3seed共9个Proteus机制移植任务，100 epoch/任务，support/query严格隔离。")

    for seed in SEEDS:
        for role in ROLES:
            target = targets[role]
            support_indices, query_indices = splits[role]
            support = {key: target[key][support_indices] for key in ("timestamps", "tam", "labels", "rows")}
            query = {key: target[key][query_indices] for key in ("timestamps", "tam", "labels", "rows")}
            model, checkpoint = load_model(seed)
            initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            report = adapt_proteus(model, source_train, support, seed, role)
            query_logits = predict(model, query["timestamps"], query["tam"], C["eval_batch_size"])
            query_predictions = query_logits.argmax(1)
            with np.load(R / "artifacts" / f"pred_{seed}_{role}_frozen.npz") as frozen:
                frozen_predictions = frozen["predictions"]
                assert np.array_equal(query["rows"], frozen["rows"])
            row = {"seed": seed, "role": role, "condition": "proteus_port", "steps": 100, "rows": len(query["labels"]), **cell_metrics(query["labels"], frozen_predictions, query_predictions), **report}
            all_rows.append(row)
            np.savez_compressed(R / "artifacts" / f"pred_{seed}_{role}_proteus_port.npz", rows=query["rows"], labels=query["labels"], predictions=query_predictions, logits=query_logits)
            torch.save({"state_dict": model.state_dict(), "seed": seed, "role": role, "initial_checkpoint": str(checkpoint.relative_to(ROOT))}, R / "checkpoints" / f"proteus_{seed}_{role}.pt")
            torch.save({"initial_state": initial_state}, R / "artifacts" / f"initial_{seed}_{role}.pt")
            save_json("scores.json", all_rows)
            event("cell_completed", seed=seed, role=role, macro_f1=row["macro_f1"], pseudo_coverage=report["final_pseudo_coverage"])
            del model
            torch.cuda.empty_cache()

    save_json("scores.json", all_rows)
    registry("running", "全部36组预测（27个历史对照+9个Proteus移植）完成；正在执行独立指标、预测、checkpoint和权限审计。")
    subprocess.run([sys.executable, str(R / "postrun_audit.py")], check=True)

    def mean(condition, key):
        return float(np.mean([row[key] for row in all_rows if row["condition"] == condition]))

    verdict = {
        "proteus_minus_frozen_macro_f1_mean_pp": (mean("proteus_port", "macro_f1") - mean("frozen", "macro_f1")) * 100,
        "proteus_minus_global_macro_f1_mean_pp": (mean("proteus_port", "macro_f1") - mean("global_entropy_adapter", "macro_f1")) * 100,
        "proteus_minus_stepwise_macro_f1_mean_pp": (mean("proteus_port", "macro_f1") - mean("stepwise_rollback", "macro_f1")) * 100,
        "proteus_minus_frozen_negative_rate_pp": (mean("proteus_port", "negative_adaptation_rate") - mean("frozen", "negative_adaptation_rate")) * 100,
        "proteus_minus_global_negative_rate_pp": (mean("proteus_port", "negative_adaptation_rate") - mean("global_entropy_adapter", "negative_adaptation_rate")) * 100,
        "proteus_mean_pseudo_coverage": mean("proteus_port", "final_pseudo_coverage"),
    }
    save_json("verdict.json", verdict)
    lines = ["# Proteus 机制忠实移植 baseline 结果", "", "本 run 复现官方 Proteus 的损失项、动态带宽 MMD、GMM 伪标签和 100 epoch 更新，但底座是当前 wide+BN，适应使用独立无标签 support，query 标签只在最终预测后评分。因此结果是机制移植 baseline，不是官方 RF 端到端复现。", "", "| 条件 | Macro-F1均值 | Accuracy均值 | 负适应率均值 | 纠错率均值 |", "|---|---:|---:|---:|---:|"]
    for condition in ["frozen", "global_entropy_adapter", "stepwise_rollback", "proteus_port"]:
        lines.append(f"| {condition} | {mean(condition, 'macro_f1') * 100:.3f}% | {mean(condition, 'accuracy') * 100:.3f}% | {mean(condition, 'negative_adaptation_rate') * 100:.3f}% | {mean(condition, 'recovery_rate') * 100:.3f}% |")
    lines += ["", "裁决与对照：", "```json", json.dumps(verdict, ensure_ascii=False, indent=2), "```", "", "全部9个Proteus任务的无标签损失、GMM覆盖率、query预测和checkpoint已保存；未按query标签选epoch或阈值。"]
    (R / "RESULTS.md").write_text("\n".join(lines) + "\n")
    registry("completed", f"完成且审计通过；Proteus移植相对frozen Macro-F1 {verdict['proteus_minus_frozen_macro_f1_mean_pp']:.3f}pp，相对stepwise {verdict['proteus_minus_stepwise_macro_f1_mean_pp']:.3f}pp。")
    event("completed", elapsed_seconds=time.time() - started, verdict=verdict)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save_json("failure.json", {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        registry("failed", f"执行失败：{type(error).__name__}: {error}；产物保留。")
        raise
