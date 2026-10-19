"""Bounded source-only training; one independent model per seed."""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F
from model import MultiViewModel, span_mask

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
CFG = json.loads((RUN / "config.json").read_text())
DEVICE = "cuda:0"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def registry(state, summary):
    subprocess.run([sys.executable, str(ROOT / "scripts/experiment.py"), "status", "--id", RUN.name, "--state", state, "--summary", summary], check=True)


def metric(y, pred):
    cm = np.bincount(y * 102 + pred, minlength=102 * 102).reshape(102, 102)
    den = cm.sum(0) + cm.sum(1)
    return {"accuracy": float((y == pred).mean()), "macro_f1": float(np.divide(2 * cm.diagonal(), den, out=np.zeros(102), where=den > 0).mean())}


def predict(model, data):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for i in range(0, len(data["labels"]), CFG["training"]["eval_batch_size"]):
            z = model(data["timestamps"][i:i + 64], data["tam"][i:i + 64])
            outputs.append(z.cpu().numpy())
    out = np.concatenate(outputs)
    assert out.shape == (len(data["labels"]), 102) and np.isfinite(out).all()
    return out


def learning_rate(step):
    for first, last, lr in CFG["training"]["lr_schedule"]:
        if first <= step <= last:
            return lr
    raise ValueError(step)


def train_seed(seed, data):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MultiViewModel("wide_bn", 102).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=CFG["training"]["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
    source, valid = data["source"], data["valid"]
    n, bs, steps = len(source["labels"]), CFG["training"]["batch_size"], CFG["training"]["steps"]
    pieces, cycle, count = [], 0, 0
    while count < steps * bs:
        pieces.append(torch.randperm(n, generator=torch.Generator().manual_seed(seed + 9000 + cycle)))
        cycle += 1
        count += n
    indices = torch.cat(pieces)[:steps * bs].reshape(steps, bs)
    aug_rng = torch.Generator().manual_seed(seed + 50000)
    history, best, best_step = [], -1.0, 0
    start = time.monotonic()
    ck_path = RUN / "checkpoints" / f"pure_source_wide_bn_{seed}.pth"
    y_valid = valid["labels"].cpu().numpy()
    for step in range(1, steps + 1):
        if time.monotonic() - start > CFG["budget"]["max_seconds_per_seed"]:
            raise TimeoutError(f"seed {seed} exceeded per-seed budget")
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        idx = indices[step - 1].to(DEVICE)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x, tam, y = source["timestamps"][idx], span_mask(source["tam"][idx], aug_rng), source["labels"][idx]
        logits = model(x, tam)
        loss = F.cross_entropy(logits, y)
        assert torch.isfinite(loss), f"nonfinite loss at seed={seed} step={step}"
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            save_json(RUN / "artifacts" / f"progress_{seed}.json", {"seed": seed, "step": step, "total_steps": steps, "best_step": best_step, "elapsed_seconds": time.monotonic() - start})
        if step in CFG["training"]["eval_steps"]:
            z = predict(model, valid)
            scores = metric(y_valid, z.argmax(1))
            row = {"step": step, "lr": lr, "train_loss_last": float(loss.detach()), "valid": scores, "elapsed_seconds": time.monotonic() - start}
            history.append(row)
            if scores["accuracy"] > best:
                best, best_step = scores["accuracy"], step
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "seed": seed, "step": step, "config_sha256": digest(RUN / "config.json")}, ck_path)
            save_json(RUN / "artifacts" / f"history_{seed}.json", history)
            print(json.dumps({"seed": seed, **row}), flush=True)
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    assert ck["step"] == best_step and ck["config_sha256"] == digest(RUN / "config.json")
    reloaded = MultiViewModel("wide_bn", 102).to(DEVICE)
    reloaded.load_state_dict(ck["state_dict"], strict=True)
    z = predict(reloaded, valid)
    scores = metric(y_valid, z.argmax(1))
    assert scores["accuracy"] == best
    np.savez_compressed(RUN / "artifacts" / f"valid_predictions_{seed}.npz", labels=y_valid, predictions=z.argmax(1), logits=z, rows=valid["rows"].cpu().numpy())
    report = {"seed": seed, "best_step": best_step, "steps": steps, "valid": scores, "elapsed_seconds": time.monotonic() - start, "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()), "checkpoint_sha256": digest(ck_path)}
    save_json(RUN / "artifacts" / f"report_{seed}.json", report)
    print(json.dumps({"seed_completed": report}), flush=True)
    del model, reloaded, optimizer
    torch.cuda.empty_cache()
    return report


def main():
    assert CFG["status"] == "frozen" and CFG["id"] == RUN.name
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(CFG["budget"]["physical_gpu"])
    assert torch.cuda.is_available()
    frozen = json.loads((RUN / "artifacts/freeze.json").read_text())
    for relative, expected in frozen.items():
        assert digest(ROOT / relative) == expected, f"frozen file changed: {relative}"
    preflight = json.loads((RUN / "artifacts/data_preflight.json").read_text())
    assert preflight["passed"] and preflight["sha256"]["prepared"] == digest(RUN / "artifacts/prepared.pt")
    torch.set_num_threads(CFG["budget"]["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    registry("running", "18,553条保守源域行的wide_bn三seed从头训练启动。")
    raw = torch.load(RUN / "artifacts/prepared.pt", map_location="cpu", weights_only=False)
    data = {role: {k: v.to(DEVICE) for k, v in entry.items()} for role, entry in raw.items()}
    started = time.monotonic()
    reports = []
    for seed in CFG["seeds"]:
        if time.monotonic() - started > CFG["budget"]["max_total_seconds"]:
            raise TimeoutError("total training budget exceeded")
        reports.append(train_seed(seed, data))
        save_json(RUN / "artifacts/reports.json", reports)
    assert len(reports) == len(CFG["seeds"])
    accuracy = np.array([r["valid"]["accuracy"] for r in reports])
    f1 = np.array([r["valid"]["macro_f1"] for r in reports])
    result = ["# 纯净源域 wide_bn 基石模型结果", "", "完成三 seed 从头训练、valid 预测与 checkpoint 重载。源期指标来自已观察的官方 valid；无独立 Day 0 test。", "", "| Seed | Best step | Valid Accuracy | Valid Macro-F1 |", "|---:|---:|---:|---:|"]
    for r in reports:
        result.append(f"| {r['seed']} | {r['best_step']} | {r['valid']['accuracy']*100:.3f}% | {r['valid']['macro_f1']*100:.3f}% |")
    result += ["", f"三 seed 均值±样本标准差：Accuracy {accuracy.mean()*100:.3f}±{accuracy.std(ddof=1)*100:.3f}%，Macro-F1 {f1.mean()*100:.3f}±{f1.std(ddof=1)*100:.3f}%。", "", "输入清单和源文件散列见 artifacts/data_preflight.json；逐 seed checkpoint、history、预测与重载报告见 checkpoints/ 和 artifacts/。未访问未来日期或外部数据。"]
    (RUN / "RESULTS.md").write_text("\n".join(result) + "\n")
    registry("completed", "纯净源域wide_bn三seed训练及valid重载完成；结果见RESULTS.md。")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        save_json(RUN / "artifacts/failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        registry("failed", f"纯净源域训练失败：{type(exc).__name__}: {exc}；产物保留。")
        raise
