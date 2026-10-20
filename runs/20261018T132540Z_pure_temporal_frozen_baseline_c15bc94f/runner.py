"""Nine frozen CPU inference cells on the already-observed TemporalDrift dates."""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from model import MultiViewModel

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / "config.json").read_text())
ART = RUN / "artifacts"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def registry(state, summary):
    subprocess.run([sys.executable, str(ROOT / "scripts/experiment.py"), "status", "--id", RUN.name, "--state", state, "--summary", summary], check=True)


def fp_rows(values):
    return [hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest() for row in values]


def metrics(labels, predictions):
    n = C["classes"]
    cm = np.bincount(labels * n + predictions, minlength=n * n).reshape(n, n)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(n), where=den > 0)
    return {"accuracy": float(np.mean(labels == predictions)), "macro_f1": float(f1.mean())}


def predict(model, timestamps, tam):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(timestamps), C["budget"]["batch_size"]):
            end = start + C["budget"]["batch_size"]
            parts.append(model(timestamps[start:end], tam[start:end]).numpy())
    logits = np.concatenate(parts)
    assert logits.shape == (len(timestamps), C["classes"]) and np.isfinite(logits).all()
    return logits


def main():
    assert C["id"] == RUN.name and C["status"] == "frozen"
    assert C["budget"]["device"] == "cpu" and not torch.cuda.is_initialized()
    torch.set_num_threads(C["budget"]["threads"])
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    source_run = ROOT / C["source_run"]
    source_reports = {r["seed"]: r for r in json.loads((source_run / "artifacts/reports.json").read_text())}
    source_audit = json.loads((source_run / "artifacts/weight_audit.json").read_text())
    assert source_audit["passed"] and sorted(source_reports) == sorted(C["seeds"])
    source_path = source_run / "artifacts/prepared.pt"
    prepared = torch.load(source_path, map_location="cpu", weights_only=True)
    source_fps = set(fp_rows(prepared["source"]["timestamps"].numpy()))
    source_fps.update(fp_rows(prepared["valid"]["timestamps"].numpy()))
    assert len(source_fps) >= 18553
    del prepared
    frozen_files = [RUN / name for name in ("PLAN.md", "SOURCES.md", "config.json", "runner.py", "model.py", "base_model.py", "candidate_base.py", "encoder_base.py", "framework_base.py", "generator_base.py", "input_base.py", "rf_native.py", "view_base.py")]
    frozen_files += [ROOT / C["dataset_config"], source_path, source_run / "artifacts/weight_audit.json"]
    for seed in C["seeds"]:
        ck_path = source_run / "checkpoints" / f"pure_source_wide_bn_{seed}.pth"
        assert digest(ck_path) == source_reports[seed]["checkpoint_sha256"]
        frozen_files.append(ck_path)
    for role, relative in C["targets"].items():
        frozen_files.extend([ROOT / relative, ROOT / C["split_source_run"] / "artifacts" / f"split_{role}.npz"])
    freeze = {str(path.relative_to(ROOT)): digest(path) for path in frozen_files}
    save_json(ART / "freeze.json", freeze)

    # Audit all three independent splits before any target forward or score.
    splits = {}
    for role, relative in C["targets"].items():
        if time.monotonic() - started > C["budget"]["max_wall_seconds"]:
            raise TimeoutError("wall budget exhausted")
        data = torch.load(ROOT / relative, map_location="cpu", weights_only=True)
        assert set(("timestamps", "tam", "labels", "rows")).issubset(data)
        assert data["timestamps"].shape[1] == 5000 and data["tam"].shape[1:] == (2, 1800)
        with np.load(ROOT / C["split_source_run"] / "artifacts" / f"split_{role}.npz", allow_pickle=False) as split:
            support_indices = split["support_indices"]
            query_indices = split["query_indices"]
            assert np.array_equal(data["rows"][support_indices].numpy(), split["support"])
            assert np.array_equal(data["rows"][query_indices].numpy(), split["query"])
        assert len(support_indices) == C["reference_rows"]
        assert len(set(support_indices) & set(query_indices)) == 0
        support_fps = set(fp_rows(data["timestamps"][support_indices].numpy()))
        query_fps = fp_rows(data["timestamps"][query_indices].numpy())
        assert len(support_fps) == len(support_indices)
        assert len(set(query_fps)) == len(query_fps)
        assert source_fps.isdisjoint(support_fps | set(query_fps))
        assert support_fps.isdisjoint(query_fps)
        save_json(ART / f"split_audit_{role}.json", {"role": role, "reference_rows": len(support_indices), "query_rows": len(query_indices), "source_exact_overlap": 0, "reference_query_exact_overlap": 0, "source_checkpoint_run": C["source_run"], "split_source_run": C["split_source_run"]})
        splits[role] = query_indices
        del data
    registry("running", "三日期split与纯净源/valid精确隔离通过；CPU冻结推断9单元启动。")
    scores = []
    for role, relative in C["targets"].items():
        data = torch.load(ROOT / relative, map_location="cpu", weights_only=True)
        query_indices = splits[role]
        x, tam = data["timestamps"][query_indices], data["tam"][query_indices]
        rows = data["rows"][query_indices].numpy()
        for seed in C["seeds"]:
            if time.monotonic() - started > C["budget"]["max_wall_seconds"]:
                raise TimeoutError("wall budget exhausted")
            checkpoint = torch.load(source_run / "checkpoints" / f"pure_source_wide_bn_{seed}.pth", map_location="cpu", weights_only=True)
            assert checkpoint["seed"] == seed and checkpoint["step"] == source_reports[seed]["best_step"]
            model = MultiViewModel("wide_bn", C["classes"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            begun = time.monotonic()
            logits = predict(model, x, tam)
            inference_seconds = time.monotonic() - begun
            predictions = logits.argmax(axis=1)
            labels = data["labels"][query_indices].numpy()
            assert labels.shape == predictions.shape and len(np.unique(labels)) == C["classes"]
            probabilities = torch.softmax(torch.from_numpy(logits), dim=1)
            entropy = float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)).mean())
            norms = np.linalg.norm(logits, axis=1)
            result = {"seed": seed, "role": role, "condition": "frozen", "rows": len(labels), **metrics(labels, predictions), "mean_entropy_nats": entropy, "mean_logit_l2": float(norms.mean()), "inference_seconds": inference_seconds, "checkpoint_sha256": source_reports[seed]["checkpoint_sha256"]}
            out_path = ART / f"pred_{seed}_{role}_frozen.npz"
            with out_path.with_suffix(".npz.tmp").open("wb") as stream:
                np.savez_compressed(stream, rows=rows, labels=labels, predictions=predictions, logits=logits)
            out_path.with_suffix(".npz.tmp").replace(out_path)
            scores.append(result)
            save_json(ART / "scores.json", scores)
            print(json.dumps(result), flush=True)
            del model, checkpoint, logits
        del data, x, tam
    assert len(scores) == C["budget"]["cells"]
    assert sum(path.stat().st_size for path in ART.rglob("*") if path.is_file()) < C["budget"]["max_output_gib"] * 1024**3
    lines = ["# 纯净源域 TemporalDrift 冻结退化基线", "", "三份纯净源域权重在固定独立 query 上完成9个冻结推断单元。仅为已观察开发证据。", "", "| Day | Mean Accuracy ± SD | Mean Macro-F1 ± SD |", "|---|---:|---:|"]
    for role in C["targets"]:
        picked = [r for r in scores if r["role"] == role]
        acc = np.array([r["accuracy"] for r in picked]) * 100
        f1 = np.array([r["macro_f1"] for r in picked]) * 100
        lines.append(f"| {role} | {acc.mean():.3f} ± {acc.std(ddof=1):.3f}% | {f1.mean():.3f} ± {f1.std(ddof=1):.3f}% |")
    lines += ["", "逐seed指标及预测见 artifacts/scores.json 和 pred_*.npz；输入/代码散列见 artifacts/freeze.json。无TTA、目标标签训练或外部评分。", ""]
    (RUN / "RESULTS.md").write_text("\n".join(lines))
    registry("completed", "纯净源域三seed×Day14/90/270冻结基线9单元完成；结果见RESULTS.md。")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        save_json(ART / "failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        registry("failed", f"纯净源域Temporal冻结推断失败：{type(exc).__name__}；证据保留。")
        raise
