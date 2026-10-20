"""Independent audit of the three saved source checkpoints."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import MultiViewModel


RUN = Path(__file__).resolve().parent
CFG = json.loads((RUN / "config.json").read_text())
ART = RUN / "artifacts"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scores(labels, predictions):
    classes = CFG["classes"]
    cm = np.bincount(labels * classes + predictions, minlength=classes**2).reshape(classes, classes)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(classes), where=den > 0)
    return {"accuracy": float(np.mean(labels == predictions)), "macro_f1": float(f1.mean())}


def audit():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    prepared = torch.load(ART / "prepared.pt", map_location="cpu", weights_only=True)
    valid = prepared["valid"]
    labels = valid["labels"].numpy()
    rows = valid["rows"].numpy()
    reports = {r["seed"]: r for r in json.loads((ART / "reports.json").read_text())}
    assert sorted(reports) == sorted(CFG["seeds"])
    assert len(labels) == 2160 and len(np.unique(labels)) == CFG["classes"]
    out = {"passed": True, "scope": "saved valid only; CPU and GPU checkpoint reload and independent inference", "config_sha256": sha256(RUN / "config.json"), "prepared_sha256": sha256(ART / "prepared.pt"), "seeds": []}
    assert out["prepared_sha256"] == json.loads((ART / "data_preflight.json").read_text())["sha256"]["prepared"]

    for seed in CFG["seeds"]:
        report = reports[seed]
        ck_path = RUN / "checkpoints" / f"pure_source_wide_bn_{seed}.pth"
        ck_sha = sha256(ck_path)
        assert ck_sha == report["checkpoint_sha256"]
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        assert ck["seed"] == seed and ck["step"] == report["best_step"]
        assert ck["config_sha256"] == out["config_sha256"]
        assert all(torch.isfinite(t).all().item() for t in ck["state_dict"].values())
        model = MultiViewModel("wide_bn", CFG["classes"])
        model.load_state_dict(ck["state_dict"], strict=True)
        model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(labels), CFG["training"]["eval_batch_size"]):
                end = start + CFG["training"]["eval_batch_size"]
                outputs.append(model(valid["timestamps"][start:end], valid["tam"][start:end]).numpy())
        logits = np.concatenate(outputs)
        assert logits.shape == (len(labels), CFG["classes"]) and np.isfinite(logits).all()
        pred = logits.argmax(axis=1)
        saved = np.load(ART / f"valid_predictions_{seed}.npz", allow_pickle=False)
        assert np.array_equal(saved["rows"], rows)
        assert np.array_equal(saved["labels"], labels)
        assert np.array_equal(saved["predictions"], pred)
        assert np.array_equal(saved["predictions"], saved["logits"].argmax(axis=1))
        max_abs_logit_difference = float(np.max(np.abs(logits - saved["logits"])))
        assert max_abs_logit_difference < 0.01
        gpu_model = MultiViewModel("wide_bn", CFG["classes"]).to("cuda:0")
        gpu_model.load_state_dict(ck["state_dict"], strict=True)
        gpu_model.eval()
        gpu_outputs = []
        with torch.inference_mode():
            for start in range(0, len(labels), CFG["training"]["eval_batch_size"]):
                end = start + CFG["training"]["eval_batch_size"]
                gpu_outputs.append(gpu_model(valid["timestamps"][start:end].to("cuda:0"), valid["tam"][start:end].to("cuda:0")).cpu().numpy())
        gpu_logits = np.concatenate(gpu_outputs)
        gpu_difference = float(np.max(np.abs(gpu_logits - saved["logits"])))
        assert np.array_equal(gpu_logits.argmax(axis=1), saved["predictions"])
        assert np.allclose(gpu_logits, saved["logits"], rtol=1e-4, atol=1e-4)
        metrics = scores(labels, pred)
        assert all(abs(metrics[k] - report["valid"][k]) < 1e-12 for k in metrics)
        history = json.loads((ART / f"history_{seed}.json").read_text())
        assert [r["step"] for r in history] == CFG["training"]["eval_steps"]
        best = max(r["valid"]["accuracy"] for r in history)
        first_best_step = next(r["step"] for r in history if r["valid"]["accuracy"] == best)
        assert first_best_step == ck["step"] and best == metrics["accuracy"]
        out["seeds"].append({"seed": seed, "checkpoint": str(ck_path.relative_to(RUN)), "checkpoint_sha256": ck_sha, "best_step": ck["step"], "state_tensors": len(ck["state_dict"]), "parameters": sum(v.numel() for v in model.parameters()), "valid_rows": len(labels), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "max_abs_logit_difference_cpu_vs_saved": max_abs_logit_difference, "max_abs_logit_difference_gpu_vs_saved": gpu_difference, "predictions_match": True, "history_selection_match": True})
    (ART / "weight_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    audit()
