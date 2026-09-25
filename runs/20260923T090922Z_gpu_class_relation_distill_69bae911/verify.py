from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
ROLES = ("source", "valid", "day14", "day30", "day90", "day150", "day270")
CONDITIONS = ("ce", "relation_packet", "relation_window")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    table = np.bincount(102 * y + p, minlength=102**2).reshape(102, 102)
    denominator = table.sum(0) + table.sum(1)
    f1 = np.divide(2 * np.diag(table), denominator, out=np.zeros(102), where=denominator != 0)
    return float(np.mean(y == p)), float(f1.mean())


class Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packet_body = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=4), nn.ReLU(),
            nn.Conv1d(32, 64, 7, stride=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(1024, 128), nn.ReLU())
        self.head = nn.Linear(128, 102)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.packet_body(x[:, None, :]))


def main() -> None:
    torch.set_num_threads(4)
    config = json.loads((RUN / "config.json").read_text())
    manifest = json.loads((RUN / "artifacts/manifest.json").read_text())
    sealed = json.loads((RUN / "artifacts/input_seal.json").read_text())
    errors: list[str] = []
    for relative, expected in sealed.items():
        if digest(ROOT / relative) != expected:
            errors.append(f"input seal mismatch: {relative}")
    sampling_path = ROOT / manifest["sampling_manifest"]
    if digest(sampling_path) != manifest["sampling_sha256"]:
        errors.append("sampling manifest hash mismatch")
    sampling = json.loads(sampling_path.read_text())["sampling"]
    feature_run = ROOT / "runs" / config["teacher_feature_run"]
    feature_path = feature_run / "artifacts/features.npz"
    if digest(feature_path) != manifest["teacher_features_sha256"]:
        errors.append("teacher feature hash mismatch")
    if json.loads((feature_run / "artifacts/integrity.json").read_text())["errors"]:
        errors.append("upstream teacher integrity failed")
    if json.loads((feature_run / "artifacts/output_seal.json").read_text())[str(feature_path.relative_to(ROOT))] != digest(feature_path):
        errors.append("upstream teacher output seal mismatch")
    if json.loads((feature_run / "artifacts/input_seal.json").read_text())[str(sampling_path.relative_to(ROOT))] != digest(sampling_path):
        errors.append("upstream sampling seal mismatch")

    data_root = Path(json.loads((ROOT / "configs/datasets.json").read_text())["data_root"])
    data = {}
    hashes = {}
    for role in ROLES:
        item = sampling[role]
        with np.load(data_root / item["path"], allow_pickle=False) as archive:
            rows = np.asarray(item["rows"], dtype=np.int64)
            if len(rows) != item["count"] or len(np.unique(rows)) != len(rows):
                errors.append(f"{role}: invalid frozen rows")
            labels = archive["y"][rows].astype(np.int64)
            packets = np.sign(archive["X"][rows, :config["input_budget"]]).astype(np.float32)
        if len(np.unique(labels)) != 102 or not np.isfinite(packets).all():
            errors.append(f"{role}: invalid labels or packets")
        data[role] = (packets, labels)
        hashes[role] = {hashlib.sha256(row.astype(np.int8).tobytes()).hexdigest() for row in packets}
    overlap = len(hashes["source"] & hashes["valid"])
    if overlap:
        errors.append("source/valid direction overlap")
    if not np.all(np.bincount(data["source"][1], minlength=102) == 20):
        errors.append("source class count mismatch")

    with np.load(feature_path, allow_pickle=False) as archive:
        features = {f"{branch}_{seed}": archive[f"teacher_{branch}_source_{seed}"]
                    for seed in config["training_seeds"] for branch in ("packet", "window")}
    relation_path = RUN / "artifacts/relations.npz"
    prediction_path = RUN / "artifacts/predictions.npz"
    if digest(relation_path) != manifest["relation_sha256"] or digest(prediction_path) != manifest["prediction_sha256"]:
        errors.append("output hash mismatch")
    with np.load(relation_path, allow_pickle=False) as archive:
        relations = {key: archive[key] for key in archive.files}
    with np.load(prediction_path, allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in archive.files}
    if len(relations) != 18 or len(predictions) != 63:
        errors.append("array count mismatch")
    labels = data["source"][1]
    relation_replays = 0
    for seed in config["training_seeds"]:
        for branch in ("packet", "window"):
            key = f"{branch}_{seed}"
            x = features[key]
            if x.shape != (2040, 128) or not np.isfinite(x).all():
                errors.append(f"{key}: teacher feature shape/value mismatch")
                continue
            sums = np.stack([x[labels == c].sum(0) for c in range(102)])
            unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
            proto = unit(sums / 20).astype(np.float32)
            loo = unit((sums[labels] - x) / 19).astype(np.float32)
            similarity = unit(x) @ proto.T
            similarity[np.arange(len(labels)), labels] = np.sum(unit(x) * loo, axis=1)
            logits = similarity * config["relation_scale"]
            logits -= logits.max(1, keepdims=True)
            exp = np.exp(logits)
            probs = (exp / exp.sum(1, keepdims=True)).astype(np.float32)
            for suffix, rebuilt in (("prototypes", proto), ("leave_one_out", loo), ("probabilities", probs)):
                stored = relations[f"{branch}_{suffix}_{seed}"]
                if not np.allclose(stored, rebuilt, atol=2e-6, rtol=2e-6):
                    errors.append(f"{key}: {suffix} replay mismatch")
                relation_replays += 1
            info = manifest["relation_info"][key]
            top1 = np.mean(probs.argmax(1) == labels)
            entropy = np.mean(-np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1))
            if abs(top1 - info["source_true_class_top1"]) > 1e-12 or abs(entropy - info["source_mean_entropy"]) > 1e-5:
                errors.append(f"{key}: relation summary mismatch")

    history = json.loads((RUN / "artifacts/history.json").read_text())
    with (RUN / "artifacts/metrics.csv").open(newline="") as handle:
        rows = {(row["role"], row["condition"], int(row["seed"])): row for row in csv.DictReader(handle)}
    if len(history) != 9 or len(rows) != 63:
        errors.append("history/metric row count mismatch")
    replay_count = 0
    for seed in config["training_seeds"]:
        init_hashes = set()
        for condition in CONDITIONS:
            key = f"{condition}_{seed}"
            selected = max(history[key], key=lambda row: row["macro_f1"])
            info = manifest["model_info"][key]
            checkpoint = torch.load(RUN / "checkpoints" / f"{key}.pt", map_location="cpu", weights_only=True)
            if len(history[key]) != config["student_epochs"] or [row["epoch"] for row in history[key]] != list(range(1, config["student_epochs"] + 1)):
                errors.append(f"{key}: history length/epochs mismatch")
            if checkpoint["epoch"] != selected["epoch"] or info["best_epoch"] != selected["epoch"] or abs(info["valid_macro_f1"] - selected["macro_f1"]) > 1e-12:
                errors.append(f"{key}: selection mismatch")
            init_hashes.add(info["initial_hash"])
            model = Student()
            model.load_state_dict(checkpoint["student_state"])
            model.eval()
            if sum(p.numel() for p in model.parameters()) != info["inference_parameters"]:
                errors.append(f"{key}: parameter count mismatch")
            for role in ROLES:
                packets, y = data[role]
                with torch.inference_mode():
                    replay = np.concatenate([model(torch.from_numpy(packets[start:start + config["batch_size"]])).argmax(1).numpy()
                                             for start in range(0, len(packets), config["batch_size"])])
                if not np.array_equal(replay, predictions[f"{role}_{key}"]):
                    errors.append(f"{role}_{key}: checkpoint prediction mismatch")
                accuracy, f1 = score(y, predictions[f"{role}_{key}"])
                row = rows[(role, condition, seed)]
                if int(row["best_epoch"]) != selected["epoch"] or abs(float(row["accuracy"]) - accuracy) > 1e-12 or abs(float(row["macro_f1"]) - f1) > 1e-12:
                    errors.append(f"{role}_{key}: metric mismatch")
                replay_count += 1
        if len(init_hashes) != 1:
            errors.append(f"{seed}: initialization mismatch")

    result = {"source_valid_direction_overlap": overlap, "relation_arrays_replayed": relation_replays,
              "checkpoint_predictions_replayed": replay_count, "metric_rows": len(rows), "errors": errors}
    (RUN / "artifacts/integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
