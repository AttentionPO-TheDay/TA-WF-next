from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def metric(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * np.diag(confusion), denominator,
                   out=np.zeros(102), where=denominator != 0).mean()
    return float(np.mean(labels == predictions)), float(f1)


def fit(features: np.ndarray, targets: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    centered_features = features - features.mean(axis=0)
    centered_targets = targets - targets.mean(axis=0)
    system = centered_features.T @ centered_features + alpha * np.eye(features.shape[1])
    weights = np.linalg.solve(system, centered_features.T @ centered_targets)
    bias = targets.mean(axis=0) - features.mean(axis=0) @ weights
    return weights, bias


def main() -> None:
    config = json.loads((RUN / "config.json").read_text())
    metrics = json.loads((RUN / "artifacts/metrics.json").read_text())
    sampling_path = ROOT / "runs" / config["sampling_run"] / "artifacts/manifest.json"
    sampling = json.loads(sampling_path.read_text())["sampling"]
    data_root = Path(json.loads((ROOT / "configs/datasets.json").read_text())["data_root"])
    labels = {}
    for role in config["roles"]:
        item = sampling[role]
        with np.load(data_root / item["path"], allow_pickle=False) as archive:
            labels[role] = archive["y"][item["rows"]].astype(np.int64)
    with np.load(RUN / "artifacts/features.npz") as archive:
        features = {key: archive[key] for key in archive.files}
    with np.load(RUN / "artifacts/probes.npz") as archive:
        probes = {key: archive[key] for key in archive.files}
    with np.load(RUN / "artifacts/reconstructors.npz") as archive:
        reconstructors = {key: archive[key] for key in archive.files}
    with np.load(RUN / "artifacts/predictions.npz") as archive:
        predictions = {key: archive[key] for key in archive.files}
    errors = []
    for relative_path, expected_hash in json.loads((RUN / "artifacts/input_seal.json").read_text()).items():
        if hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() != expected_hash:
            errors.append(f"input seal: {relative_path}")
    for relative_path, expected_hash in json.loads((RUN / "artifacts/output_seal.json").read_text()).items():
        if hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() != expected_hash:
            errors.append(f"output seal: {relative_path}")
    one_hot = np.eye(102)[labels["source"]]
    for seed in config["seeds"]:
        normalized = {}
        for branch in ("teacher_packet", "teacher_window", "student_packet"):
            source = features[f"{branch}_source_{seed}"].astype(np.float64)
            valid = features[f"{branch}_valid_{seed}"].astype(np.float64)
            normalized[branch] = (source, valid)
            if source.shape != (2040, 128) or valid.shape != (510, 128):
                errors.append(f"{branch}_{seed}: feature shape")
        normalized["teacher_joint"] = (
            np.concatenate((normalized["teacher_packet"][0], normalized["teacher_window"][0]), axis=1),
            np.concatenate((normalized["teacher_packet"][1], normalized["teacher_window"][1]), axis=1),
        )
        for branch, (source, valid) in normalized.items():
            weights, bias = fit(source, one_hot, config["ridge_alpha"])
            if not np.allclose(weights, probes[f"{branch}_coef_{seed}"], atol=1e-6, rtol=1e-6):
                errors.append(f"{branch}_{seed}: probe weights")
            prediction = (valid @ weights + bias).argmax(axis=1)
            if not np.array_equal(prediction, predictions[f"{branch}_valid_{seed}"]):
                errors.append(f"{branch}_{seed}: probe prediction")
            accuracy, f1 = metric(labels["valid"], prediction)
            reported = metrics[str(seed)]["probes"][branch]["valid"]
            if abs(accuracy - reported["accuracy"]) > 1e-12 or abs(f1 - reported["macro_f1"]) > 1e-12:
                errors.append(f"{branch}_{seed}: probe metric")
        student_source, student_valid = normalized["student_packet"]
        for branch in ("teacher_packet", "teacher_window"):
            target_source, target_valid = normalized[branch]
            weights, bias = fit(student_source, target_source, config["ridge_alpha"])
            if not np.allclose(weights, reconstructors[f"{branch}_coef_{seed}"], atol=1e-6, rtol=1e-6):
                errors.append(f"{branch}_{seed}: reconstruction weights")
            reconstructed = student_valid @ weights + bias
            r2 = 1 - np.sum((target_valid - reconstructed) ** 2) / np.sum(target_valid ** 2)
            reported = metrics[str(seed)]["reconstruction"][branch]
            if abs(r2 - reported["valid_source_mean_r2"]) > 1e-6:
                errors.append(f"{branch}_{seed}: R2")
            probe_weights = probes[f"{branch}_coef_{seed}"]
            probe_bias = probes[f"{branch}_intercept_{seed}"]
            recovered = (reconstructed @ probe_weights + probe_bias).argmax(axis=1)
            if not np.array_equal(recovered, predictions[f"recovered_{branch}_valid_{seed}"]):
                errors.append(f"{branch}_{seed}: recovered prediction")
            accuracy, f1 = metric(labels["valid"], recovered)
            reported_score = reported["recovered_probe_valid"]
            if abs(accuracy - reported_score["accuracy"]) > 1e-12 or abs(f1 - reported_score["macro_f1"]) > 1e-12:
                errors.append(f"{branch}_{seed}: recovered metric")
    result = {"probe_predictions": len(predictions), "feature_arrays": len(features),
              "probe_models": len(probes), "reconstruction_models": len(reconstructors),
              "errors": errors}
    (RUN / "artifacts/integrity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
