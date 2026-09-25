from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ta_wf_next.traffic_views import generate_views


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
ROLES = ("source", "valid", "day14", "day30", "day90", "day150", "day270")


def metric(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    true_positive = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(102), where=denominator != 0).mean()
    return float(np.mean(labels == predictions)), float(f1)


def main() -> None:
    config = json.loads((RUN / "config.json").read_text())
    manifest = json.loads((RUN / "artifacts/manifest.json").read_text())
    prior = ROOT / manifest["sampling_manifest"]
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == manifest["sampling_sha256"]
    sampling = json.loads(prior.read_text())["sampling"]
    data_root = Path(json.loads((ROOT / "configs/datasets.json").read_text())["data_root"])
    with np.load(RUN / "artifacts/source_statistics.npz") as archive:
        mean, standard_deviation = archive["mean"], archive["std"]
    with np.load(RUN / "artifacts/predictions.npz") as archive:
        predictions = {key: archive[key] for key in archive.files}
    assert hashlib.sha256((RUN / "artifacts/predictions.npz").read_bytes()).hexdigest() == manifest["prediction_sha256"]
    with (RUN / "artifacts/metrics.csv").open(newline="") as handle:
        metrics = {(row["role"], row["condition"], int(row["seed"])): row for row in csv.DictReader(handle)}
    histories = json.loads((RUN / "artifacts/history.json").read_text())
    assert len(metrics) == len(predictions) == len(ROLES) * 2 * 3
    errors = []
    source_full = None
    all_features = {}
    for role in ROLES:
        filename = "train.npz" if role == "source" else f"{role}.npz"
        with np.load(data_root / "TemporalDrift" / filename, allow_pickle=False) as archive:
            indices = np.asarray(sampling[role]["rows"], dtype=np.int64)
            labels = archive["y"][indices].astype(np.int64)
            selected = archive["X"][indices]
        full = np.zeros((len(indices), 240), dtype=np.float32)
        neutral = np.zeros_like(full)
        for row_number, row in enumerate(selected):
            views = generate_views(row[:5000], input_kind="signed_timestamp", budget=5000,
                                   window_sizes=(50, 250))
            offset = 0
            for width, windows in views.direction_windows:
                for window_number, window in enumerate(windows):
                    column = offset + 2 * window_number
                    full[row_number, column:column + 2] = window.positive_fraction, window.transition_fraction
                    neutral[row_number, column:column + 2] = (
                        (0.5, 0.0) if window.partial else (window.positive_fraction, window.transition_fraction)
                    )
                offset += 2 * ((5000 + width - 1) // width)
        if role == "source":
            source_full = full
        all_features[role] = {"labels": labels, "full": (full - mean) / standard_deviation,
                              "tail_neutral": (neutral - mean) / standard_deviation}
    expected_mean = source_full.mean(axis=0)
    expected_std = source_full.std(axis=0)
    expected_std[expected_std < 1e-6] = 1.0
    if not (np.array_equal(mean, expected_mean) and np.array_equal(standard_deviation, expected_std)):
        errors.append("source statistics mismatch")
    for seed in config["training_seeds"]:
        for condition in config["conditions"]:
            key = f"{condition}_{seed}"
            best_epoch = max(histories[key], key=lambda item: item["macro_f1"])["epoch"]
            if best_epoch != manifest["best_epochs"][key]:
                errors.append(f"{key}: checkpoint selection mismatch")
            checkpoint = torch.load(RUN / "checkpoints" / f"{key}.pt", map_location="cpu", weights_only=True)
            if checkpoint["epoch"] != best_epoch:
                errors.append(f"{key}: saved epoch mismatch")
            model = nn.Sequential(nn.Linear(240, 128), nn.ReLU(), nn.Linear(128, 102))
            model.load_state_dict({name.removeprefix("net."): tensor for name, tensor in checkpoint["state_dict"].items()})
            model.eval()
            for role in ROLES:
                features = all_features[role][condition]
                with torch.inference_mode():
                    replay = model(torch.from_numpy(features)).argmax(dim=1).numpy()
                archived = predictions[f"{role}_{key}"]
                if not np.array_equal(replay, archived):
                    errors.append(f"{role}_{key}: prediction replay mismatch")
                accuracy, macro_f1 = metric(all_features[role]["labels"], archived)
                row = metrics[(role, condition, seed)]
                if abs(float(row["accuracy"]) - accuracy) > 1e-12 or abs(float(row["macro_f1"]) - macro_f1) > 1e-12:
                    errors.append(f"{role}_{key}: metric mismatch")
    result = {"prediction_arrays": len(predictions), "metric_rows": len(metrics),
              "checkpoint_replays": len(ROLES) * 2 * 3, "errors": errors}
    (RUN / "artifacts/integrity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
