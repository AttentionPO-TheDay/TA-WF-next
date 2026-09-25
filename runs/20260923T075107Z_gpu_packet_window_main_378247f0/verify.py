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
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * np.diag(confusion), denominator,
                   out=np.zeros(102), where=denominator != 0).mean()
    return float(np.mean(labels == predictions)), float(f1)


class ReplayModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packet_body = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=4), nn.ReLU(),
            nn.Conv1d(32, 64, 7, stride=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(),
        )
        self.window_body = nn.Sequential(nn.Linear(240, 128), nn.ReLU(),
                                         nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Linear(256, 102)

    def forward(self, packets: torch.Tensor, windows: torch.Tensor) -> torch.Tensor:
        packet_features = self.packet_body(packets[:, None, :])
        window_features = self.window_body(windows)
        return self.head(torch.cat((packet_features, window_features), axis=1))


def main() -> None:
    torch.set_num_threads(4)
    config = json.loads((RUN / "config.json").read_text())
    manifest = json.loads((RUN / "artifacts/manifest.json").read_text())
    prior = ROOT / manifest["sampling_manifest"]
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == manifest["sampling_sha256"]
    sampling = json.loads(prior.read_text())["sampling"]
    data_root = Path(json.loads((ROOT / "configs/datasets.json").read_text())["data_root"])
    with np.load(RUN / "artifacts/window_statistics.npz") as archive:
        saved_mean, saved_std = archive["mean"], archive["std"]
    with np.load(RUN / "artifacts/predictions.npz") as archive:
        predictions = {key: archive[key] for key in archive.files}
    assert hashlib.sha256((RUN / "artifacts/predictions.npz").read_bytes()).hexdigest() == manifest["prediction_sha256"]
    with (RUN / "artifacts/metrics.csv").open(newline="") as handle:
        metrics = {(row["role"], row["condition"], int(row["seed"])): row
                   for row in csv.DictReader(handle)}
    histories = json.loads((RUN / "artifacts/history.json").read_text())
    errors = []
    rebuilt = {}
    hashes = {}
    source_full = None
    for role in ROLES:
        item = sampling[role]
        with np.load(data_root / item["path"], allow_pickle=False) as archive:
            indices = np.asarray(item["rows"], dtype=np.int64)
            labels = archive["y"][indices].astype(np.int64)
            selected = archive["X"][indices]
        packets = np.sign(selected[:, :5000]).astype(np.float32)
        full = np.zeros((len(selected), 240), dtype=np.float32)
        neutral = np.zeros_like(full)
        for row_number, row in enumerate(selected):
            views = generate_views(row[:5000], input_kind="signed_timestamp",
                                   budget=5000, window_sizes=(50, 250))
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
        hashes[role] = {hashlib.sha256(packet.astype(np.int8).tobytes()).hexdigest()
                        for packet in packets}
        rebuilt[role] = {"labels": labels, "packets": packets, "windows": neutral}
    if hashes["source"] & hashes["valid"]:
        errors.append("source/valid direction overlap")
    mean = source_full.mean(axis=0)
    std = source_full.std(axis=0)
    std[std < 1e-6] = 1.0
    if not (np.array_equal(mean, saved_mean) and np.array_equal(std, saved_std)):
        errors.append("source normalization mismatch")
    for role in ROLES:
        rebuilt[role]["windows"] = (rebuilt[role]["windows"] - mean) / std
    if len(predictions) != 63 or len(metrics) != 63:
        errors.append("expected 63 prediction arrays and metric rows")
    device = torch.device(config["device"])
    for seed in config["training_seeds"]:
        for condition in config["conditions"]:
            key = f"{condition}_{seed}"
            earliest_best = max(histories[key], key=lambda epoch: epoch["macro_f1"])["epoch"]
            checkpoint = torch.load(RUN / "checkpoints" / f"{key}.pt",
                                    map_location="cpu", weights_only=True)
            if checkpoint["epoch"] != earliest_best or manifest["best_epochs"][key] != earliest_best:
                errors.append(f"{key}: epoch selection mismatch")
            model = ReplayModel().to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            for role in ROLES:
                labels = rebuilt[role]["labels"]
                packets = rebuilt[role]["packets"]
                windows = rebuilt[role]["windows"]
                if condition == "packet_only":
                    windows = np.zeros_like(windows)
                elif condition == "window_only":
                    packets = np.zeros_like(packets)
                parts = []
                with torch.inference_mode():
                    for start in range(0, len(labels), config["batch_size"]):
                        packet_batch = torch.from_numpy(packets[start:start + config["batch_size"]]).to(device)
                        window_batch = torch.from_numpy(windows[start:start + config["batch_size"]]).to(device)
                        parts.append(model(packet_batch, window_batch).argmax(axis=1).cpu().numpy())
                replay = np.concatenate(parts)
                archived = predictions[f"{role}_{key}"]
                if not np.array_equal(replay, archived):
                    errors.append(f"{role}_{key}: checkpoint replay mismatch")
                accuracy, macro_f1 = metric(labels, archived)
                row = metrics[(role, condition, seed)]
                if abs(float(row["accuracy"]) - accuracy) > 1e-12 or abs(float(row["macro_f1"]) - macro_f1) > 1e-12:
                    errors.append(f"{role}_{key}: metric mismatch")
    result = {"prediction_arrays": len(predictions), "metric_rows": len(metrics),
              "checkpoint_replays": len(ROLES) * len(config["conditions"]) * len(config["training_seeds"]),
              "source_valid_overlap": len(hashes["source"] & hashes["valid"]), "errors": errors}
    (RUN / "artifacts/integrity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
