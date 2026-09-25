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


def body() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, 32, 9, stride=4), nn.ReLU(),
        nn.Conv1d(32, 64, 7, stride=4), nn.ReLU(),
        nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(),
    )


class Teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packet_body = body()
        self.window_body = nn.Sequential(nn.Linear(240, 128), nn.ReLU(),
                                         nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Linear(256, 102)

    def features(self, packets: torch.Tensor, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.packet_body(packets[:, None, :]), self.window_body(windows)

    def forward(self, packets: torch.Tensor, windows: torch.Tensor) -> torch.Tensor:
        packet_features, window_features = self.features(packets, windows)
        return self.head(torch.cat((packet_features, window_features), dim=1))


class Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packet_body = body()
        self.head = nn.Linear(128, 102)

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        return self.head(self.packet_body(packets[:, None, :]))


def windows(selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full = np.zeros((len(selected), 240), dtype=np.float32)
    neutral = np.zeros_like(full)
    for row_number, row in enumerate(selected):
        views = generate_views(row[:5000], input_kind="signed_timestamp", budget=5000,
                               window_sizes=(50, 250))
        offset = 0
        for width, entries in views.direction_windows:
            for window_number, entry in enumerate(entries):
                column = offset + 2 * window_number
                full[row_number, column:column + 2] = entry.positive_fraction, entry.transition_fraction
                neutral[row_number, column:column + 2] = (
                    (0.5, 0.0) if entry.partial else (entry.positive_fraction, entry.transition_fraction)
                )
            offset += 2 * ((5000 + width - 1) // width)
    return full, neutral


def main() -> None:
    torch.set_num_threads(4)
    config = json.loads((RUN / "config.json").read_text())
    manifest = json.loads((RUN / "artifacts/manifest.json").read_text())
    prior = ROOT / manifest["sampling_manifest"]
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == manifest["sampling_sha256"]
    sampling = json.loads(prior.read_text())["sampling"]
    data_root = Path(json.loads((ROOT / "configs/datasets.json").read_text())["data_root"])
    with np.load(RUN / "artifacts/window_statistics.npz") as archive:
        mean, std = archive["mean"], archive["std"]
    with np.load(RUN / "artifacts/predictions.npz") as archive:
        predictions = {key: archive[key] for key in archive.files}
    with np.load(RUN / "artifacts/teacher_source_targets.npz") as archive:
        targets = {key: archive[key] for key in archive.files}
    assert hashlib.sha256((RUN / "artifacts/predictions.npz").read_bytes()).hexdigest() == manifest["prediction_sha256"]
    assert hashlib.sha256((RUN / "artifacts/teacher_source_targets.npz").read_bytes()).hexdigest() == manifest["teacher_target_sha256"]
    with (RUN / "artifacts/metrics.csv").open(newline="") as handle:
        metrics = {(row["role"], row["condition"], int(row["seed"])): row
                   for row in csv.DictReader(handle)}
    teacher_history = json.loads((RUN / "artifacts/teacher_history.json").read_text())
    student_history = json.loads((RUN / "artifacts/student_history.json").read_text())
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
        rebuilt[role] = {"packets": packets, "labels": labels}
        hashes[role] = {hashlib.sha256(packet.astype(np.int8).tobytes()).hexdigest()
                        for packet in packets}
        if role in ("source", "valid"):
            full, neutral = windows(selected)
            rebuilt[role]["windows"] = neutral
            if role == "source":
                source_full = full
    if hashes["source"] & hashes["valid"]:
        errors.append("source/valid direction overlap")
    expected_mean = source_full.mean(axis=0)
    expected_std = source_full.std(axis=0)
    expected_std[expected_std < 1e-6] = 1.0
    if not (np.array_equal(mean, expected_mean) and np.array_equal(std, expected_std)):
        errors.append("source window statistics mismatch")
    for role in ("source", "valid"):
        rebuilt[role]["windows"] = (rebuilt[role]["windows"] - mean) / std
    if len(predictions) != 63 or len(metrics) != 63 or len(targets) != 18:
        errors.append("expected 63 predictions, 63 metrics, 18 teacher target arrays")
    device = torch.device(config["device"])
    for seed in config["training_seeds"]:
        teacher_key = str(seed)
        selected_epoch = max(teacher_history[teacher_key], key=lambda item: item["macro_f1"])["epoch"]
        teacher_checkpoint = torch.load(RUN / "checkpoints" / f"teacher_{seed}.pt",
                                        map_location="cpu", weights_only=True)
        if teacher_checkpoint["epoch"] != selected_epoch or manifest["teacher_info"][teacher_key]["best_epoch"] != selected_epoch:
            errors.append(f"teacher_{seed}: selected epoch mismatch")
        teacher = Teacher().to(device)
        teacher.load_state_dict(teacher_checkpoint["state_dict"])
        teacher.eval()
        for role in ("source", "valid"):
            packets = rebuilt[role]["packets"]
            window_features = rebuilt[role]["windows"]
            packet_parts, window_parts, logit_parts = [], [], []
            with torch.inference_mode():
                for start in range(0, len(packets), config["batch_size"]):
                    packet_batch = torch.from_numpy(packets[start:start + config["batch_size"]]).to(device)
                    window_batch = torch.from_numpy(window_features[start:start + config["batch_size"]]).to(device)
                    packet_latent, window_latent = teacher.features(packet_batch, window_batch)
                    packet_parts.append(packet_latent.cpu().numpy())
                    window_parts.append(window_latent.cpu().numpy())
                    logit_parts.append(teacher.head(torch.cat((packet_latent, window_latent), dim=1)).cpu().numpy())
            if role == "valid":
                teacher_score = metric(rebuilt[role]["labels"], np.concatenate(logit_parts).argmax(axis=1))[1]
                if abs(teacher_score - manifest["teacher_info"][teacher_key]["valid_macro_f1"]) > 1e-12:
                    errors.append(f"teacher_{seed}: valid score mismatch")
            else:
                for branch, parts in (("packet", packet_parts), ("window", window_parts)):
                    raw_target = np.concatenate(parts)
                    expected_target_mean = raw_target.mean(axis=0)
                    expected_target_std = raw_target.std(axis=0)
                    expected_target_std[expected_target_std < config["feature_standard_deviation_floor"]] = 1.0
                    replay_target = (raw_target - expected_target_mean) / expected_target_std
                    if not np.allclose(replay_target, targets[f"{branch}_{seed}"], atol=1e-6, rtol=1e-6):
                        errors.append(f"teacher_{seed}: {branch} target replay mismatch")
                    if not np.array_equal(expected_target_mean, targets[f"{branch}_mean_{seed}"]):
                        errors.append(f"teacher_{seed}: {branch} mean mismatch")
        for condition in config["student_conditions"]:
            key = f"{condition}_{seed}"
            selected_epoch = max(student_history[key], key=lambda item: item["macro_f1"])["epoch"]
            checkpoint = torch.load(RUN / "checkpoints" / f"{key}.pt",
                                    map_location="cpu", weights_only=True)
            if checkpoint["epoch"] != selected_epoch or manifest["student_info"][key]["best_epoch"] != selected_epoch:
                errors.append(f"{key}: selected epoch mismatch")
            student = Student().to(device)
            student.load_state_dict(checkpoint["student_state"])
            student.eval()
            for role in ROLES:
                packets = rebuilt[role]["packets"]
                parts = []
                with torch.inference_mode():
                    for start in range(0, len(packets), config["batch_size"]):
                        batch = torch.from_numpy(packets[start:start + config["batch_size"]]).to(device)
                        parts.append(student(batch).argmax(dim=1).cpu().numpy())
                replay = np.concatenate(parts)
                archived = predictions[f"{role}_{key}"]
                if not np.array_equal(replay, archived):
                    errors.append(f"{role}_{key}: checkpoint replay mismatch")
                accuracy, macro_f1 = metric(rebuilt[role]["labels"], archived)
                row = metrics[(role, condition, seed)]
                if abs(float(row["accuracy"]) - accuracy) > 1e-12 or abs(float(row["macro_f1"]) - macro_f1) > 1e-12:
                    errors.append(f"{role}_{key}: metric mismatch")
    result = {"student_prediction_arrays": len(predictions), "metric_rows": len(metrics),
              "student_replays": 63, "teacher_feature_target_arrays": len(targets),
              "source_valid_overlap": len(hashes["source"] & hashes["valid"]), "errors": errors}
    (RUN / "artifacts/integrity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
