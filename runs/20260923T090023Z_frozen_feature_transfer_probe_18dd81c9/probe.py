from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ta_wf_next.traffic_views import generate_views


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
ROLES = ("source", "valid")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return {"accuracy": float(np.mean(labels == predictions)),
            "macro_f1": float(np.divide(2 * np.diag(confusion), denominator,
                                        out=np.zeros(102), where=denominator != 0).mean())}


def packet_body() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, 32, 9, stride=4), nn.ReLU(),
        nn.Conv1d(32, 64, 7, stride=4), nn.ReLU(),
        nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(),
    )


class Teacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packet_body = packet_body()
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
        self.packet_body = packet_body()
        self.head = nn.Linear(128, 102)

    def features(self, packets: torch.Tensor) -> torch.Tensor:
        return self.packet_body(packets[:, None, :])

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(packets))


def extract_windows(raw: np.ndarray, budget: int, widths: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    dimension = 2 * sum((budget + width - 1) // width for width in widths)
    full = np.zeros((len(raw), dimension), dtype=np.float32)
    neutral = np.zeros_like(full)
    for row_number, row in enumerate(raw):
        views = generate_views(row[:budget], input_kind="signed_timestamp", budget=budget,
                               window_sizes=widths)
        offset = 0
        for width, entries in views.direction_windows:
            for window_number, entry in enumerate(entries):
                column = offset + 2 * window_number
                full[row_number, column:column + 2] = entry.positive_fraction, entry.transition_fraction
                neutral[row_number, column:column + 2] = (
                    (0.5, 0.0) if entry.partial else (entry.positive_fraction, entry.transition_fraction)
                )
            offset += 2 * ((budget + width - 1) // width)
    return full, neutral


def normalize(source: np.ndarray, valid: np.ndarray,
              floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    mean = source.mean(axis=0)
    standard_deviation = source.std(axis=0)
    constant_count = int(np.count_nonzero(standard_deviation < floor))
    standard_deviation[standard_deviation < floor] = 1.0
    return ((source - mean) / standard_deviation,
            (valid - mean) / standard_deviation, mean,
            standard_deviation, constant_count)


def fit_ridge(features: np.ndarray, targets: np.ndarray,
              alpha: float) -> tuple[np.ndarray, np.ndarray]:
    feature_mean = features.mean(axis=0)
    target_mean = targets.mean(axis=0)
    centered_features = features - feature_mean
    centered_targets = targets - target_mean
    gram = centered_features.T @ centered_features
    gram.flat[::len(gram) + 1] += alpha
    coefficients = np.linalg.solve(gram, centered_features.T @ centered_targets)
    intercept = target_mean - feature_mean @ coefficients
    return coefficients, intercept


def predict_ridge(features: np.ndarray, coefficients: np.ndarray,
                  intercept: np.ndarray) -> np.ndarray:
    return features @ coefficients + intercept


def frozen_features(teacher: Teacher, student: Student, packets: np.ndarray,
                    windows: np.ndarray, device: torch.device,
                    batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    teacher.eval()
    student.eval()
    parts = [[] for _ in range(5)]
    with torch.inference_mode():
        for start in range(0, len(packets), batch_size):
            packet_batch = torch.from_numpy(packets[start:start + batch_size]).to(device)
            window_batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
            teacher_packet, teacher_window = teacher.features(packet_batch, window_batch)
            student_packet = student.features(packet_batch)
            outputs = (teacher_packet, teacher_window, student_packet,
                       teacher.head(torch.cat((teacher_packet, teacher_window), dim=1)),
                       student.head(student_packet))
            for index, output in enumerate(outputs):
                parts[index].append(output.cpu().numpy())
    return tuple(np.concatenate(part).astype(np.float32) for part in parts)


def main() -> None:
    started = time.monotonic()
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text())
    assert config["status"] == "frozen" and config["gpu"] is True
    assert config["device"] == "cuda:0" and torch.cuda.is_available()
    assert config["roles"] == list(ROLES)
    assert not (RUN / "artifacts/metrics.json").exists()
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    device = torch.device(config["device"])
    source_run = ROOT / "runs" / config["source_run"]
    source_integrity = json.loads((source_run / "artifacts/integrity.json").read_text())
    assert source_integrity["errors"] == []
    source_manifest = json.loads((source_run / "artifacts/manifest.json").read_text())
    sampling_path = ROOT / "runs" / config["sampling_run"] / "artifacts/manifest.json"
    assert sha256(sampling_path) == source_manifest["sampling_sha256"]
    sampling = json.loads(sampling_path.read_text())["sampling"]
    data_root = Path(json.loads((ROOT / "configs/datasets.json").read_text())["data_root"])
    data = {}
    hashes = {}
    full_source = None
    for role in ROLES:
        item = sampling[role]
        with np.load(data_root / item["path"], allow_pickle=False) as archive:
            indices = np.asarray(item["rows"], dtype=np.int64)
            labels = archive["y"][indices].astype(np.int64)
            raw = archive["X"][indices]
        assert len(raw) == item["count"] and len(np.unique(labels)) == 102
        assert np.isfinite(raw).all()
        packets = np.sign(raw[:, :config["input_budget"]]).astype(np.float32)
        full, neutral = extract_windows(raw, config["input_budget"], tuple(config["window_sizes"]))
        data[role] = {"packets": packets, "windows": neutral, "labels": labels}
        hashes[role] = {hashlib.sha256(packet.astype(np.int8).tobytes()).hexdigest()
                        for packet in packets}
        if role == "source":
            full_source = full
    assert not hashes["source"] & hashes["valid"]
    window_mean = full_source.mean(axis=0)
    window_std = full_source.std(axis=0)
    window_std[window_std < 1e-6] = 1.0
    with np.load(source_run / "artifacts/window_statistics.npz") as archive:
        assert np.array_equal(window_mean, archive["mean"])
        assert np.array_equal(window_std, archive["std"])
    for role in ROLES:
        data[role]["windows"] = (data[role]["windows"] - window_mean) / window_std
    checkpoint_paths = [source_run / "checkpoints" / f"{kind}_{seed}.pt"
                        for seed in config["seeds"] for kind in ("teacher", "ce")]
    seal_paths = [RUN / "PLAN.md", config_path, Path(__file__), sampling_path,
                  source_run / "artifacts/manifest.json", source_run / "artifacts/integrity.json",
                  source_run / "artifacts/window_statistics.npz",
                  ROOT / "configs/datasets.json", ROOT / "src/ta_wf_next/traffic_views.py",
                  ROOT / "src/ta_wf_next/burst_tokens.py", *checkpoint_paths]
    (RUN / "artifacts/input_seal.json").write_text(json.dumps(
        {str(path.relative_to(ROOT)): sha256(path) for path in seal_paths}, indent=2))
    one_hot = np.eye(102, dtype=np.float64)[data["source"]["labels"]]
    metrics = {}
    probe_models = {}
    recon_models = {}
    feature_arrays = {}
    predictions = {}
    batch_size = 128
    for seed in config["seeds"]:
        teacher_checkpoint = torch.load(source_run / "checkpoints" / f"teacher_{seed}.pt",
                                        map_location="cpu", weights_only=True)
        student_checkpoint = torch.load(source_run / "checkpoints" / f"ce_{seed}.pt",
                                        map_location="cpu", weights_only=True)
        assert teacher_checkpoint["epoch"] == source_manifest["teacher_info"][str(seed)]["best_epoch"]
        assert student_checkpoint["epoch"] == source_manifest["student_info"][f"ce_{seed}"]["best_epoch"]
        teacher = Teacher().to(device)
        student = Student().to(device)
        teacher.load_state_dict(teacher_checkpoint["state_dict"])
        student.load_state_dict(student_checkpoint["student_state"])
        extracted = {role: frozen_features(teacher, student, data[role]["packets"],
                                           data[role]["windows"], device, batch_size)
                     for role in ROLES}
        teacher_anchor = metric(data["valid"]["labels"], extracted["valid"][3].argmax(axis=1))
        student_anchor = metric(data["valid"]["labels"], extracted["valid"][4].argmax(axis=1))
        assert abs(teacher_anchor["macro_f1"] - source_manifest["teacher_info"][str(seed)]["valid_macro_f1"]) < 1e-12
        assert abs(student_anchor["macro_f1"] - source_manifest["student_info"][f"ce_{seed}"]["valid_macro_f1"]) < 1e-12
        branch_arrays = {}
        constants = {}
        for branch, index in (("teacher_packet", 0), ("teacher_window", 1), ("student_packet", 2)):
            train, valid, branch_mean, branch_std, constant_count = normalize(
                extracted["source"][index], extracted["valid"][index],
                config["feature_standard_deviation_floor"])
            branch_arrays[branch] = (train.astype(np.float64), valid.astype(np.float64))
            constants[branch] = constant_count
            feature_arrays[f"{branch}_source_{seed}"] = train.astype(np.float32)
            feature_arrays[f"{branch}_valid_{seed}"] = valid.astype(np.float32)
            feature_arrays[f"{branch}_mean_{seed}"] = branch_mean
            feature_arrays[f"{branch}_std_{seed}"] = branch_std
        branch_arrays["teacher_joint"] = (
            np.concatenate((branch_arrays["teacher_packet"][0], branch_arrays["teacher_window"][0]), axis=1),
            np.concatenate((branch_arrays["teacher_packet"][1], branch_arrays["teacher_window"][1]), axis=1),
        )
        seed_metrics = {"teacher_anchor": teacher_anchor, "student_anchor": student_anchor,
                        "constant_dimensions": constants, "probes": {}, "reconstruction": {}}
        fitted_probes = {}
        for branch, (source_features, valid_features) in branch_arrays.items():
            coefficients, intercept = fit_ridge(source_features, one_hot, config["ridge_alpha"])
            fitted_probes[branch] = (coefficients, intercept)
            probe_models[f"{branch}_coef_{seed}"] = coefficients
            probe_models[f"{branch}_intercept_{seed}"] = intercept
            source_prediction = predict_ridge(source_features, coefficients, intercept).argmax(axis=1)
            valid_prediction = predict_ridge(valid_features, coefficients, intercept).argmax(axis=1)
            predictions[f"{branch}_valid_{seed}"] = valid_prediction.astype(np.int16)
            seed_metrics["probes"][branch] = {
                "source": metric(data["source"]["labels"], source_prediction),
                "valid": metric(data["valid"]["labels"], valid_prediction),
            }
        student_source, student_valid = branch_arrays["student_packet"]
        for branch in ("teacher_packet", "teacher_window"):
            target_source, target_valid = branch_arrays[branch]
            coefficients, intercept = fit_ridge(student_source, target_source,
                                                config["ridge_alpha"])
            recon_models[f"{branch}_coef_{seed}"] = coefficients
            recon_models[f"{branch}_intercept_{seed}"] = intercept
            source_prediction = predict_ridge(student_source, coefficients, intercept)
            valid_prediction = predict_ridge(student_valid, coefficients, intercept)
            source_mse = float(np.mean((target_source - source_prediction) ** 2))
            valid_mse = float(np.mean((target_valid - valid_prediction) ** 2))
            valid_r2 = 1 - float(np.sum((target_valid - valid_prediction) ** 2)
                                 / np.sum(target_valid ** 2))
            probe_coefficients, probe_intercept = fitted_probes[branch]
            recovered_prediction = predict_ridge(valid_prediction, probe_coefficients,
                                                 probe_intercept).argmax(axis=1)
            predictions[f"recovered_{branch}_valid_{seed}"] = recovered_prediction.astype(np.int16)
            seed_metrics["reconstruction"][branch] = {
                "source_mse": source_mse, "valid_mse": valid_mse,
                "valid_source_mean_r2": valid_r2,
                "recovered_probe_valid": metric(data["valid"]["labels"], recovered_prediction),
            }
        metrics[str(seed)] = seed_metrics
        print(seed, "probe F1", {key: round(value["valid"]["macro_f1"], 4)
                                 for key, value in seed_metrics["probes"].items()},
              "reconstruction R2", {key: round(value["valid_source_mean_r2"], 4)
                                    for key, value in seed_metrics["reconstruction"].items()}, flush=True)
        if time.monotonic() - started > config["time_limit_seconds"]:
            raise TimeoutError("diagnostic budget exceeded; partial run retained")
    (RUN / "artifacts/metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(RUN / "artifacts/features.npz", **feature_arrays)
    np.savez_compressed(RUN / "artifacts/probes.npz", **probe_models)
    np.savez_compressed(RUN / "artifacts/reconstructors.npz", **recon_models)
    np.savez_compressed(RUN / "artifacts/predictions.npz", **predictions)
    outputs = [RUN / "artifacts" / filename for filename in
               ("metrics.json", "features.npz", "probes.npz", "reconstructors.npz", "predictions.npz")]
    (RUN / "artifacts/output_seal.json").write_text(json.dumps(
        {str(path.relative_to(ROOT)): sha256(path) for path in outputs}, indent=2))
    print("COMPLETE", round(time.monotonic() - started, 2), "seconds", flush=True)


if __name__ == "__main__":
    main()
