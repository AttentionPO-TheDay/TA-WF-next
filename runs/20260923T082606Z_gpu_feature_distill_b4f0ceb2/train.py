from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ta_wf_next.traffic_views import generate_views


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PRIOR = ROOT / "runs/20260922T023810Z_cpu_temporal_multiview_screen_v2_34c00b37/artifacts/manifest.json"
ROLES = ("source", "valid", "day14", "day30", "day90", "day150", "day270")
CONDITIONS = ("ce", "align_packet", "align_window")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    denominator = confusion.sum(0) + confusion.sum(1)
    return {"accuracy": float(np.mean(labels == predictions)),
            "macro_f1": float(np.divide(2 * np.diag(confusion), denominator,
                                        out=np.zeros(102), where=denominator != 0).mean())}


def extract_windows(rows: np.ndarray, budget: int, widths: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    dimension = 2 * sum((budget + width - 1) // width for width in widths)
    full = np.zeros((len(rows), dimension), dtype=np.float32)
    neutral = np.zeros_like(full)
    for row_number, row in enumerate(rows):
        views = generate_views(row[:budget], input_kind="signed_timestamp", budget=budget, window_sizes=widths)
        offset = 0
        for width, windows in views.direction_windows:
            for window_number, window in enumerate(windows):
                column = offset + 2 * window_number
                full[row_number, column:column + 2] = window.positive_fraction, window.transition_fraction
                neutral[row_number, column:column + 2] = (
                    (0.5, 0.0) if window.partial else (window.positive_fraction, window.transition_fraction)
                )
            offset += 2 * ((budget + width - 1) // width)
    return full, neutral


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


def initialize(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initial_hash(modules: tuple[nn.Module, ...]) -> str:
    return hashlib.sha256(b"".join(parameter.detach().cpu().numpy().tobytes()
                                    for module in modules for parameter in module.parameters())).hexdigest()


def teacher_outputs(model: Teacher, packets: np.ndarray, windows: np.ndarray,
                    device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits, packet_features, window_features = [], [], []
    with torch.inference_mode():
        for start in range(0, len(packets), batch_size):
            packet_batch = torch.from_numpy(packets[start:start + batch_size]).to(device)
            window_batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
            packet_latent, window_latent = model.features(packet_batch, window_batch)
            logits.append(model.head(torch.cat((packet_latent, window_latent), dim=1)).cpu().numpy())
            packet_features.append(packet_latent.cpu().numpy())
            window_features.append(window_latent.cpu().numpy())
    return tuple(np.concatenate(part).astype(np.float32) for part in
                 (logits, packet_features, window_features))


def student_predictions(model: Student, packets: np.ndarray,
                        device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(packets), batch_size):
            batch = torch.from_numpy(packets[start:start + batch_size]).to(device)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int16)


def main() -> None:
    started = time.monotonic()
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text())
    assert config["status"] == "frozen" and config["gpu"] is True
    assert config["device"] == "cuda:0" and torch.cuda.is_available()
    assert config["student_conditions"] == list(CONDITIONS)
    assert not (RUN / "artifacts/predictions.npz").exists()
    assert not list((RUN / "checkpoints").glob("*.pt"))
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    device = torch.device(config["device"])
    budget = config["input_budget"]
    widths = tuple(config["window_sizes"])
    data_root = Path(json.loads((ROOT / "configs/datasets.json").read_text())["data_root"])
    sampling = json.loads(PRIOR.read_text())["sampling"]
    data = {}
    direction_hashes = {}
    full_source_windows = None
    for role in ROLES:
        item = sampling[role]
        with np.load(data_root / item["path"], allow_pickle=False) as archive:
            indices = np.asarray(item["rows"], dtype=np.int64)
            labels = archive["y"][indices].astype(np.int64)
            raw = archive["X"][indices]
        assert len(raw) == item["count"] and len(np.unique(labels)) == 102
        assert np.isfinite(raw).all()
        packets = np.sign(raw[:, :budget]).astype(np.float32)
        data[role] = {"packets": packets, "labels": labels}
        direction_hashes[role] = {hashlib.sha256(packet.astype(np.int8).tobytes()).hexdigest()
                                  for packet in packets}
        if role in ("source", "valid"):
            full, neutral = extract_windows(raw, budget, widths)
            data[role]["windows"] = neutral
            if role == "source":
                full_source_windows = full
        print(f"extracted {role}: {len(raw)}", flush=True)
    assert not direction_hashes["source"] & direction_hashes["valid"]
    mean = full_source_windows.mean(axis=0)
    standard_deviation = full_source_windows.std(axis=0)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    np.savez_compressed(RUN / "artifacts/window_statistics.npz", mean=mean, std=standard_deviation)
    for role in ("source", "valid"):
        data[role]["windows"] = (data[role]["windows"] - mean) / standard_deviation
        assert np.isfinite(data[role]["windows"]).all()
    seal_paths = (RUN / "PLAN.md", config_path, Path(__file__), PRIOR,
                  ROOT / "configs/datasets.json", ROOT / "src/ta_wf_next/traffic_views.py",
                  ROOT / "src/ta_wf_next/burst_tokens.py", RUN / "artifacts/window_statistics.npz")
    (RUN / "artifacts/pretraining_seal.json").write_text(json.dumps(
        {str(path.relative_to(ROOT)): sha256(path) for path in seal_paths}, indent=2))
    teacher_histories = {}
    teacher_info = {}
    student_histories = {}
    student_info = {}
    targets_archive = {}
    predictions = {}
    metric_rows = []
    for seed in config["training_seeds"]:
        initialize(seed)
        teacher = Teacher().to(device)
        teacher_hash = initial_hash((teacher,))
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        loader = DataLoader(TensorDataset(torch.from_numpy(data["source"]["packets"]),
                                          torch.from_numpy(data["source"]["windows"]),
                                          torch.from_numpy(data["source"]["labels"])),
                            batch_size=config["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        history = []
        best_f1 = -1.0
        phase_started = time.monotonic()
        for epoch in range(1, config["teacher_epochs"] + 1):
            teacher.train()
            total_loss = 0.0
            total_correct = 0
            for packet_batch, window_batch, labels in loader:
                packet_batch = packet_batch.to(device)
                window_batch = window_batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = teacher(packet_batch, window_batch)
                loss = nn.functional.cross_entropy(logits, labels)
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(labels)
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            valid_logits = teacher_outputs(teacher, data["valid"]["packets"],
                                           data["valid"]["windows"], device,
                                           config["batch_size"])[0]
            score = metric(data["valid"]["labels"], valid_logits.argmax(axis=1))
            history.append({"epoch": epoch, "train_loss": total_loss / len(data["source"]["labels"]),
                            "train_accuracy": total_correct / len(data["source"]["labels"]), **score})
            if score["macro_f1"] > best_f1:
                best_f1 = score["macro_f1"]
                best_epoch = epoch
                best_state = {name: tensor.detach().cpu().clone()
                              for name, tensor in teacher.state_dict().items()}
        teacher.load_state_dict(best_state)
        teacher_histories[str(seed)] = history
        teacher_info[str(seed)] = {"best_epoch": best_epoch, "valid_macro_f1": best_f1,
                                   "initial_hash": teacher_hash,
                                   "parameters": sum(parameter.numel() for parameter in teacher.parameters()),
                                   "train_seconds": time.monotonic() - phase_started}
        torch.save({"state_dict": best_state, "seed": seed, "epoch": best_epoch},
                   RUN / "checkpoints" / f"teacher_{seed}.pt")
        _, packet_target, window_target = teacher_outputs(teacher, data["source"]["packets"],
                                                            data["source"]["windows"], device,
                                                            config["batch_size"])
        targets = {}
        for name, raw_target in (("packet", packet_target), ("window", window_target)):
            target_mean = raw_target.mean(axis=0)
            target_std = raw_target.std(axis=0)
            target_std[target_std < config["feature_standard_deviation_floor"]] = 1.0
            normalized = (raw_target - target_mean) / target_std
            assert normalized.shape == (len(data["source"]["labels"]), 128)
            assert np.isfinite(normalized).all()
            targets[name] = normalized.astype(np.float32)
            targets_archive[f"{name}_{seed}"] = targets[name]
            targets_archive[f"{name}_mean_{seed}"] = target_mean
            targets_archive[f"{name}_std_{seed}"] = target_std
        print(f"teacher_{seed}: valid F1 {best_f1:.4f}, epoch {best_epoch}", flush=True)
        student_initial_hash = None
        for condition in CONDITIONS:
            initialize(seed)
            student = Student().to(device)
            projector = nn.Linear(128, 128).to(device)
            current_hash = initial_hash((student, projector))
            if student_initial_hash is None:
                student_initial_hash = current_hash
            assert student_initial_hash == current_hash
            optimizer = torch.optim.AdamW(list(student.parameters()) + list(projector.parameters()),
                                          lr=config["learning_rate"], weight_decay=config["weight_decay"])
            target = (targets["packet"] if condition == "align_packet" else
                      targets["window"] if condition == "align_window" else
                      np.zeros_like(targets["packet"]))
            loader = DataLoader(TensorDataset(torch.from_numpy(data["source"]["packets"]),
                                              torch.from_numpy(data["source"]["labels"]),
                                              torch.from_numpy(target)),
                                batch_size=config["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            history = []
            best_f1 = -1.0
            phase_started = time.monotonic()
            for epoch in range(1, config["student_epochs"] + 1):
                student.train()
                projector.train()
                total_loss = 0.0
                total_ce = 0.0
                total_feature_loss = 0.0
                total_correct = 0
                for packet_batch, labels, target_batch in loader:
                    packet_batch = packet_batch.to(device)
                    labels = labels.to(device)
                    target_batch = target_batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    features = student.features(packet_batch)
                    logits = student.head(features)
                    cross_entropy = nn.functional.cross_entropy(logits, labels)
                    feature_loss = (nn.functional.mse_loss(projector(features), target_batch)
                                    if condition != "ce" else cross_entropy.new_zeros(()))
                    loss = cross_entropy + config["feature_loss_weight"] * feature_loss
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(labels)
                    total_ce += float(cross_entropy.item()) * len(labels)
                    total_feature_loss += float(feature_loss.item()) * len(labels)
                    total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                valid_prediction = student_predictions(student, data["valid"]["packets"],
                                                       device, config["batch_size"])
                score = metric(data["valid"]["labels"], valid_prediction)
                history.append({"epoch": epoch, "train_loss": total_loss / len(data["source"]["labels"]),
                                "train_ce": total_ce / len(data["source"]["labels"]),
                                "train_feature_loss": total_feature_loss / len(data["source"]["labels"]),
                                "train_accuracy": total_correct / len(data["source"]["labels"]), **score})
                if score["macro_f1"] > best_f1:
                    best_f1 = score["macro_f1"]
                    best_epoch = epoch
                    best_student_state = {name: tensor.detach().cpu().clone()
                                          for name, tensor in student.state_dict().items()}
                    best_projector_state = {name: tensor.detach().cpu().clone()
                                            for name, tensor in projector.state_dict().items()}
            key = f"{condition}_{seed}"
            student.load_state_dict(best_student_state)
            student_histories[key] = history
            student_info[key] = {"best_epoch": best_epoch, "valid_macro_f1": best_f1,
                                 "initial_hash": current_hash,
                                 "inference_parameters": sum(parameter.numel() for parameter in student.parameters()),
                                 "training_parameters": sum(parameter.numel() for parameter in student.parameters())
                                                        + sum(parameter.numel() for parameter in projector.parameters()),
                                 "train_seconds": time.monotonic() - phase_started}
            torch.save({"student_state": best_student_state, "projector_state": best_projector_state,
                        "condition": condition, "seed": seed, "epoch": best_epoch},
                       RUN / "checkpoints" / f"{key}.pt")
            for role in ROLES:
                prediction = student_predictions(student, data[role]["packets"],
                                                 device, config["batch_size"])
                predictions[f"{role}_{key}"] = prediction
                metric_rows.append({"role": role, "condition": condition, "seed": seed,
                                    "best_epoch": best_epoch, **metric(data[role]["labels"], prediction)})
            (RUN / "artifacts/history_partial.json").write_text(json.dumps(student_histories, indent=2))
            print(f"{key}: valid F1 {best_f1:.4f}, epoch {best_epoch}", flush=True)
            if time.monotonic() - started > config["time_limit_seconds"]:
                raise TimeoutError("GPU budget exceeded; partial run retained")
    (RUN / "artifacts/teacher_history.json").write_text(json.dumps(teacher_histories, indent=2))
    (RUN / "artifacts/student_history.json").write_text(json.dumps(student_histories, indent=2))
    with (RUN / "artifacts/metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "condition", "seed", "best_epoch", "accuracy", "macro_f1"))
        writer.writeheader()
        writer.writerows(metric_rows)
    np.savez_compressed(RUN / "artifacts/predictions.npz", **predictions)
    np.savez_compressed(RUN / "artifacts/teacher_source_targets.npz", **targets_archive)
    (RUN / "artifacts/manifest.json").write_text(json.dumps({
        "sampling_manifest": str(PRIOR.relative_to(ROOT)), "sampling_sha256": sha256(PRIOR),
        "teacher_info": teacher_info, "student_info": student_info,
        "prediction_sha256": sha256(RUN / "artifacts/predictions.npz"),
        "teacher_target_sha256": sha256(RUN / "artifacts/teacher_source_targets.npz"),
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
