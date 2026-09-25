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
TEACHERS = ("packet_teacher", "fusion_teacher")
STUDENTS = ("ce", "kd_packet", "kd_fusion")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.divide(2 * np.diag(confusion), denominator,
                                    out=np.zeros(102), where=denominator != 0).mean()),
    }


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

    def forward(self, packets: torch.Tensor, windows: torch.Tensor) -> torch.Tensor:
        packet_features = self.packet_body(packets[:, None, :])
        window_features = self.window_body(windows)
        return self.head(torch.cat((packet_features, window_features), dim=1))


class Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packet_body = packet_body()
        self.head = nn.Linear(128, 102)

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        return self.head(self.packet_body(packets[:, None, :]))


def infer_teacher(model: Teacher, packets: np.ndarray, windows: np.ndarray,
                  device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(packets), batch_size):
            packet_batch = torch.from_numpy(packets[start:start + batch_size]).to(device)
            window_batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
            outputs.append(model(packet_batch, window_batch).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def infer_student(model: Student, packets: np.ndarray,
                  device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(packets), batch_size):
            packet_batch = torch.from_numpy(packets[start:start + batch_size]).to(device)
            outputs.append(model(packet_batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.int16)


def initialize(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_hash(model: nn.Module) -> str:
    return hashlib.sha256(b"".join(parameter.detach().cpu().numpy().tobytes()
                                    for parameter in model.parameters())).hexdigest()


def main() -> None:
    started = time.monotonic()
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text())
    assert config["status"] == "frozen" and config["gpu"] is True
    assert config["device"] == "cuda:0" and torch.cuda.is_available()
    assert config["teacher_conditions"] == list(TEACHERS)
    assert config["student_conditions"] == list(STUDENTS)
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
    window_full_source = None
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
                window_full_source = full
        print(f"extracted {role}: {len(raw)}", flush=True)
    assert not direction_hashes["source"] & direction_hashes["valid"]
    mean = window_full_source.mean(axis=0)
    standard_deviation = window_full_source.std(axis=0)
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
    teacher_logits = {}
    student_histories = {}
    student_info = {}
    student_predictions = {}
    metric_rows = []
    for seed in config["training_seeds"]:
        initial_teacher_hash = None
        for condition in TEACHERS:
            initialize(seed)
            model = Teacher().to(device)
            current_hash = model_hash(model)
            if initial_teacher_hash is None:
                initial_teacher_hash = current_hash
            assert initial_teacher_hash == current_hash
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])
            source_windows = (data["source"]["windows"] if condition == "fusion_teacher"
                              else np.zeros_like(data["source"]["windows"]))
            valid_windows = (data["valid"]["windows"] if condition == "fusion_teacher"
                             else np.zeros_like(data["valid"]["windows"]))
            loader = DataLoader(TensorDataset(torch.from_numpy(data["source"]["packets"]),
                                              torch.from_numpy(source_windows),
                                              torch.from_numpy(data["source"]["labels"])),
                                batch_size=config["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            history = []
            best_f1 = -1.0
            phase_started = time.monotonic()
            for epoch in range(1, config["teacher_epochs"] + 1):
                model.train()
                total_loss = 0.0
                total_correct = 0
                for packet_batch, window_batch, labels in loader:
                    packet_batch = packet_batch.to(device)
                    window_batch = window_batch.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(packet_batch, window_batch)
                    loss = nn.functional.cross_entropy(logits, labels)
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(labels)
                    total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                valid_logits = infer_teacher(model, data["valid"]["packets"], valid_windows,
                                             device, config["batch_size"])
                score = metric(data["valid"]["labels"], valid_logits.argmax(axis=1))
                history.append({"epoch": epoch, "train_loss": total_loss / len(data["source"]["labels"]),
                                "train_accuracy": total_correct / len(data["source"]["labels"]), **score})
                if score["macro_f1"] > best_f1:
                    best_f1 = score["macro_f1"]
                    best_epoch = epoch
                    best_state = {name: tensor.detach().cpu().clone()
                                  for name, tensor in model.state_dict().items()}
            key = f"{condition}_{seed}"
            model.load_state_dict(best_state)
            teacher_logits[key] = infer_teacher(model, data["source"]["packets"], source_windows,
                                                device, config["batch_size"])
            teacher_histories[key] = history
            teacher_info[key] = {"best_epoch": best_epoch, "valid_macro_f1": best_f1,
                                 "parameters": sum(parameter.numel() for parameter in model.parameters()),
                                 "initial_hash": current_hash, "train_seconds": time.monotonic() - phase_started}
            torch.save({"state_dict": best_state, "condition": condition, "seed": seed,
                        "epoch": best_epoch}, RUN / "checkpoints" / f"{key}.pt")
            print(f"{key}: valid F1 {best_f1:.4f}, epoch {best_epoch}", flush=True)
        initial_student_hash = None
        for condition in STUDENTS:
            initialize(seed)
            model = Student().to(device)
            current_hash = model_hash(model)
            if initial_student_hash is None:
                initial_student_hash = current_hash
            assert initial_student_hash == current_hash
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])
            if condition == "kd_packet":
                targets = teacher_logits[f"packet_teacher_{seed}"]
            elif condition == "kd_fusion":
                targets = teacher_logits[f"fusion_teacher_{seed}"]
            else:
                targets = np.zeros((len(data["source"]["labels"]), 102), dtype=np.float32)
            loader = DataLoader(TensorDataset(torch.from_numpy(data["source"]["packets"]),
                                              torch.from_numpy(data["source"]["labels"]),
                                              torch.from_numpy(targets)),
                                batch_size=config["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            history = []
            best_f1 = -1.0
            phase_started = time.monotonic()
            for epoch in range(1, config["student_epochs"] + 1):
                model.train()
                total_loss = 0.0
                total_correct = 0
                for packet_batch, labels, target_logits in loader:
                    packet_batch = packet_batch.to(device)
                    labels = labels.to(device)
                    target_logits = target_logits.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(packet_batch)
                    cross_entropy = nn.functional.cross_entropy(logits, labels)
                    if condition == "ce":
                        loss = cross_entropy
                    else:
                        temperature = config["distill_temperature"]
                        weight = config["distill_weight"]
                        soft_targets = nn.functional.softmax(target_logits / temperature, dim=1)
                        soft_outputs = nn.functional.log_softmax(logits / temperature, dim=1)
                        distillation = nn.functional.kl_div(soft_outputs, soft_targets,
                                                            reduction="batchmean") * temperature**2
                        loss = (1 - weight) * cross_entropy + weight * distillation
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(labels)
                    total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                valid_prediction = infer_student(model, data["valid"]["packets"],
                                                 device, config["batch_size"])
                score = metric(data["valid"]["labels"], valid_prediction)
                history.append({"epoch": epoch, "train_loss": total_loss / len(data["source"]["labels"]),
                                "train_accuracy": total_correct / len(data["source"]["labels"]), **score})
                if score["macro_f1"] > best_f1:
                    best_f1 = score["macro_f1"]
                    best_epoch = epoch
                    best_state = {name: tensor.detach().cpu().clone()
                                  for name, tensor in model.state_dict().items()}
            key = f"{condition}_{seed}"
            model.load_state_dict(best_state)
            student_histories[key] = history
            student_info[key] = {"best_epoch": best_epoch, "valid_macro_f1": best_f1,
                                 "parameters": sum(parameter.numel() for parameter in model.parameters()),
                                 "initial_hash": current_hash, "train_seconds": time.monotonic() - phase_started}
            torch.save({"state_dict": best_state, "condition": condition, "seed": seed,
                        "epoch": best_epoch}, RUN / "checkpoints" / f"{key}.pt")
            for role in ROLES:
                prediction = infer_student(model, data[role]["packets"], device, config["batch_size"])
                student_predictions[f"{role}_{key}"] = prediction
                metric_rows.append({"role": role, "condition": condition, "seed": seed,
                                    "best_epoch": best_epoch, **metric(data[role]["labels"], prediction)})
            (RUN / "artifacts/history_partial.json").write_text(json.dumps(student_histories, indent=2))
            print(f"{key}: valid F1 {best_f1:.4f}, epoch {best_epoch}", flush=True)
            if time.monotonic() - started > config["time_limit_seconds"]:
                raise TimeoutError("GPU budget exceeded; partial run retained")
    assert len({item["parameters"] for item in student_info.values()}) == 1
    (RUN / "artifacts/teacher_history.json").write_text(json.dumps(teacher_histories, indent=2))
    (RUN / "artifacts/student_history.json").write_text(json.dumps(student_histories, indent=2))
    with (RUN / "artifacts/metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "condition", "seed", "best_epoch", "accuracy", "macro_f1"))
        writer.writeheader()
        writer.writerows(metric_rows)
    np.savez_compressed(RUN / "artifacts/predictions.npz", **student_predictions)
    np.savez_compressed(RUN / "artifacts/teacher_source_logits.npz", **teacher_logits)
    (RUN / "artifacts/manifest.json").write_text(json.dumps({
        "sampling_manifest": str(PRIOR.relative_to(ROOT)), "sampling_sha256": sha256(PRIOR),
        "teacher_info": teacher_info, "student_info": student_info,
        "prediction_sha256": sha256(RUN / "artifacts/predictions.npz"),
        "teacher_logit_sha256": sha256(RUN / "artifacts/teacher_source_logits.npz"),
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
