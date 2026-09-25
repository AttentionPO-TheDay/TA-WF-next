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
CONDITIONS = ("packet_only", "window_only", "fusion")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    true_positive = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.divide(2 * true_positive, denominator, out=np.zeros(102), where=denominator != 0).mean()),
    }


def extract(rows: np.ndarray, budget: int, widths: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = 2 * sum((budget + width - 1) // width for width in widths)
    packets = np.sign(rows[:, :budget]).astype(np.float32)
    windows_full = np.zeros((len(rows), dimension), dtype=np.float32)
    windows_neutral = np.zeros_like(windows_full)
    for row_number, row in enumerate(rows):
        views = generate_views(row[:budget], input_kind="signed_timestamp", budget=budget, window_sizes=widths)
        offset = 0
        for width, windows in views.direction_windows:
            for window_number, window in enumerate(windows):
                column = offset + 2 * window_number
                windows_full[row_number, column:column + 2] = (
                    window.positive_fraction, window.transition_fraction
                )
                windows_neutral[row_number, column:column + 2] = (
                    (0.5, 0.0) if window.partial else (window.positive_fraction, window.transition_fraction)
                )
            offset += 2 * ((budget + width - 1) // width)
    return packets, windows_full, windows_neutral


class MainModel(nn.Module):
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
        packet_features = self.packet_body(packets.unsqueeze(1))
        window_features = self.window_body(windows)
        return self.head(torch.cat((packet_features, window_features), dim=1))


def observed_inputs(role: dict[str, np.ndarray], condition: str) -> tuple[np.ndarray, np.ndarray]:
    packets = role["packets"] if condition != "window_only" else np.zeros_like(role["packets"])
    windows = role["windows"] if condition != "packet_only" else np.zeros_like(role["windows"])
    return packets, windows


def predict(model: MainModel, packets: np.ndarray, windows: np.ndarray,
            device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(packets), batch_size):
            packet_batch = torch.from_numpy(packets[start:start + batch_size]).to(device)
            window_batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
            batches.append(model(packet_batch, window_batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(batches).astype(np.int16)


def main() -> None:
    started = time.monotonic()
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text())
    assert config["status"] == "frozen" and config["gpu"] is True
    assert config["conditions"] == list(CONDITIONS)
    assert config["device"] == "cuda:0" and torch.cuda.is_available()
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
    source_full_windows = None
    for role in ROLES:
        item = sampling[role]
        with np.load(data_root / item["path"], allow_pickle=False) as archive:
            indices = np.asarray(item["rows"], dtype=np.int64)
            labels = archive["y"][indices].astype(np.int64)
            rows = archive["X"][indices]
        assert len(rows) == item["count"] and len(np.unique(labels)) == 102
        assert np.isfinite(rows).all()
        packets, windows_full, windows_neutral = extract(rows, budget, widths)
        assert packets.shape[1] == budget and windows_neutral.shape[1] == 240
        direction_hashes[role] = {
            hashlib.sha256(packet.astype(np.int8).tobytes()).hexdigest() for packet in packets
        }
        data[role] = {"labels": labels, "packets": packets, "windows": windows_neutral}
        if role == "source":
            source_full_windows = windows_full
        print(f"extracted {role}: {len(rows)}", flush=True)
    assert not direction_hashes["source"] & direction_hashes["valid"]
    mean = source_full_windows.mean(axis=0)
    standard_deviation = source_full_windows.std(axis=0)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    np.savez_compressed(RUN / "artifacts/window_statistics.npz", mean=mean, std=standard_deviation)
    for role in ROLES:
        data[role]["windows"] = (data[role]["windows"] - mean) / standard_deviation
        assert np.isfinite(data[role]["windows"]).all()
    seal_paths = (RUN / "PLAN.md", config_path, Path(__file__), PRIOR,
                  ROOT / "configs/datasets.json", ROOT / "src/ta_wf_next/traffic_views.py",
                  ROOT / "src/ta_wf_next/burst_tokens.py", RUN / "artifacts/window_statistics.npz")
    (RUN / "artifacts/pretraining_seal.json").write_text(json.dumps(
        {str(path.relative_to(ROOT)): sha256(path) for path in seal_paths}, indent=2))
    histories = {}
    best_epochs = {}
    initial_hashes = {}
    predictions = {}
    metric_rows = []
    model_info = {}
    for seed in config["training_seeds"]:
        for condition in CONDITIONS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = MainModel().to(device)
            initial_hash = hashlib.sha256(b"".join(
                parameter.detach().cpu().numpy().tobytes() for parameter in model.parameters())).hexdigest()
            initial_hashes.setdefault(str(seed), initial_hash)
            assert initial_hashes[str(seed)] == initial_hash
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            train_packets, train_windows = observed_inputs(data["source"], condition)
            train_labels = data["source"]["labels"]
            loader = DataLoader(TensorDataset(torch.from_numpy(train_packets),
                                              torch.from_numpy(train_windows),
                                              torch.from_numpy(train_labels)),
                                batch_size=config["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            valid_packets, valid_windows = observed_inputs(data["valid"], condition)
            history = []
            best_f1 = -1.0
            condition_started = time.monotonic()
            for epoch in range(1, config["epochs"] + 1):
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
                valid_prediction = predict(model, valid_packets, valid_windows,
                                           device, config["batch_size"])
                score = metric(data["valid"]["labels"], valid_prediction)
                history.append({"epoch": epoch, "train_loss": total_loss / len(train_labels),
                                "train_accuracy": total_correct / len(train_labels), **score})
                if score["macro_f1"] > best_f1:
                    best_f1 = score["macro_f1"]
                    best_epoch = epoch
                    best_state = {name: tensor.detach().cpu().clone()
                                  for name, tensor in model.state_dict().items()}
            key = f"{condition}_{seed}"
            histories[key] = history
            best_epochs[key] = best_epoch
            model.load_state_dict(best_state)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            model_info[key] = {"parameters": parameter_count,
                               "train_seconds": time.monotonic() - condition_started}
            torch.save({"state_dict": best_state, "epoch": best_epoch, "condition": condition,
                        "seed": seed, "parameters": parameter_count}, RUN / "checkpoints" / f"{key}.pt")
            for role in ROLES:
                packets, windows = observed_inputs(data[role], condition)
                prediction = predict(model, packets, windows, device, config["batch_size"])
                predictions[f"{role}_{key}"] = prediction
                metric_rows.append({"role": role, "condition": condition, "seed": seed,
                                    "best_epoch": best_epoch,
                                    **metric(data[role]["labels"], prediction)})
            (RUN / "artifacts/history_partial.json").write_text(json.dumps(histories, indent=2))
            print(f"{key}: best epoch {best_epoch}, valid F1 {best_f1:.4f}, "
                  f"{model_info[key]['train_seconds']:.1f}s", flush=True)
            if time.monotonic() - started > config["time_limit_seconds"]:
                raise TimeoutError("GPU budget exceeded; partial run retained")
    assert len({item["parameters"] for item in model_info.values()}) == 1
    (RUN / "artifacts/history.json").write_text(json.dumps(histories, indent=2))
    with (RUN / "artifacts/metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "condition", "seed", "best_epoch", "accuracy", "macro_f1"))
        writer.writeheader()
        writer.writerows(metric_rows)
    np.savez_compressed(RUN / "artifacts/predictions.npz", **predictions)
    (RUN / "artifacts/manifest.json").write_text(json.dumps({
        "sampling_manifest": str(PRIOR.relative_to(ROOT)), "sampling_sha256": sha256(PRIOR),
        "best_epochs": best_epochs, "initial_hashes": initial_hashes, "models": model_info,
        "prediction_sha256": sha256(RUN / "artifacts/predictions.npz"),
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
