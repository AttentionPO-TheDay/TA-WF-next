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

from ta_wf_next.models import DF
from ta_wf_next.traffic_views import generate_views


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
ROLES = ("source", "valid", "day14", "day30", "day90", "day150", "day270")
CONDITIONS = ("df_only", "df_constant", "df_window")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * np.diag(confusion), denominator, out=np.zeros(102), where=denominator != 0)
    return {"accuracy": float(np.mean(labels == predictions)), "macro_f1": float(f1.mean())}


def extract(rows: np.ndarray, budget: int, widths: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = 2 * sum((budget + width - 1) // width for width in widths)
    packets = np.sign(rows[:, :budget]).astype(np.float32)
    full = np.zeros((len(rows), dimension), dtype=np.float32)
    neutral = np.zeros_like(full)
    for row_number, row in enumerate(rows):
        views = generate_views(row[:budget], input_kind="signed_timestamp", budget=budget, window_sizes=widths)
        offset = 0
        for width, windows in views.direction_windows:
            for window_number, window in enumerate(windows):
                column = offset + 2 * window_number
                pair = (window.positive_fraction, window.transition_fraction)
                full[row_number, column:column + 2] = pair
                neutral[row_number, column:column + 2] = (0.5, 0.0) if window.partial else pair
            offset += 2 * ((budget + width - 1) // width)
    return packets, full, neutral


class DFFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.df = DF(102)
        del self.df.mlp
        self.window_body = nn.Sequential(nn.Linear(240, 128), nn.ReLU(),
                                         nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Linear(640, 102)

    def forward(self, packet: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        features = self.df.forward_features(packet[:, None, :])
        token_features = self.window_body(window)
        return self.head(torch.cat((features, token_features), dim=1))


def initialize(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predictions(model: nn.Module, condition: str, packet: np.ndarray, window: np.ndarray,
                device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(packet), batch_size):
            x = torch.from_numpy(packet[start:start + batch_size]).to(device)
            if condition == "df_only":
                logits, _ = model(x[:, None, :])
            else:
                w = torch.from_numpy(window[start:start + batch_size]).to(device)
                logits = model(x, w)
            output.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(output).astype(np.int16)


def main() -> None:
    started = time.monotonic()
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text())
    assert config["status"] == "frozen" and config["gpu"] is True
    assert config["device"] == "cuda:0" and torch.cuda.is_available()
    assert config["conditions"] == list(CONDITIONS)
    assert not (RUN / "artifacts/predictions.npz").exists()
    assert not list((RUN / "checkpoints").glob("*.pt"))
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    device = torch.device(config["device"])
    sampling_path = ROOT / "runs" / config["sampling_run"] / "artifacts/manifest.json"
    sampling = json.loads(sampling_path.read_text())["sampling"]
    data_root = Path(json.loads((ROOT / "configs/datasets.json").read_text())["data_root"])
    widths = tuple(config["window_sizes"])
    data = {}
    hashes = {}
    source_full = None
    for role in ROLES:
        item = sampling[role]
        with np.load(data_root / item["path"], allow_pickle=False) as archive:
            indices = np.asarray(item["rows"], dtype=np.int64)
            labels = archive["y"][indices].astype(np.int64)
            rows = archive["X"][indices]
        assert len(rows) == item["count"] and len(np.unique(labels)) == 102
        assert np.isfinite(rows).all()
        packet, full, neutral = extract(rows, config["input_budget"], widths)
        assert packet.shape == (len(rows), 5000) and neutral.shape == (len(rows), 240)
        data[role] = {"packet": packet, "window": neutral, "labels": labels}
        hashes[role] = {hashlib.sha256(row.astype(np.int8).tobytes()).hexdigest() for row in packet}
        if role == "source":
            source_full = full
        print(f"loaded {role}: {len(rows)}", flush=True)
    assert not hashes["source"] & hashes["valid"]
    mean = source_full.mean(0)
    std = source_full.std(0)
    std[std < 1e-6] = 1.0
    np.savez_compressed(RUN / "artifacts/window_statistics.npz", mean=mean, std=std)
    for role in ROLES:
        data[role]["window"] = ((data[role]["window"] - mean) / std).astype(np.float32)
        assert np.isfinite(data[role]["window"]).all()
    sealed = (RUN / "PLAN.md", config_path, Path(__file__), sampling_path,
              ROOT / "configs/datasets.json", ROOT / "src/ta_wf_next/models/df.py",
              ROOT / "src/ta_wf_next/traffic_views.py", ROOT / "src/ta_wf_next/burst_tokens.py",
              RUN / "artifacts/window_statistics.npz")
    (RUN / "artifacts/input_seal.json").write_text(json.dumps(
        {str(path.relative_to(ROOT)): sha256(path) for path in sealed}, indent=2))

    histories, predictions_out, info = {}, {}, {}
    metric_rows = []
    for seed in config["training_seeds"]:
        backbone_hash = None
        fusion_hash = None
        for condition in CONDITIONS:
            initialize(seed)
            model = (DF(102) if condition == "df_only" else DFFusion()).to(device)
            backbone = model if condition == "df_only" else model.df
            current_backbone_hash = hashlib.sha256(b"".join(
                tensor.detach().cpu().numpy().tobytes() for tensor in backbone.feature_extraction.parameters()
            ) + b"".join(tensor.detach().cpu().numpy().tobytes()
                       for tensor in backbone.classifier.parameters())).hexdigest()
            if backbone_hash is None:
                backbone_hash = current_backbone_hash
            assert current_backbone_hash == backbone_hash
            if condition != "df_only":
                current_fusion_hash = hashlib.sha256(b"".join(
                    tensor.detach().cpu().numpy().tobytes() for tensor in model.parameters())).hexdigest()
                if fusion_hash is None:
                    fusion_hash = current_fusion_hash
                assert current_fusion_hash == fusion_hash
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])
            source = data["source"]
            window = (np.zeros_like(source["window"]) if condition == "df_constant"
                      else source["window"])
            loader = DataLoader(TensorDataset(torch.from_numpy(source["packet"]),
                                              torch.from_numpy(window), torch.from_numpy(source["labels"])),
                                batch_size=config["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            history = []
            best_f1 = -1.0
            phase_started = time.monotonic()
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                total_loss = 0.0
                total_correct = 0
                for packet_batch, window_batch, labels in loader:
                    packet_batch, window_batch, labels = (x.to(device) for x in (packet_batch, window_batch, labels))
                    optimizer.zero_grad(set_to_none=True)
                    if condition == "df_only":
                        logits, _ = model(packet_batch[:, None, :])
                    else:
                        logits = model(packet_batch, window_batch)
                    loss = nn.functional.cross_entropy(logits, labels)
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(labels)
                    total_correct += int((logits.argmax(1) == labels).sum().item())
                valid = data["valid"]
                valid_window = np.zeros_like(valid["window"]) if condition == "df_constant" else valid["window"]
                valid_predictions = predictions(model, condition, valid["packet"], valid_window,
                                                device, config["batch_size"])
                valid_score = metric(valid["labels"], valid_predictions)
                history.append({"epoch": epoch, "train_loss": total_loss / len(source["labels"]),
                                "train_accuracy": total_correct / len(source["labels"]), **valid_score})
                if valid_score["macro_f1"] > best_f1:
                    best_f1 = valid_score["macro_f1"]
                    best_epoch = epoch
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                if time.monotonic() - started > config["time_limit_seconds"]:
                    raise TimeoutError("GPU budget exceeded; partial run retained")
            key = f"{condition}_{seed}"
            histories[key] = history
            model.load_state_dict(best_state)
            info[key] = {"best_epoch": best_epoch, "valid_macro_f1": best_f1,
                         "parameters": sum(parameter.numel() for parameter in model.parameters()),
                         "backbone_initial_hash": backbone_hash, "fusion_initial_hash": fusion_hash,
                         "train_seconds": time.monotonic() - phase_started}
            torch.save({"state_dict": best_state, "epoch": best_epoch, "condition": condition, "seed": seed},
                       RUN / "checkpoints" / f"{key}.pt")
            for role in ROLES:
                item = data[role]
                use_window = np.zeros_like(item["window"]) if condition == "df_constant" else item["window"]
                predicted = predictions(model, condition, item["packet"], use_window,
                                        device, config["batch_size"])
                predictions_out[f"{role}_{key}"] = predicted
                metric_rows.append({"role": role, "condition": condition, "seed": seed,
                                    "best_epoch": best_epoch, **metric(item["labels"], predicted)})
            (RUN / "artifacts/history_partial.json").write_text(json.dumps(histories, indent=2))
            print(f"{key}: valid F1 {best_f1:.4f}, epoch {best_epoch}, "
                  f"{info[key]['train_seconds']:.1f}s", flush=True)
    (RUN / "artifacts/history.json").write_text(json.dumps(histories, indent=2))
    with (RUN / "artifacts/metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "condition", "seed", "best_epoch", "accuracy", "macro_f1"))
        writer.writeheader()
        writer.writerows(metric_rows)
    np.savez_compressed(RUN / "artifacts/predictions.npz", **predictions_out)
    (RUN / "artifacts/manifest.json").write_text(json.dumps({
        "sampling_manifest": str(sampling_path.relative_to(ROOT)), "sampling_sha256": sha256(sampling_path),
        "prediction_sha256": sha256(RUN / "artifacts/predictions.npz"), "models": info,
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
