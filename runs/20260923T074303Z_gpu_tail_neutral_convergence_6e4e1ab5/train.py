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
CONDITIONS = ("full", "tail_neutral")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    true_positive = np.diag(confusion)
    denominator = confusion.sum(0) + confusion.sum(1)
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.divide(2 * true_positive, denominator, out=np.zeros(102), where=denominator != 0).mean()),
    }


def extract(rows: np.ndarray, budget: int, widths: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, int]:
    dimension = 2 * sum((budget + width - 1) // width for width in widths)
    full = np.zeros((len(rows), dimension), dtype=np.float32)
    neutral = np.zeros_like(full)
    partial_count = 0
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
                partial_count += int(window.partial)
            offset += 2 * ((budget + width - 1) // width)
    return full, neutral, partial_count


class WindowNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(240, 128), nn.ReLU(), nn.Linear(128, 102))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def main() -> None:
    started = time.monotonic()
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text())
    assert config["status"] == "frozen" and config["device"] == "cuda:0" and config["gpu"] is True
    assert config["conditions"] == list(CONDITIONS)
    assert not (RUN / "artifacts/predictions.npz").exists(), "refuse to overwrite completed experiment"
    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    device = torch.device(config["device"])
    budget = config["input_budget"]
    widths = tuple(config["window_sizes"])
    data_root = Path(json.loads((ROOT / "configs/datasets.json").read_text())["data_root"])
    sampling = json.loads(PRIOR.read_text())["sampling"]
    data = {}
    partial_counts = {}
    for role in ROLES:
        filename = "train.npz" if role == "source" else f"{role}.npz"
        path = data_root / "TemporalDrift" / filename
        with np.load(path, allow_pickle=False) as archive:
            indices = np.asarray(sampling[role]["rows"], dtype=np.int64)
            labels = archive["y"][indices].astype(np.int64)
            selected_rows = archive["X"][indices]
        assert len(indices) == sampling[role]["count"]
        assert len(np.unique(labels)) == 102 and np.isfinite(selected_rows).all()
        full, neutral, count = extract(selected_rows, budget, widths)
        assert full.shape[1] == 240 and np.isfinite(full).all() and np.isfinite(neutral).all()
        data[role] = {"labels": labels, "full": full, "tail_neutral": neutral}
        partial_counts[role] = count
        print(f"extracted {role}: {len(labels)} rows, {count} partial windows", flush=True)
    source = data["source"]["full"]
    mean = source.mean(axis=0)
    standard_deviation = source.std(axis=0)
    standard_deviation[standard_deviation < 1e-6] = 1.0
    np.savez_compressed(RUN / "artifacts/source_statistics.npz", mean=mean, std=standard_deviation)
    for role in ROLES:
        for condition in CONDITIONS:
            data[role][condition] = (data[role][condition] - mean) / standard_deviation
    seal_sources = [RUN / "PLAN.md", config_path, Path(__file__), PRIOR,
                    ROOT / "src/ta_wf_next/traffic_views.py", ROOT / "src/ta_wf_next/burst_tokens.py",
                    RUN / "artifacts/source_statistics.npz"]
    (RUN / "artifacts/pretraining_seal.json").write_text(json.dumps(
        {str(path.relative_to(ROOT)): sha256(path) for path in seal_sources}, indent=2))
    histories = {}
    best_epochs = {}
    predictions = {}
    rows = []
    for seed in config["training_seeds"]:
        for condition in CONDITIONS:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = WindowNet().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            train_x = torch.from_numpy(data["source"][condition])
            train_y = torch.from_numpy(data["source"]["labels"])
            loader = DataLoader(TensorDataset(train_x, train_y), batch_size=config["batch_size"],
                                shuffle=True, generator=torch.Generator().manual_seed(seed))
            valid_x = torch.from_numpy(data["valid"][condition]).to(device)
            valid_y = data["valid"]["labels"]
            history = []
            best_f1 = -1.0
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                total_loss = 0.0
                total_correct = 0
                for features, labels in loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features)
                    loss = nn.functional.cross_entropy(logits, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(labels)
                    total_correct += int((logits.argmax(1) == labels).sum().item())
                model.eval()
                with torch.inference_mode():
                    valid_predictions = model(valid_x).argmax(1).cpu().numpy()
                score = metric(valid_y, valid_predictions)
                history.append({"epoch": epoch, "train_loss": total_loss / len(train_y),
                                "train_accuracy": total_correct / len(train_y), **score})
                if score["macro_f1"] > best_f1:
                    best_f1 = score["macro_f1"]
                    best_epoch = epoch
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            key = f"{condition}_{seed}"
            histories[key] = history
            best_epochs[key] = best_epoch
            model.load_state_dict(best_state)
            torch.save({"state_dict": best_state, "epoch": best_epoch, "condition": condition,
                        "seed": seed, "parameters": sum(parameter.numel() for parameter in model.parameters())},
                       RUN / "checkpoints" / f"{key}.pt")
            with torch.inference_mode():
                for role in ROLES:
                    features = torch.from_numpy(data[role][condition]).to(device)
                    prediction = model(features).argmax(1).cpu().numpy().astype(np.int16)
                    predictions[f"{role}_{key}"] = prediction
                    rows.append({"role": role, "condition": condition, "seed": seed,
                                 "best_epoch": best_epoch, **metric(data[role]["labels"], prediction)})
            print(f"{key}: epoch {best_epoch}, valid macro-F1 {best_f1:.4f}", flush=True)
            if time.monotonic() - started > config["time_limit_seconds"]:
                raise TimeoutError("training budget exceeded; partial run retained")
    (RUN / "artifacts/history.json").write_text(json.dumps(histories, indent=2))
    with (RUN / "artifacts/metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "condition", "seed", "best_epoch", "accuracy", "macro_f1"))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(RUN / "artifacts/predictions.npz", **predictions)
    (RUN / "artifacts/manifest.json").write_text(json.dumps({
        "sampling_manifest": str(PRIOR.relative_to(ROOT)), "sampling_sha256": sha256(PRIOR),
        "partial_counts": partial_counts, "best_epochs": best_epochs,
        "prediction_sha256": sha256(RUN / "artifacts/predictions.npz"),
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
