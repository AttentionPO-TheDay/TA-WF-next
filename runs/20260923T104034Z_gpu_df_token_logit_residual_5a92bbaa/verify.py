from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ta_wf_next.models import DF
from ta_wf_next.traffic_views import generate_views


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
ROLES = ("source", "valid", "day14", "day30", "day90", "day150", "day270")
CONDITIONS = ("df_only", "residual_constant", "residual_window")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResidualHead(nn.Module):
    def __init__(self, multiplier: float) -> None:
        super().__init__()
        self.df = DF(102)
        self.fine = nn.Sequential(nn.Conv1d(2, 32, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(32, 32, 3, padding=1), nn.GELU(),
                                  nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.coarse = nn.Sequential(nn.Conv1d(2, 32, 3, padding=1), nn.GELU(),
                                    nn.Conv1d(32, 32, 3, padding=1), nn.GELU(),
                                    nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.adapter = nn.Sequential(nn.Linear(576, 128), nn.ReLU(), nn.Linear(128, 102))
        self.multiplier = multiplier

    def forward(self, packet: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        feature = self.df.forward_features(packet[:, None, :])
        fine = window[:, :200].reshape(-1, 100, 2).transpose(1, 2)
        coarse = window[:, 200:].reshape(-1, 20, 2).transpose(1, 2)
        token = torch.cat((self.fine(fine), self.coarse(coarse)), dim=1)
        return self.df.mlp(feature) + self.multiplier * self.adapter(torch.cat((feature, token), dim=1))


def metric(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    table = np.bincount(y * 102 + p, minlength=102**2).reshape(102, 102)
    denominator = table.sum(0) + table.sum(1)
    f1 = np.divide(2 * np.diag(table), denominator, out=np.zeros(102), where=denominator != 0)
    return float(np.mean(y == p)), float(f1.mean())


def main() -> None:
    torch.set_num_threads(4)
    config = json.loads((RUN / "config.json").read_text())
    manifest = json.loads((RUN / "artifacts/manifest.json").read_text())
    sealed = json.loads((RUN / "artifacts/input_seal.json").read_text())
    errors = []
    for relative, expected in sealed.items():
        if digest(ROOT / relative) != expected:
            errors.append(f"input seal mismatch: {relative}")
    sampling_path = ROOT / manifest["sampling_manifest"]
    if digest(sampling_path) != manifest["sampling_sha256"]:
        errors.append("sampling hash mismatch")
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
            traces = archive["X"][indices]
        if len(indices) != item["count"] or len(np.unique(indices)) != len(indices) or len(np.unique(labels)) != 102:
            errors.append(f"{role}: frozen sample/label mismatch")
        packets = np.sign(traces[:, :5000]).astype(np.float32)
        full = np.zeros((len(traces), 240), dtype=np.float32)
        neutral = np.zeros_like(full)
        for i, trace in enumerate(traces):
            views = generate_views(trace[:5000], input_kind="signed_timestamp", budget=5000,
                                   window_sizes=(50, 250))
            offset = 0
            for width, windows in views.direction_windows:
                for j, window in enumerate(windows):
                    column = offset + 2 * j
                    full[i, column:column + 2] = window.positive_fraction, window.transition_fraction
                    neutral[i, column:column + 2] = ((0.5, 0.0) if window.partial else
                                                     (window.positive_fraction, window.transition_fraction))
                offset += 2 * ((5000 + width - 1) // width)
        if role == "source":
            full_source = full
        data[role] = (packets, neutral, labels)
        hashes[role] = {hashlib.sha256(row.astype(np.int8).tobytes()).hexdigest() for row in packets}
    overlap = len(hashes["source"] & hashes["valid"])
    if overlap:
        errors.append("source/valid direction overlap")
    with np.load(RUN / "artifacts/window_statistics.npz", allow_pickle=False) as archive:
        mean, std = archive["mean"], archive["std"]
    expected_mean, expected_std = full_source.mean(0), full_source.std(0)
    expected_std[expected_std < 1e-6] = 1.0
    if not np.array_equal(mean, expected_mean) or not np.array_equal(std, expected_std):
        errors.append("source-only normalization mismatch")
    for role in ROLES:
        packet, neutral, labels = data[role]
        data[role] = packet, ((neutral - mean) / std).astype(np.float32), labels
    prediction_path = RUN / "artifacts/predictions.npz"
    if digest(prediction_path) != manifest["prediction_sha256"]:
        errors.append("prediction hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in archive.files}
    with (RUN / "artifacts/metrics.csv").open(newline="") as handle:
        rows = {(row["role"], row["condition"], int(row["seed"])): row
                for row in csv.DictReader(handle)}
    histories = json.loads((RUN / "artifacts/history.json").read_text())
    if len(predictions) != 63 or len(rows) != 63 or len(histories) != 9:
        errors.append("array/metric/history count mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    replay_count = 0
    for seed in config["training_seeds"]:
        df_hashes = set()
        residual_hashes = set()
        for condition in CONDITIONS:
            key = f"{condition}_{seed}"
            history = histories[key]
            selected = max(history, key=lambda row: row["macro_f1"])
            info = manifest["models"][key]
            checkpoint = torch.load(RUN / "checkpoints" / f"{key}.pt", map_location="cpu", weights_only=True)
            if len(history) != 45 or [row["epoch"] for row in history] != list(range(1, 46)):
                errors.append(f"{key}: epoch history mismatch")
            if checkpoint["epoch"] != selected["epoch"] or info["best_epoch"] != selected["epoch"] or abs(info["valid_macro_f1"] - selected["macro_f1"]) > 1e-12:
                errors.append(f"{key}: earliest-best selection mismatch")
            df_hashes.add(info["initial_df_hash"])
            if condition != "df_only":
                residual_hashes.add(info["initial_residual_hash"])
            model = (DF(102) if condition == "df_only" else ResidualHead(config["residual_multiplier"])).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            if sum(p.numel() for p in model.parameters()) != info["parameters"]:
                errors.append(f"{key}: parameter count mismatch")
            for role in ROLES:
                packet, window, labels = data[role]
                if condition == "residual_constant":
                    window = np.zeros_like(window)
                output = []
                with torch.inference_mode():
                    for start in range(0, len(packet), config["batch_size"]):
                        x = torch.from_numpy(packet[start:start + config["batch_size"]]).to(device)
                        if condition == "df_only":
                            logits, _ = model(x[:, None, :])
                        else:
                            w = torch.from_numpy(window[start:start + config["batch_size"]]).to(device)
                            logits = model(x, w)
                        output.append(logits.argmax(1).cpu().numpy())
                replay = np.concatenate(output)
                if not np.array_equal(replay, predictions[f"{role}_{key}"]):
                    errors.append(f"{role}_{key}: checkpoint replay mismatch")
                accuracy, f1 = metric(labels, predictions[f"{role}_{key}"])
                row = rows[(role, condition, seed)]
                if int(row["best_epoch"]) != selected["epoch"] or abs(float(row["accuracy"]) - accuracy) > 1e-12 or abs(float(row["macro_f1"]) - f1) > 1e-12:
                    errors.append(f"{role}_{key}: metric mismatch")
                replay_count += 1
        if len(df_hashes) != 1 or len(residual_hashes) != 1:
            errors.append(f"{seed}: initial weights mismatch")
    result = {"source_valid_direction_overlap": overlap, "checkpoint_predictions_replayed": replay_count,
              "metric_rows": len(rows), "errors": errors}
    (RUN / "artifacts/integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
