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


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
ROLES = ("source", "valid", "day14", "day30", "day90", "day150", "day270")
CONDITIONS = ("ce", "relation_packet", "relation_window")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    denominator = confusion.sum(0) + confusion.sum(1)
    return {"accuracy": float(np.mean(labels == predictions)),
            "macro_f1": float(np.divide(2 * np.diag(confusion), denominator,
                                        out=np.zeros(102), where=denominator != 0).mean())}


def unit_length(features: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(lengths, 1e-8)


def relations(features: np.ndarray, labels: np.ndarray,
              scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    class_sums = np.zeros((102, 128), dtype=np.float32)
    class_counts = np.bincount(labels, minlength=102)
    assert np.all(class_counts == 20)
    np.add.at(class_sums, labels, features)
    prototypes = unit_length(class_sums / class_counts[:, None])
    leave_one_out = unit_length((class_sums[labels] - features) / (class_counts[labels, None] - 1))
    normalized_features = unit_length(features)
    similarities = normalized_features @ prototypes.T
    similarities[np.arange(len(labels)), labels] = np.sum(normalized_features * leave_one_out, axis=1)
    logits = torch.from_numpy(similarities * scale)
    probabilities = torch.softmax(logits, dim=1).numpy().astype(np.float32)
    assert np.isfinite(probabilities).all()
    true_top1 = float(np.mean(probabilities.argmax(axis=1) == labels))
    entropy = float(np.mean(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)))
    info = {"source_true_class_top1": true_top1, "source_mean_entropy": entropy,
            "min_class_count": int(class_counts.min()), "max_class_count": int(class_counts.max())}
    return prototypes.astype(np.float32), leave_one_out.astype(np.float32), probabilities, info


def packet_body() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(1, 32, 9, stride=4), nn.ReLU(),
        nn.Conv1d(32, 64, 7, stride=4), nn.ReLU(),
        nn.AdaptiveAvgPool1d(16), nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(),
    )


class Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.packet_body = packet_body()
        self.head = nn.Linear(128, 102)

    def features(self, packets: torch.Tensor) -> torch.Tensor:
        return self.packet_body(packets[:, None, :])

    def forward(self, packets: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(packets))


def predict(model: Student, packets: np.ndarray,
            device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(packets), batch_size):
            batch = torch.from_numpy(packets[start:start + batch_size]).to(device)
            outputs.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.int16)


def initialize(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initial_hash(modules: tuple[nn.Module, ...]) -> str:
    return hashlib.sha256(b"".join(parameter.detach().cpu().numpy().tobytes()
                                    for module in modules for parameter in module.parameters())).hexdigest()


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
    feature_run = ROOT / "runs" / config["teacher_feature_run"]
    feature_path = feature_run / "artifacts/features.npz"
    feature_integrity = json.loads((feature_run / "artifacts/integrity.json").read_text())
    assert feature_integrity["errors"] == []
    output_seal = json.loads((feature_run / "artifacts/output_seal.json").read_text())
    assert output_seal[str(feature_path.relative_to(ROOT))] == sha256(feature_path)
    input_seal = json.loads((feature_run / "artifacts/input_seal.json").read_text())
    assert input_seal[str(sampling_path.relative_to(ROOT))] == sha256(sampling_path)
    with np.load(feature_path, allow_pickle=False) as archive:
        teacher_features = {f"{branch}_{seed}": archive[f"teacher_{branch}_source_{seed}"].astype(np.float32)
                            for seed in config["training_seeds"] for branch in ("packet", "window")}
    data_root = Path(json.loads((ROOT / "configs/datasets.json").read_text())["data_root"])
    data = {}
    direction_hashes = {}
    for role in ROLES:
        item = sampling[role]
        with np.load(data_root / item["path"], allow_pickle=False) as archive:
            indices = np.asarray(item["rows"], dtype=np.int64)
            labels = archive["y"][indices].astype(np.int64)
            packets = np.sign(archive["X"][indices, :config["input_budget"]]).astype(np.float32)
        assert len(labels) == item["count"] and len(np.unique(labels)) == 102
        assert np.isfinite(packets).all()
        data[role] = {"packets": packets, "labels": labels}
        direction_hashes[role] = {hashlib.sha256(packet.astype(np.int8).tobytes()).hexdigest()
                                  for packet in packets}
        print(f"loaded {role}: {len(labels)}", flush=True)
    assert not direction_hashes["source"] & direction_hashes["valid"]
    assert all(value.shape == (len(data["source"]["labels"]), 128)
               and np.isfinite(value).all() for value in teacher_features.values())
    seal_paths = (RUN / "PLAN.md", config_path, Path(__file__), sampling_path,
                  ROOT / "configs/datasets.json", feature_path,
                  feature_run / "artifacts/input_seal.json",
                  feature_run / "artifacts/output_seal.json",
                  feature_run / "artifacts/integrity.json")
    (RUN / "artifacts/input_seal.json").write_text(json.dumps(
        {str(path.relative_to(ROOT)): sha256(path) for path in seal_paths}, indent=2))
    relation_arrays = {}
    relation_info = {}
    histories = {}
    model_info = {}
    predictions = {}
    metric_rows = []
    for seed in config["training_seeds"]:
        relation_data = {}
        for branch in ("packet", "window"):
            prototypes, leave_one_out, probabilities, info = relations(
                teacher_features[f"{branch}_{seed}"], data["source"]["labels"],
                config["relation_scale"])
            relation_data[branch] = (prototypes, leave_one_out, probabilities)
            relation_info[f"{branch}_{seed}"] = info
            relation_arrays[f"{branch}_prototypes_{seed}"] = prototypes
            relation_arrays[f"{branch}_leave_one_out_{seed}"] = leave_one_out
            relation_arrays[f"{branch}_probabilities_{seed}"] = probabilities
        reference_hash = None
        for condition in CONDITIONS:
            initialize(seed)
            student = Student().to(device)
            projector = nn.Linear(128, 128).to(device)
            current_hash = initial_hash((student, projector))
            if reference_hash is None:
                reference_hash = current_hash
            assert reference_hash == current_hash
            optimizer = torch.optim.AdamW(list(student.parameters()) + list(projector.parameters()),
                                          lr=config["learning_rate"], weight_decay=config["weight_decay"])
            indices = np.arange(len(data["source"]["labels"]), dtype=np.int64)
            loader = DataLoader(TensorDataset(torch.from_numpy(data["source"]["packets"]),
                                              torch.from_numpy(data["source"]["labels"]),
                                              torch.from_numpy(indices)),
                                batch_size=config["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            if condition == "ce":
                branch = None
            else:
                branch = condition.removeprefix("relation_")
                prototypes, leave_one_out, probabilities = relation_data[branch]
                prototypes_tensor = torch.from_numpy(prototypes).to(device)
                leave_one_out_tensor = torch.from_numpy(leave_one_out).to(device)
                probabilities_tensor = torch.from_numpy(probabilities).to(device)
            history = []
            best_f1 = -1.0
            phase_started = time.monotonic()
            for epoch in range(1, config["student_epochs"] + 1):
                student.train()
                projector.train()
                total_loss = 0.0
                total_relation_loss = 0.0
                total_correct = 0
                for packet_batch, labels, sample_indices in loader:
                    packet_batch = packet_batch.to(device)
                    labels = labels.to(device)
                    sample_indices = sample_indices.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    features = student.features(packet_batch)
                    logits = student.head(features)
                    cross_entropy = nn.functional.cross_entropy(logits, labels)
                    if branch is None:
                        relation_loss = cross_entropy.new_zeros(())
                    else:
                        projected = nn.functional.normalize(projector(features), dim=1)
                        similarities = projected @ prototypes_tensor.T
                        own_similarity = (projected * leave_one_out_tensor[sample_indices]).sum(dim=1)
                        similarities = similarities.scatter(1, labels[:, None], own_similarity[:, None])
                        relation_logits = similarities * config["relation_scale"]
                        relation_loss = nn.functional.kl_div(
                            nn.functional.log_softmax(relation_logits, dim=1),
                            probabilities_tensor[sample_indices], reduction="batchmean")
                    loss = cross_entropy + config["relation_loss_weight"] * relation_loss
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(labels)
                    total_relation_loss += float(relation_loss.item()) * len(labels)
                    total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                valid_prediction = predict(student, data["valid"]["packets"], device,
                                           config["batch_size"])
                score = metric(data["valid"]["labels"], valid_prediction)
                history.append({"epoch": epoch, "train_loss": total_loss / len(data["source"]["labels"]),
                                "train_relation_loss": total_relation_loss / len(data["source"]["labels"]),
                                "train_accuracy": total_correct / len(data["source"]["labels"]), **score})
                if score["macro_f1"] > best_f1:
                    best_f1 = score["macro_f1"]
                    best_epoch = epoch
                    best_student = {name: tensor.detach().cpu().clone()
                                    for name, tensor in student.state_dict().items()}
                    best_projector = {name: tensor.detach().cpu().clone()
                                      for name, tensor in projector.state_dict().items()}
            key = f"{condition}_{seed}"
            student.load_state_dict(best_student)
            histories[key] = history
            model_info[key] = {"best_epoch": best_epoch, "valid_macro_f1": best_f1,
                               "initial_hash": current_hash,
                               "inference_parameters": sum(parameter.numel() for parameter in student.parameters()),
                               "training_parameters": sum(parameter.numel() for parameter in student.parameters())
                                                      + sum(parameter.numel() for parameter in projector.parameters()),
                               "train_seconds": time.monotonic() - phase_started}
            torch.save({"student_state": best_student, "projector_state": best_projector,
                        "condition": condition, "seed": seed, "epoch": best_epoch},
                       RUN / "checkpoints" / f"{key}.pt")
            for role in ROLES:
                role_prediction = predict(student, data[role]["packets"], device,
                                          config["batch_size"])
                predictions[f"{role}_{key}"] = role_prediction
                metric_rows.append({"role": role, "condition": condition, "seed": seed,
                                    "best_epoch": best_epoch,
                                    **metric(data[role]["labels"], role_prediction)})
            (RUN / "artifacts/history_partial.json").write_text(json.dumps(histories, indent=2))
            print(f"{key}: valid F1 {best_f1:.4f}, epoch {best_epoch}", flush=True)
            if time.monotonic() - started > config["time_limit_seconds"]:
                raise TimeoutError("GPU budget exceeded; partial run retained")
    (RUN / "artifacts/history.json").write_text(json.dumps(histories, indent=2))
    with (RUN / "artifacts/metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "condition", "seed", "best_epoch", "accuracy", "macro_f1"))
        writer.writeheader()
        writer.writerows(metric_rows)
    np.savez_compressed(RUN / "artifacts/predictions.npz", **predictions)
    np.savez_compressed(RUN / "artifacts/relations.npz", **relation_arrays)
    (RUN / "artifacts/manifest.json").write_text(json.dumps({
        "sampling_manifest": str(sampling_path.relative_to(ROOT)),
        "sampling_sha256": sha256(sampling_path),
        "teacher_features_sha256": sha256(feature_path),
        "relation_info": relation_info, "model_info": model_info,
        "prediction_sha256": sha256(RUN / "artifacts/predictions.npz"),
        "relation_sha256": sha256(RUN / "artifacts/relations.npz"),
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
