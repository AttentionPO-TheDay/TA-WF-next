"""Explicit minimal helpers copied from lowlr Proteus port; GMM and pseudo pipeline excluded."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
R = Path(__file__).resolve().parent
ROOT = R.parents[1]
C = json.loads((R / "config.json").read_text())
from model import MultiViewModel
DEVICE = torch.device("cuda:0")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def save_json(name, value):
    path = R / "artifacts" / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    try:
        temporary.replace(path)
    except PermissionError:
        # Shared overlay filesystems can reject rename while allowing a
        # direct write. Preserve the atomic path when available, and fall
        # back without losing the run-local record.
        path.write_text(temporary.read_text())
        temporary.unlink(missing_ok=True)

def event(stage, **payload):
    with (R / "artifacts/events.jsonl").open("a") as handle:
        handle.write(json.dumps({"utc": time.time(), "stage": stage, **payload}, ensure_ascii=False) + "\n")
    print(stage, json.dumps(payload, ensure_ascii=False), flush=True)

def registry(state, summary):
    subprocess.run([sys.executable, str(ROOT / "scripts/experiment.py"), "status", "--id", R.name, "--state", state, "--summary", summary], check=True)

def metric(labels, predictions):
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(102), where=denominator > 0)
    return {"accuracy": float((labels == predictions).mean()), "macro_f1": float(f1.mean())}

def fp_rows(values):
    return [hashlib.sha256(np.ascontiguousarray(row, dtype=np.float32).tobytes()).hexdigest() for row in values]

def reconstruct(values):
    output = np.zeros((len(values), 2, 1800), np.float32)
    for index, row in enumerate(values):
        nonzero = row[row != 0]
        bins = np.minimum(np.floor(np.abs(nonzero).astype(np.float64) * 1799 / 80), 1799).astype(np.int64)
        channels = (nonzero < 0).astype(np.int64)
        output[index] = np.bincount(bins + 1800 * channels, minlength=3600).reshape(2, 1800)
    return output

def load_prepared(path, role=None):
    data = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    if "timestamps" not in data:
        data = data[role or "source"]
    timestamps = data["timestamps"].numpy().astype(np.float32, copy=False)
    tam = data["tam"].numpy().astype(np.float32, copy=False)
    labels = data["labels"].numpy().astype(np.int64, copy=False)
    rows = data["rows"].numpy().astype(np.int64, copy=False)
    assert timestamps.ndim == 2 and timestamps.shape[1] == 5000
    assert tam.shape == (len(timestamps), 2, 1800)
    assert labels.shape == rows.shape == (len(timestamps),)
    assert np.isfinite(timestamps).all() and np.isfinite(tam).all()
    assert (timestamps != 0).any(1).all()
    assert np.array_equal(reconstruct(timestamps), tam)
    return {"timestamps": timestamps, "tam": tam, "labels": labels, "rows": rows}

def load_model(seed):
    checkpoint = ROOT / C["base_checkpoint_run"] / "checkpoints" / f"wide_bn_{seed}_best.pt"
    model = MultiViewModel("wide_bn").to(DEVICE)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint

def forward_with_features(model, timestamps, tam):
    generated = model.generator(timestamps, tam)
    packet = generated["packet"]
    temporal = generated["time"]
    packet_valid = generated["packet_mask"]
    time_valid = generated["time_mask"]
    packet = model.packet_projection(packet) + model.packet_positions + model.view_types[0]
    temporal = model.time_projection(temporal) + model.time_positions + model.view_types[1]
    valid = torch.cat([packet_valid, time_valid], 1)
    padding = ~valid
    encoded = torch.cat([packet, temporal], 1).masked_fill(padding[..., None], 0)
    for layer in model.head:
        encoded = layer(encoded, src_key_padding_mask=padding)
        encoded = encoded.masked_fill(padding[..., None], 0)
    encoded = model.final_norm(encoded).masked_fill(padding[..., None], 0)
    packet_encoded = encoded[:, :100]
    time_encoded = encoded[:, 100:]
    packet_mean = (packet_encoded * packet_valid[..., None]).sum(1) / packet_valid.sum(1).clamp_min(1)[..., None]
    time_mean = (time_encoded * time_valid[..., None]).sum(1) / time_valid.sum(1).clamp_min(1)[..., None]
    features = torch.cat([packet_mean, time_mean], -1)
    return model.classifier(features), features

def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)

def compute_gaussian_kernel(source, target):
    sample_count = int(source.size(0)) + int(target.size(0))
    combined = torch.cat([source, target], dim=0)
    l2_distance = ((combined.unsqueeze(0) - combined.unsqueeze(1)) ** 2).sum(2)
    bandwidth = torch.sum(l2_distance) / (sample_count ** 2 - sample_count)
    bandwidth = torch.clamp(bandwidth, min=1e-5)
    return torch.exp(-l2_distance / (bandwidth + 1e-5))

def calculate_mmd_loss(source_features, target_features):
    batch_size = min(source_features.size(0), target_features.size(0))
    source_features = source_features[:batch_size]
    target_features = target_features[:batch_size]
    kernels = compute_gaussian_kernel(source_features, target_features)
    xx = kernels[:batch_size, :batch_size]
    yy = kernels[batch_size:, batch_size:]
    xy = kernels[:batch_size, batch_size:]
    yx = kernels[batch_size:, :batch_size]
    return torch.mean(xx + yy - xy - yx)

@torch.no_grad()
def predict(model, timestamps, tam, batch_size=128):
    outputs = []
    for start in range(0, len(timestamps), batch_size):
        xb = torch.from_numpy(timestamps[start : start + batch_size]).to(DEVICE)
        tb = torch.from_numpy(tam[start : start + batch_size]).to(DEVICE)
        outputs.append(forward_with_features(model, xb, tb)[0].cpu())
    return torch.cat(outputs).numpy()

def split_targets(source, role, target):
    source_fingerprints = set(fp_rows(source["timestamps"]))
    fingerprints = fp_rows(target["timestamps"])
    first = {}
    source_overlap = 0
    for index, fingerprint in enumerate(fingerprints):
        if fingerprint in source_fingerprints:
            source_overlap += 1
            continue
        first.setdefault(fingerprint, index)
    keep = np.array(sorted(first.values()), dtype=np.int64)
    keys = [fingerprints[index] for index in keep]
    order = np.argsort([hashlib.sha256(f"{role}|{fingerprint}|20261009".encode()).hexdigest() for fingerprint in keys])
    keep = keep[order]
    support = keep[: C["adaptation"]["target_support_rows"]]
    query = keep[C["adaptation"]["target_support_rows"] :]
    assert len(set(fingerprints[index] for index in support)) == len(support)
    assert set(fingerprints[index] for index in support).isdisjoint(set(fingerprints[index] for index in query))
    return support, query, {"raw_rows": len(target["timestamps"]), "source_overlap_excluded": source_overlap, "unique_after_source_exclusion": len(keep), "support_rows": len(support), "query_rows": len(query), "query_classes": int(len(np.unique(target["labels"][query])))}

def cell_metrics(labels, frozen_predictions, predictions):
    frozen = metric(labels, frozen_predictions)
    adapted = metric(labels, predictions)
    frozen_correct = frozen_predictions == labels
    frozen_wrong = ~frozen_correct
    return {**adapted, "base_accuracy": frozen["accuracy"], "base_macro_f1": frozen["macro_f1"], "negative_adaptation_rate": float((frozen_correct & (predictions != labels)).sum() / max(1, frozen_correct.sum())), "recovery_rate": float((frozen_wrong & (predictions == labels)).sum() / max(1, frozen_wrong.sum())), "changed_rate": float((predictions != frozen_predictions).mean())}
