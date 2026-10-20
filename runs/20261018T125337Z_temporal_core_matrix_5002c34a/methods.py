"""Run-local adaptation helpers, adapted from prior Proteus/Tent development runs."""
import numpy as np
import torch
from sklearn.mixture import GaussianMixture


def forward_features(model, timestamps, tam):
    generated = model.generator(timestamps, tam)
    packet, temporal = generated["packet"], generated["time"]
    packet_valid, time_valid = generated["packet_mask"], generated["time_mask"]
    packet = model.packet_projection(packet) + model.packet_positions + model.view_types[0]
    temporal = model.time_projection(temporal) + model.time_positions + model.view_types[1]
    padding = ~torch.cat([packet_valid, time_valid], 1)
    encoded = torch.cat([packet, temporal], 1).masked_fill(padding[..., None], 0)
    for layer in model.head:
        encoded = layer(encoded, src_key_padding_mask=padding).masked_fill(padding[..., None], 0)
    encoded = model.final_norm(encoded).masked_fill(padding[..., None], 0)
    p = encoded[:, :100]
    t = encoded[:, 100:]
    p = (p * packet_valid[..., None]).sum(1) / packet_valid.sum(1).clamp_min(1)[..., None]
    t = (t * time_valid[..., None]).sum(1) / time_valid.sum(1).clamp_min(1)[..., None]
    features = torch.cat([p, t], -1)
    return model.classifier(features), features


def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def mmd(source, target):
    n = min(len(source), len(target))
    joined = torch.cat([source[:n], target[:n]], 0)
    distance = ((joined[:, None] - joined[None, :]) ** 2).sum(-1)
    bandwidth = (distance.sum() / (4 * n * n - 2 * n)).clamp_min(1e-5)
    kernel = torch.exp(-distance / (bandwidth + 1e-5))
    xx = kernel[:n, :n]
    yy = kernel[n:, n:]
    xy = kernel[:n, n:]
    yx = kernel[n:, :n]
    return (xx + yy - xy - yx).mean()


@torch.no_grad()
def gmm_pseudo(model, support, device, batch_size, threshold):
    model.eval()
    entropies, predictions = [], []
    for i in range(0, len(support["timestamps"]), batch_size):
        x = support["timestamps"][i:i + batch_size].to(device)
        tam = support["tam"][i:i + batch_size].to(device)
        logits, _ = forward_features(model, x, tam)
        entropies.append(entropy(logits).cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
    raw = np.concatenate(entropies).astype(np.float64)
    span = raw.max() - raw.min()
    assert np.isfinite(span) and span > 0
    normalized = ((raw - raw.min()) / span).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, tol=1e-6, random_state=2024).fit(normalized)
    probability = gmm.predict_proba(normalized)[:, np.argmin(gmm.means_.ravel())]
    selected = probability >= threshold
    assert selected.sum() >= batch_size
    return {"selected": selected, "labels": np.concatenate(predictions).astype(np.int64), "coverage": float(selected.mean()), "means": gmm.means_.ravel().tolist()}
