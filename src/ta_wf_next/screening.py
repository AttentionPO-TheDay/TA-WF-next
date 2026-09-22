"""Frozen-encoder screening primitives for exp_6238dacf9aa142cc."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def indices_sha256(indices: Iterable[int]) -> str:
    array = np.asarray(list(indices), dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one immutable archive; caller releases arrays before moving to next date."""
    with np.load(path, allow_pickle=False) as archive:
        x = archive["X"]
        y = archive["y"]
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"Unexpected arrays in {path}: X{x.shape}, y{y.shape}")
    if not np.array_equal(y, y.astype(np.int64)):
        raise ValueError(f"Non-integral labels in {path}")
    return x, y.astype(np.int64)


class TraceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray | None = None):
        self.x = x
        self.y = y
        self.indices = np.arange(len(y), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Tensor]:
        index = int(self.indices[item])
        # Sign is the only timestamp-derived information admitted to either backbone.
        trace = np.sign(self.x[index, :5000]).astype(np.float32, copy=False)
        return torch.from_numpy(trace[None, :]), torch.tensor(self.y[index]), torch.tensor(index)


LEGACY_LOCAL_SPECS = {
    "df": {
        "layer": "feature_extraction",
        "channels": 256,
        "positions": 18,
        "effective_stride": 256,
        "theoretical_receptive_field": 1786,
        "center_formula": "213 + 256*i input-index units",
        "valid_position_start": 3,
        "valid_position_stop_exclusive": 16,
    },
    "varcnn_direction": {
        "layer": "dir_encoder.convs",
        "channels": 512,
        "positions": 157,
        "effective_stride": 32,
        "theoretical_receptive_field": 1755,
        "center_formula": "0.5 + 32*i input-index units",
        "valid_position_start": 28,
        "valid_position_stop_exclusive": 129,
    },
}

LOCAL_SPECS = {
    "df": {"layer": "feature_extraction.0", "channels": 32, "positions": 1249,
           "effective_stride": 4, "theoretical_receptive_field": 22,
           "center_formula": "3 + 4*i (pixel-edge coordinates)",
           "valid_position_start": 2, "valid_position_stop_exclusive": 1247},
    "varcnn_direction": {"layer": "dir_encoder.convs.0", "channels": 64, "positions": 1250,
                         "effective_stride": 4, "theoretical_receptive_field": 35,
                         "center_formula": "0.5 + 4*i (pixel-edge coordinates)",
                         "valid_position_start": 5, "valid_position_stop_exclusive": 1246},
}


def region_descriptors(local: Tensor, model_name: str, regions: int = 4) -> Tensor:
    """Pool valid, ordered conv positions into fixed regions; output [B,R,C]."""
    spec = LEGACY_LOCAL_SPECS[model_name]
    if tuple(local.shape[1:]) != (spec["channels"], spec["positions"]):
        raise ValueError(f"Unexpected {model_name} local shape {tuple(local.shape)}")
    valid = local[:, :, spec["valid_position_start"] : spec["valid_position_stop_exclusive"]]
    chunks = torch.tensor_split(valid, regions, dim=2)
    desc = torch.stack([chunk.mean(dim=2) for chunk in chunks], dim=1)
    return torch.nn.functional.normalize(desc, dim=2)


def valid_local_positions(inputs: Tensor, model_name: str) -> Tensor:
    """Conservative RF validity: every covered input must be nonzero and finite.

    Zero timestamps cannot be distinguished from padding after sign conversion.
    Treat all zero positions as unknown, including interior zeros; never infer
    packet validity from nonzero count alone. Bounds are inclusive packet indices.
    """
    if inputs.ndim != 3 or inputs.shape[1:] != (1, 5000):
        raise ValueError("Expected [batch, 1, 5000]")
    spec = LOCAL_SPECS[model_name]
    centers = torch.arange(spec["positions"], device=inputs.device) * spec["effective_stride"]
    left, right = ((centers - 8, centers + 13) if model_name == "df"
                   else (centers - 17, centers + 17))
    inside = (left >= 0) & (right < 5000)
    missing = ((inputs[:, 0] == 0) | ~torch.isfinite(inputs[:, 0])).long()
    prefix = torch.nn.functional.pad(missing.cumsum(1), (1, 0))
    gaps = prefix[:, (right + 1).clamp(0, 5000)] - prefix[:, left.clamp(0, 5000)]
    return inside[None, :] & (gaps == 0)


def masked_descriptors(local: Tensor, inputs: Tensor, model_name: str):
    """Return four regions, same-layer global mean, eligibility and RF counts.

    D and E share exactly the same positions. E averages raw positions (not
    normalized region means). Fewer than four valid positions means ineligible.
    """
    spec = LOCAL_SPECS[model_name]
    if tuple(local.shape[1:]) != (spec["channels"], spec["positions"]):
        raise ValueError("Unexpected local feature shape")
    mask = valid_local_positions(inputs, model_name)
    counts = mask.sum(1)
    eligible = counts >= 4
    regions = local.new_zeros((len(local), 4, local.shape[1]))
    pooled = local.new_zeros((len(local), local.shape[1]))
    for i in range(len(local)):
        if eligible[i]:
            values = local[i, :, mask[i]]
            pooled[i] = values.mean(1)
            regions[i] = torch.stack([part.mean(1) for part in torch.tensor_split(values, 4, dim=1)])
    return (torch.nn.functional.normalize(regions, dim=2),
            torch.nn.functional.normalize(pooled, dim=1), eligible, counts)


def classify_masked(query_global, query_local, query_same, query_ok,
                    ref_global, ref_local, ref_same, ref_ok, ref_labels):
    """B/C/D/E; D/E use the same eligible reference subset and fallback to C.

    If any reference class lacks eligible traces, D/E fall back for every query
    to avoid silently evaluating a reduced class set. Pairs=-1 marks fallback.
    No query labels are accepted. 'used' identifies actual regional inference.
    """
    classes = torch.unique(ref_labels, sorted=True)
    prototypes = torch.nn.functional.normalize(torch.stack([
        ref_global[ref_labels == label].mean(0) for label in classes]), dim=1)
    b = classes[(query_global @ prototypes.T).argmax(1)]
    c = ref_labels[(query_global @ ref_global.T).argmax(1)]
    d, e = c.clone(), c.clone()
    pairs = torch.full((len(c), 5), -1, dtype=torch.long, device=c.device)
    used = torch.zeros_like(query_ok)
    indices = torch.where(ref_ok)[0]
    coverage = torch.equal(torch.unique(ref_labels[ref_ok]), torch.unique(ref_labels))
    if coverage and query_ok.any():
        q = query_local[query_ok]; r = ref_local[ref_ok]
        similarities = (q.flatten(0, 1) @ r.flatten(0, 1).T).reshape(len(q), 4, len(r), 4)
        best, pairing = similarities.max(3)
        winner = best.mean(1).argmax(1)
        selected = pairing.permute(0, 2, 1)[torch.arange(len(q), device=q.device), winner]
        local_pairs = torch.cat((winner[:, None], selected), dim=1)
        d[query_ok] = ref_labels[ref_ok][winner]
        e[query_ok] = ref_labels[ref_ok][(query_same[query_ok] @ ref_same[ref_ok].T).argmax(1)]
        local_pairs[:, 0] = indices[local_pairs[:, 0]]
        pairs[query_ok] = local_pairs
        used = query_ok.clone()
    return b, c, d, e, pairs, used


def classify_features(
    query_global: Tensor,
    query_local: Tensor,
    ref_global: Tensor,
    ref_local: Tensor,
    ref_labels: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return B/C/D predictions and D's selected reference and region pairing."""
    classes = torch.unique(ref_labels, sorted=True)
    prototypes = torch.stack([ref_global[ref_labels == label].mean(0) for label in classes])
    prototypes = torch.nn.functional.normalize(prototypes, dim=1)
    pred_b = classes[(query_global @ prototypes.T).argmax(1)]
    pred_c = ref_labels[(query_global @ ref_global.T).argmax(1)]

    batch, q_regions, width = query_local.shape
    refs, r_regions, ref_width = ref_local.shape
    if width != ref_width:
        raise ValueError("Local descriptor widths differ")
    similarities = (query_local.reshape(batch * q_regions, width) @ ref_local.reshape(refs * r_regions, width).T)
    similarities = similarities.reshape(batch, q_regions, refs, r_regions)
    best_region_values, best_region_indices = similarities.max(dim=3)
    reference_scores = best_region_values.mean(dim=1)
    selected_ref = reference_scores.argmax(dim=1)
    pred_d = ref_labels[selected_ref]
    selected_pairs = best_region_indices.permute(0, 2, 1)[torch.arange(batch, device=query_local.device), selected_ref]
    return pred_b, pred_c, pred_d, torch.stack((selected_ref, selected_pairs[:, 0], selected_pairs[:, 1], selected_pairs[:, 2], selected_pairs[:, 3]), dim=1)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: int = 102) -> dict[str, float]:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    tp = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(0)
    actual = confusion.sum(1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return {
        "accuracy": float(tp.sum() / confusion.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
