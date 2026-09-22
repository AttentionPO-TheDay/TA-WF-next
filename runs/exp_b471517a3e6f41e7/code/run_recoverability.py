#!/usr/bin/env python3
"""Frozen final-embedding recoverability diagnostic.

There is deliberately no backbone training or update path in this program.
Stages are separated as audit -> source -> evaluate -> summarize so current
query scoring cannot influence manifests or source-side selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ta_wf_next.models import DF, VarCNNDirection
from ta_wf_next.screening import TraceDataset, load_npz, sha256_file

RUN = ROOT / "runs" / "exp_b471517a3e6f41e7"
DONOR = ROOT / "runs" / "exp_9121b664a1854097"
SPLIT_PATH = ROOT / "runs" / "exp_6238dacf9aa142cc" / "artifacts" / "splits_v3.json"
DATA = Path("/mnt/data2/ren/datasets/TemporalDrift")
DATES = ("day14", "day90", "day270")
SHOTS = (3, 10)
SEEDS = (1729, 6238, 20260916)
CLASSES = 102
C_VALUES = (0.1, 1.0, 10.0)
CHECKPOINTS = {
    "df": {"file": "df_best.pt", "sha256": "1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae", "epoch": 29},
    "varcnn_direction": {"file": "varcnn_direction_best.pt", "sha256": "fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83", "epoch": 23},
}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite formal artifact: {path}")


def int_list_hash(values: list[int] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def string_list_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def admitted_hashes(x: np.ndarray, indices: np.ndarray | None = None) -> list[str]:
    selected = np.arange(len(x), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    output: list[str] = []
    for start in range(0, len(selected), 256):
        rows = np.sign(x[selected[start:start + 256], :5000]).astype(np.int8, copy=False)
        output.extend(hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in rows)
    return output


def verify_static_inputs(model_name: str | None = None) -> tuple[dict, dict | None, Path | None]:
    config = json.loads((RUN / "config.json").read_text())
    if config["maximum_new_backbone_training_runs"] != 0:
        raise ValueError("Backbone training budget is not zero")
    if sha256_file(SPLIT_PATH) != config["split"]["sha256"]:
        raise ValueError("Split hash mismatch")
    split = json.loads(SPLIT_PATH.read_text())
    if split.get("schema_version") != 3 or not split.get("all_102_classes_in_each_role"):
        raise ValueError("Unexpected split schema or class coverage")
    for filename, expected in split["source_files_sha256"].items():
        if sha256_file(DATA / filename) != expected:
            raise ValueError(f"Dataset hash mismatch: {filename}")
    if model_name is None:
        return split, None, None
    spec = CHECKPOINTS[model_name]
    path = DONOR / "checkpoints" / spec["file"]
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {model_name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != model_name or checkpoint.get("seed") != 6238 or checkpoint.get("epoch") != spec["epoch"]:
        raise ValueError(f"Checkpoint metadata mismatch: {model_name}")
    history = json.loads((DONOR / "artifacts" / f"{model_name}_history.json").read_text())
    best = max(history, key=lambda row: (row["val_macro_f1"], -row["epoch"]))
    if best["epoch"] != spec["epoch"] or best["val_macro_f1"] != checkpoint["selection_value"]:
        raise ValueError(f"Checkpoint/history mismatch: {model_name}")
    return split, checkpoint, path


def audit() -> None:
    audit_path = RUN / "artifacts" / "data_isolation_audit.json"
    manifest_path = RUN / "artifacts" / "support_manifests.json"
    refuse_existing(audit_path)
    refuse_existing(manifest_path)
    split, _, _ = verify_static_inputs()
    sx, sy = load_npz(DATA / "train.npz")
    vx, vy = load_npz(DATA / "valid.npz")
    source_by_role: dict[str, dict[str, list[int]]] = {}
    source_hash_union: set[str] = set()
    source_role_sets: dict[str, set[str]] = {}
    source_summaries = {}
    for role in ("supervised_train", "reference", "source_holdout"):
        indices = np.asarray(split["roles"][role]["indices"], dtype=np.int64)
        hashes = admitted_hashes(sx, indices)
        mapping: dict[str, list[int]] = {}
        for index, content_hash in zip(indices.tolist(), hashes):
            mapping.setdefault(content_hash, []).append(index)
        source_by_role[role] = mapping
        source_role_sets[role] = set(mapping)
        source_hash_union.update(mapping)
        source_summaries[role] = {"rows": len(indices), "unique_content": len(mapping), "duplicate_rows": len(indices) - len(mapping)}
    valid_hashes = admitted_hashes(vx)
    valid_mapping: dict[str, list[int]] = {}
    for index, content_hash in enumerate(valid_hashes):
        valid_mapping.setdefault(content_hash, []).append(index)
    source_by_role["source_validation"] = valid_mapping
    source_role_sets["source_validation"] = set(valid_mapping)
    source_hash_union.update(valid_mapping)
    source_summaries["source_validation"] = {"rows": len(vy), "unique_content": len(valid_mapping), "duplicate_rows": len(vy) - len(valid_mapping)}
    del sx, sy, vx, vy

    role_intersections = {}
    source_roles = list(source_role_sets)
    for i, left in enumerate(source_roles):
        for right in source_roles[i + 1:]:
            role_intersections[f"{left}__{right}"] = len(source_role_sets[left] & source_role_sets[right])

    audit_dates = {}
    manifests = {"schema_version": 1, "dates": {}, "seeds": list(SEEDS), "shots": list(SHOTS)}
    date_hash_sets: dict[str, set[str]] = {}
    infeasible = []
    for date in DATES:
        x, y = load_npz(DATA / f"{date}.npz")
        hashes = admitted_hashes(x)
        groups: dict[str, list[int]] = {}
        for index, content_hash in enumerate(hashes):
            groups.setdefault(content_hash, []).append(index)
        conflicts = []
        duplicate_groups = []
        canonical = []
        excluded_internal = []
        excluded_source = []
        overlap_by_role = {role: [] for role in source_roles}
        for content_hash, rows in groups.items():
            labels = sorted({int(y[row]) for row in rows})
            if len(labels) > 1:
                conflicts.append({"content_hash": content_hash, "rows": rows, "labels": labels})
                continue
            if len(rows) > 1:
                duplicate_groups.append({"content_hash": content_hash, "kept": min(rows), "excluded": sorted(rows)[1:], "label": labels[0]})
                excluded_internal.extend(sorted(rows)[1:])
            if content_hash in source_hash_union:
                excluded_source.extend(rows)
                for role in source_roles:
                    if content_hash in source_role_sets[role]:
                        overlap_by_role[role].append({"current_rows": sorted(rows), "source_rows": source_by_role[role][content_hash], "label": labels[0], "content_hash": content_hash})
            else:
                canonical.append(min(rows))
        if conflicts:
            infeasible.append({"date": date, "reason": "conflicting_labels_within_current", "groups": len(conflicts)})
        canonical = sorted(canonical)
        counts = {str(label): int(np.sum(y[canonical] == label)) for label in range(CLASSES)}
        insufficient = {label: count for label, count in counts.items() if count < 10}
        if insufficient:
            infeasible.append({"date": date, "reason": "fewer_than_10_canonical_rows", "classes": insufficient})
        date_hash_sets[date] = {hashes[row] for row in canonical}
        audit_dates[date] = {
            "total_rows": len(y), "raw_class_counts": {str(label): int(np.sum(y == label)) for label in range(CLASSES)},
            "unique_content_groups": len(groups), "within_date_duplicate_groups": len(duplicate_groups),
            "within_date_duplicate_rows_excluded": sorted(excluded_internal), "within_date_duplicate_details": duplicate_groups,
            "conflicting_label_groups": conflicts, "source_overlap_rows_excluded": sorted(set(excluded_source)),
            "source_overlap_details_by_role": overlap_by_role, "eligible_canonical_rows": len(canonical),
            "eligible_class_counts": counts, "classes_below_10": insufficient,
        }
        date_manifest = {
            "eligible_rows": [{"row_index": row, "label": int(y[row]), "content_hash": hashes[row]} for row in canonical],
            "eligible_row_indices_sha256": int_list_hash(canonical),
            "eligible_content_hashes_sha256": string_list_hash([hashes[row] for row in canonical]),
            "configurations": {},
        }
        for seed in SEEDS:
            rng = np.random.Generator(np.random.PCG64(seed))
            ten_by_class = {}
            for label in range(CLASSES):
                rows = np.asarray([row for row in canonical if int(y[row]) == label], dtype=np.int64)
                ten_by_class[label] = rng.permutation(rows)[:10].tolist() if len(rows) >= 10 else []
            for shot in SHOTS:
                support = sorted(row for label in range(CLASSES) for row in ten_by_class[label][:shot])
                support_set = set(support)
                query = [row for row in canonical if row not in support_set]
                support_hashes = [hashes[row] for row in support]
                query_hashes = [hashes[row] for row in query]
                if set(support) & set(query) or set(support_hashes) & set(query_hashes):
                    raise ValueError(f"Support/query overlap: {date} seed={seed} shot={shot}")
                key = f"seed{seed}_shot{shot}"
                date_manifest["configurations"][key] = {
                    "seed": seed, "shot": shot, "support_rows": support, "query_rows": query,
                    "support_content_hashes": support_hashes,
                    "support_rows_sha256": int_list_hash(support), "query_rows_sha256": int_list_hash(query),
                    "support_content_hashes_sha256": string_list_hash(support_hashes),
                    "query_content_hashes_sha256": string_list_hash(query_hashes),
                    "support_query_row_intersection": 0, "support_query_content_intersection": 0,
                }
            three = set(date_manifest["configurations"][f"seed{seed}_shot3"]["support_rows"])
            ten = set(date_manifest["configurations"][f"seed{seed}_shot10"]["support_rows"])
            if not three < ten:
                raise ValueError(f"3-shot is not a strict subset: {date} seed={seed}")
        manifests["dates"][date] = date_manifest
        del x, y

    cross_dates = {}
    for i, left in enumerate(DATES):
        for right in DATES[i + 1:]:
            shared = sorted(date_hash_sets[left] & date_hash_sets[right])
            cross_dates[f"{left}__{right}"] = {"count": len(shared), "content_hashes": shared}
    output = {
        "passed": not infeasible and all(value == 0 for value in role_intersections.values()),
        "plan_sha256": sha256_file(RUN / "RECOVERABILITY_PLAN.md"), "split_sha256": sha256_file(SPLIT_PATH),
        "dataset_sha256": split["source_files_sha256"], "source_roles": source_summaries,
        "source_role_content_intersections": role_intersections, "dates": audit_dates,
        "cross_current_date_content_intersections_after_source_exclusion": cross_dates,
        "infeasible": infeasible, "rules_applied_before_support_sampling": True,
    }
    atomic_json(manifest_path, manifests)
    output["support_manifest_sha256"] = sha256_file(manifest_path)
    atomic_json(audit_path, output)
    print(json.dumps({"audit": str(audit_path), "passed": output["passed"], "infeasible": infeasible}, indent=2))
    if not output["passed"]:
        raise SystemExit(2)


def make_model(name: str, checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model = DF(CLASSES) if name == "df" else VarCNNDirection(CLASSES)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def normalized(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms != 0)


def extract(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, indices: np.ndarray | None, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    loader = DataLoader(TraceDataset(x, y, indices), batch_size=128, shuffle=False, num_workers=0)
    features, logits, row_ids = [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for xb, _, rows in loader:
            out, feat = model(xb.to(device))
            if feat.ndim != 2 or feat.shape[1] != 512:
                raise ValueError(f"Unexpected final embedding shape: {tuple(feat.shape)}")
            features.append(feat.cpu().numpy())
            logits.append(out.cpu().numpy())
            row_ids.append(rows.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return normalized(np.concatenate(features)), np.concatenate(logits), np.concatenate(row_ids), time.perf_counter() - start


def new_linear(c_value: float) -> LogisticRegression:
    return LogisticRegression(penalty="l2", C=c_value, solver="lbfgs", max_iter=300, tol=1e-4,
                              fit_intercept=True, class_weight=None, random_state=6238)


def fit_with_record(classifier: LogisticRegression, x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(x, y)
    elapsed = time.perf_counter() - start
    warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return elapsed, warned


def basic_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    confusion = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    tp = np.diag(confusion).astype(np.float64)
    actual = confusion.sum(1)
    predicted = confusion.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return {"accuracy": float(tp.sum() / confusion.sum()), "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean())}


def scored(truth: np.ndarray, prediction: np.ndarray) -> dict:
    confusion = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    tp = np.diag(confusion).astype(np.float64)
    actual, predicted = confusion.sum(1), confusion.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    per_site = {str(label): {"support": int(actual[label]), "accuracy": float(recall[label]),
                             "precision": float(precision[label]), "recall": float(recall[label]), "f1": float(f1[label])}
                for label in range(CLASSES)}
    return basic_metrics(truth, prediction) | {"per_website": per_site}


def source(name: str) -> None:
    output_path = RUN / "artifacts" / f"{name}_source_linear.json"
    model_path = RUN / "artifacts" / f"{name}_g_source.joblib"
    refuse_existing(output_path); refuse_existing(model_path)
    audit_record = json.loads((RUN / "artifacts" / "data_isolation_audit.json").read_text())
    if not audit_record.get("passed"):
        raise ValueError("Data isolation audit did not pass")
    split, checkpoint, checkpoint_path = verify_static_inputs(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(name, checkpoint, device)
    sx, sy = load_npz(DATA / "train.npz"); vx, vy = load_npz(DATA / "valid.npz")
    train_indices = np.asarray(split["roles"]["supervised_train"]["indices"], dtype=np.int64)
    train_x, _, train_rows, train_seconds = extract(model, sx, sy, train_indices, device)
    val_x, _, val_rows, val_seconds = extract(model, vx, vy, None, device)
    train_y, val_y = sy[train_rows], vy[val_rows]
    del model, sx, sy, vx, vy
    torch.cuda.empty_cache()
    candidates, classifiers = [], []
    for c_value in C_VALUES:
        classifier = new_linear(c_value)
        fit_seconds, warned = fit_with_record(classifier, train_x, train_y)
        prediction = classifier.predict(val_x)
        candidates.append({"C": c_value, "source_validation": basic_metrics(val_y, prediction),
                           "fit_seconds": fit_seconds, "n_iter": classifier.n_iter_.tolist(), "convergence_warning": warned})
        classifiers.append(classifier)
    winner_index = max(range(len(candidates)), key=lambda i: (candidates[i]["source_validation"]["macro_f1"], -candidates[i]["C"]))
    winner = classifiers[winner_index]
    temporary = model_path.with_suffix(".joblib.tmp")
    joblib.dump(winner, temporary); temporary.replace(model_path)
    output = {
        "model": name, "checkpoint": {"path": str(checkpoint_path), "sha256": CHECKPOINTS[name]["sha256"],
        "best_epoch": CHECKPOINTS[name]["epoch"], "model_definition": "src/ta_wf_next/models/df.py::DF" if name == "df" else "src/ta_wf_next/models/varcnn.py::VarCNNDirection"},
        "embedding": {"location": "forward_features output / mlp input", "dimension": 512, "normalization": "row_l2"},
        "fit_role": "source supervised_train only", "selection_role": "official source validation only",
        "candidates": candidates, "selected_C": candidates[winner_index]["C"], "selected_validation": candidates[winner_index]["source_validation"],
        "model_artifact": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "extraction_seconds": {"supervised_train": train_seconds, "source_validation": val_seconds},
        "feature_extraction_device": str(device),
        "new_backbone_training_runs": 0,
    }
    atomic_json(output_path, output)
    print(json.dumps({"model": name, "selected_C": output["selected_C"], "validation_macro_f1": output["selected_validation"]["macro_f1"]}, indent=2))


def prototype_predict(support_x: np.ndarray, support_y: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    prototypes = np.stack([support_x[support_y == label].mean(0) for label in range(CLASSES)])
    prototypes = normalized(prototypes)
    return (query_x @ prototypes.T).argmax(1).astype(np.int64)


def evaluate(name: str, date: str) -> None:
    if date not in DATES:
        raise ValueError(date)
    output_paths = [RUN / "artifacts" / f"{name}_{date}_seed{seed}_shot{shot}.json" for seed in SEEDS for shot in SHOTS]
    for path in output_paths: refuse_existing(path)
    audit_record = json.loads((RUN / "artifacts" / "data_isolation_audit.json").read_text())
    if not audit_record.get("passed") or audit_record.get("infeasible"):
        raise ValueError("Audit not passed or contains infeasible configurations")
    manifest_path = RUN / "artifacts" / "support_manifests.json"
    if sha256_file(manifest_path) != audit_record["support_manifest_sha256"]:
        raise ValueError("Support manifest hash mismatch")
    manifests = json.loads(manifest_path.read_text())
    split, checkpoint, checkpoint_path = verify_static_inputs(name)
    source_record = json.loads((RUN / "artifacts" / f"{name}_source_linear.json").read_text())
    source_model_path = Path(source_record["model_artifact"]["path"])
    if sha256_file(source_model_path) != source_record["model_artifact"]["sha256"]:
        raise ValueError("G-source model hash mismatch")
    g_source = joblib.load(source_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(name, checkpoint, device)
    x, y = load_npz(DATA / f"{date}.npz")
    features, logits, rows, extract_seconds = extract(model, x, y, None, device)
    if not np.array_equal(rows, np.arange(len(y))): raise ValueError("Unexpected extraction row order")
    del model, x
    torch.cuda.empty_cache()
    c_value = float(source_record["selected_C"])
    date_manifest = manifests["dates"][date]
    staged = []
    for seed in SEEDS:
        for shot in SHOTS:
            key = f"seed{seed}_shot{shot}"
            item = date_manifest["configurations"][key]
            support_rows = np.asarray(item["support_rows"], dtype=np.int64)
            query_rows = np.asarray(item["query_rows"], dtype=np.int64)
            support_y = y[support_rows]
            classifier = new_linear(c_value)
            fit_seconds, warned = fit_with_record(classifier, features[support_rows], support_y)
            predictions = {
                "A": logits[query_rows].argmax(1).astype(np.int64),
                "G_source": g_source.predict(features[query_rows]).astype(np.int64),
                "G_current": classifier.predict(features[query_rows]).astype(np.int64),
                "simple_maintenance": prototype_predict(features[support_rows], support_y, features[query_rows]),
            }
            staged.append((seed, shot, item, support_rows, query_rows, predictions, classifier, fit_seconds, warned))
    # Query labels are accessed only after every prediction for this date is fixed.
    for seed, shot, item, support_rows, query_rows, predictions, classifier, fit_seconds, warned in staged:
        query_y = y[query_rows]
        metrics = {method: scored(query_y, prediction) for method, prediction in predictions.items()}
        deltas = {}
        for metric in ("accuracy", "macro_f1"):
            deltas[metric] = {
                "G_current_minus_A": metrics["G_current"][metric] - metrics["A"][metric],
                "G_current_minus_G_source": metrics["G_current"][metric] - metrics["G_source"][metric],
                "simple_minus_A": metrics["simple_maintenance"][metric] - metrics["A"][metric],
                "simple_minus_G_current": metrics["simple_maintenance"][metric] - metrics["G_current"][metric],
            }
        output = {
            "model": name, "date": date, "shot": shot, "seed": seed,
            "checkpoint": {"path": str(checkpoint_path), "sha256": CHECKPOINTS[name]["sha256"], "best_epoch": CHECKPOINTS[name]["epoch"]},
            "embedding": {"location": "forward_features output / mlp input", "dimension": 512, "normalization_for_G": "row_l2"},
            "manifest": {k: item[k] for k in ("support_rows_sha256", "query_rows_sha256", "support_content_hashes_sha256", "query_content_hashes_sha256")},
            "support_rows": support_rows.tolist(), "query_rows": query_rows.tolist(), "query_truth": query_y.tolist(),
            "predictions": {method: prediction.tolist() for method, prediction in predictions.items()},
            "metrics": metrics, "deltas": deltas,
            "G_current_fit": {"C_inherited_from_G_source": c_value, "fit_seconds": fit_seconds,
                              "n_iter": classifier.n_iter_.tolist(), "convergence_warning": warned,
                              "fit_rows": int(len(support_rows)), "query_labels_used_for_fit_or_selection": False},
            "date_embedding_extraction_seconds_shared_across_configurations": extract_seconds,
            "feature_extraction_device": str(device),
            "new_backbone_training_runs": 0,
        }
        path = RUN / "artifacts" / f"{name}_{date}_seed{seed}_shot{shot}.json"
        atomic_json(path, output)
        print(path)


def mean_sd(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1)), "values": array.tolist()}


def summarize() -> None:
    csv_path = RUN / "artifacts" / "metrics_long.csv"
    summary_path = RUN / "artifacts" / "summary.json"
    refuse_existing(csv_path); refuse_existing(summary_path)
    records, long_rows = [], []
    for name in CHECKPOINTS:
        for date in DATES:
            for seed in SEEDS:
                for shot in SHOTS:
                    path = RUN / "artifacts" / f"{name}_{date}_seed{seed}_shot{shot}.json"
                    record = json.loads(path.read_text()); records.append(record)
                    for method, metrics in record["metrics"].items():
                        long_rows.append({"backbone": name, "date": date, "shot": shot, "seed": seed, "method": method,
                                          **{metric: metrics[metric] for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader(); writer.writerows(long_rows)
    grouped = {}
    for name in CHECKPOINTS:
        grouped[name] = {}
        for date in DATES:
            grouped[name][date] = {}
            for shot in SHOTS:
                subset = [r for r in records if r["model"] == name and r["date"] == date and r["shot"] == shot]
                methods = {}
                for method in ("A", "G_source", "G_current", "simple_maintenance"):
                    methods[method] = {metric: mean_sd([r["metrics"][method][metric] for r in subset]) for metric in ("accuracy", "macro_f1")}
                deltas = {delta: {metric: mean_sd([r["deltas"][metric][delta] for r in subset]) for metric in ("accuracy", "macro_f1")}
                          for delta in ("G_current_minus_A", "G_current_minus_G_source", "simple_minus_A", "simple_minus_G_current")}
                grouped[name][date][str(shot)] = {"methods": methods, "deltas": deltas}
    shot_gains = {}
    for name in CHECKPOINTS:
        shot_gains[name] = {}
        for date in DATES:
            shot_gains[name][date] = {}
            for method in ("A", "G_source", "G_current", "simple_maintenance"):
                shot_gains[name][date][method] = {}
                for metric in ("accuracy", "macro_f1"):
                    values = []
                    for seed in SEEDS:
                        r3 = next(r for r in records if r["model"] == name and r["date"] == date and r["shot"] == 3 and r["seed"] == seed)
                        r10 = next(r for r in records if r["model"] == name and r["date"] == date and r["shot"] == 10 and r["seed"] == seed)
                        values.append(r10["metrics"][method][metric] - r3["metrics"][method][metric])
                    shot_gains[name][date][method][metric] = mean_sd(values)
    judgments = {}
    for shot in SHOTS:
        by_model = {}
        for name in CHECKPOINTS:
            qualifying = []
            for date in DATES:
                values = [r["deltas"]["macro_f1"]["G_current_minus_A"] for r in records if r["model"] == name and r["date"] == date and r["shot"] == shot]
                if all(value > 0 for value in values) and np.mean(values) >= .05: qualifying.append(date)
            by_model[name] = {"qualifying_dates": qualifying, "passes": len(qualifying) >= 2}
        judgments[f"stable_recovery_{shot}shot"] = {"by_backbone": by_model, "passes_both": all(v["passes"] for v in by_model.values())}
    ten_gain = {}
    proto_close = {}
    weak = {}
    for name in CHECKPOINTS:
        gain_dates = [date for date in DATES if shot_gains[name][date]["G_current"]["macro_f1"]["mean"] >= .02]
        ten_gain[name] = {"qualifying_dates": gain_dates, "passes": len(gain_dates) >= 2}
        close_dates = []
        all_simple_gap = []
        for date in DATES:
            vals = [r["deltas"]["macro_f1"]["simple_minus_G_current"] for r in records if r["model"] == name and r["date"] == date]
            if abs(float(np.mean(vals))) <= .02: close_dates.append(date)
            all_simple_gap.extend(vals)
        proto_close[name] = {"qualifying_dates": close_dates, "overall_mean_gap": float(np.mean(all_simple_gap)),
                             "passes": len(close_dates) >= 2 and abs(float(np.mean(all_simple_gap))) <= .02}
        ten_deltas = [r["deltas"]["macro_f1"]["G_current_minus_A"] for r in records if r["model"] == name and r["shot"] == 10]
        weak[name] = {"mean_10shot_G_current_minus_A": float(np.mean(ten_deltas)),
                      "passes_weak_condition": not judgments["stable_recovery_10shot"]["by_backbone"][name]["passes"] and float(np.mean(ten_deltas)) < .05}
    judgments["ten_minus_three_substantial"] = {"by_backbone": ten_gain, "passes_both": all(v["passes"] for v in ten_gain.values())}
    judgments["simple_close_to_G_current"] = {"by_backbone": proto_close, "passes_both": all(v["passes"] for v in proto_close.values())}
    judgments["weak_frozen_representation_condition"] = weak
    source = {name: json.loads((RUN / "artifacts" / f"{name}_source_linear.json").read_text()) for name in CHECKPOINTS}
    summary = {"grouped_seed_mean_sd": grouped, "paired_10shot_minus_3shot": shot_gains, "pre_registered_judgments": judgments,
               "source_linear_selection": {name: {"selected_C": value["selected_C"], "validation_macro_f1": value["selected_validation"]["macro_f1"]} for name, value in source.items()},
               "new_backbone_training_runs": 0}
    atomic_json(summary_path, summary)
    print(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("audit")
    source_parser = sub.add_parser("source"); source_parser.add_argument("--model", choices=CHECKPOINTS, required=True)
    eval_parser = sub.add_parser("evaluate"); eval_parser.add_argument("--model", choices=CHECKPOINTS, required=True); eval_parser.add_argument("--date", choices=DATES, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.stage == "audit": audit()
    elif args.stage == "source": source(args.model)
    elif args.stage == "evaluate": evaluate(args.model, args.date)
    else: summarize()


if __name__ == "__main__":
    main()
