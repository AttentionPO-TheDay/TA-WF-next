#!/usr/bin/env python3
"""Bounded frozen-representation diagnostic for exp_4b72ed0c9a354d54.

There is deliberately no backbone training path in this program.  The source,
freeze, and future stages are separate so future features cannot be evaluated
before the source matrix and immutable selection artifact exist.
"""
from __future__ import annotations

import argparse
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
from ta_wf_next.screening import TraceDataset, classification_metrics, load_npz, sha256_file

RUN = ROOT / "runs" / "exp_4b72ed0c9a354d54"
DONOR = ROOT / "runs" / "exp_9121b664a1854097"
SPLIT = ROOT / "runs" / "exp_6238dacf9aa142cc" / "artifacts" / "splits_v3.json"
DATA = Path("/mnt/data2/ren/datasets/TemporalDrift")

CHECKPOINTS = {
    "df": {"file": "df_best.pt", "sha256": "1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae", "epoch": 29},
    "varcnn_direction": {"file": "varcnn_direction_best.pt", "sha256": "fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83", "epoch": 23},
}

LAYERS = {
    "df": {
        "shallow": {"name": "feature_extraction.0", "channels": 32, "positions": 1249, "stride": 4, "left0": -8, "right0": 13, "rf": 22},
        "intermediate": {"name": "feature_extraction.1", "channels": 64, "positions": 311, "stride": 16, "left0": -40, "right0": 65, "rf": 106},
    },
    "varcnn_direction": {
        "shallow": {"name": "dir_encoder.convs.0", "channels": 64, "positions": 1250, "stride": 4, "left0": -17, "right0": 17, "rf": 35},
        "intermediate": {"name": "dir_encoder.convs.3", "channels": 128, "positions": 625, "stride": 8, "left0": -181, "right0": 181, "rf": 363},
    },
}

REPRESENTATIONS = ("shallow_global", "shallow_ordered4", "intermediate_global", "intermediate_ordered4")
C_VALUES = (0.1, 1.0, 10.0)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite formal artifact: {path}")


def verify_inputs(name: str) -> dict:
    config = json.loads((RUN / "config.json").read_text())
    if config["maximum_new_backbone_training_runs"] != 0:
        raise ValueError("Backbone training budget is not zero")
    if sha256_file(SPLIT) != config["split"]["sha256"]:
        raise ValueError("Split hash mismatch")
    split = json.loads(SPLIT.read_text())
    if not split.get("all_102_classes_in_each_role") or split.get("schema_version") != 3:
        raise ValueError("Unexpected split schema or coverage")
    for filename, expected in split["source_files_sha256"].items():
        if sha256_file(DATA / filename) != expected:
            raise ValueError(f"Dataset hash mismatch: {filename}")
    spec = CHECKPOINTS[name]
    checkpoint_path = DONOR / "checkpoints" / spec["file"]
    if sha256_file(checkpoint_path) != spec["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != name or checkpoint.get("seed") != 6238 or checkpoint.get("epoch") != spec["epoch"]:
        raise ValueError(f"Checkpoint metadata mismatch: {name}")
    history = json.loads((DONOR / "artifacts" / f"{name}_history.json").read_text())
    best = max(history, key=lambda row: (row["val_macro_f1"], -row["epoch"]))
    if best["epoch"] != spec["epoch"] or best["val_macro_f1"] != checkpoint["selection_value"]:
        raise ValueError(f"Checkpoint/history best conflict: {name}")
    return {"config": config, "split": split, "checkpoint": checkpoint, "checkpoint_path": checkpoint_path, "history_best": best}


def make_model(name: str, checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model = (DF(102) if name == "df" else VarCNNDirection(102))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def layer_mask(inputs: torch.Tensor, spec: dict) -> torch.Tensor:
    positions = torch.arange(spec["positions"], device=inputs.device)
    left = spec["left0"] + spec["stride"] * positions
    right = spec["right0"] + spec["stride"] * positions
    inside = (left >= 0) & (right < 5000)
    missing = ((inputs[:, 0] == 0) | ~torch.isfinite(inputs[:, 0])).long()
    prefix = torch.nn.functional.pad(missing.cumsum(1), (1, 0))
    gaps = prefix[:, (right + 1).clamp(0, 5000)] - prefix[:, left.clamp(0, 5000)]
    return inside[None, :] & (gaps == 0)


def pool_representations(local: torch.Tensor, inputs: torch.Tensor, spec: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuple(local.shape[1:]) != (spec["channels"], spec["positions"]):
        raise ValueError(f"Hook shape mismatch: got {tuple(local.shape)}, spec={spec}")
    mask = layer_mask(inputs, spec)
    counts = mask.sum(1)
    weights = mask.to(local.dtype)
    global_mean = torch.einsum("bcp,bp->bc", local, weights) / counts.clamp_min(1)[:, None]
    global_mean[counts == 0] = 0
    rank = mask.long().cumsum(1) - 1
    quotient = counts // 4
    remainder = counts % 4
    regions = []
    for region in range(4):
        start = region * quotient + torch.minimum(remainder, torch.full_like(remainder, region))
        size = quotient + (remainder > region).long()
        region_mask = mask & (rank >= start[:, None]) & (rank < (start + size)[:, None])
        pooled = torch.einsum("bcp,bp->bc", local, region_mask.to(local.dtype)) / size.clamp_min(1)[:, None]
        pooled[size == 0] = 0
        regions.append(pooled)
    region_tensor = torch.stack(regions, dim=1)
    ordered = region_tensor.flatten(1)
    global_mean = torch.nn.functional.normalize(global_mean, dim=1)
    ordered = torch.nn.functional.normalize(ordered, dim=1)
    region_tensor = torch.nn.functional.normalize(region_tensor, dim=2)
    ordered[counts < 4] = 0
    region_tensor[counts < 4] = 0
    return global_mean, ordered, region_tensor, counts


def extract(model: torch.nn.Module, name: str, device: torch.device, x: np.ndarray, y: np.ndarray, indices: np.ndarray | None) -> tuple[dict[str, np.ndarray], np.ndarray, dict, float]:
    loader = DataLoader(TraceDataset(x, y, indices), batch_size=128, shuffle=False, num_workers=0)
    modules = dict(model.named_modules())
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for tier, spec in LAYERS[name].items():
        def hook(module, hook_inputs, output, tier=tier):
            captured[tier] = output
        handles.append(modules[spec["name"]].register_forward_hook(hook))
    chunks = {key: [] for key in REPRESENTATIONS}
    chunks.update({"shallow_unordered4": [], "intermediate_unordered4": []})
    labels = []
    counts = {tier: [] for tier in ("shallow", "intermediate")}
    start = time.perf_counter()
    with torch.inference_mode():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            captured.clear()
            model(xb)
            for tier in ("shallow", "intermediate"):
                glob, ordered, regions, count = pool_representations(captured[tier], xb, LAYERS[name][tier])
                chunks[f"{tier}_global"].append(glob.cpu().numpy())
                chunks[f"{tier}_ordered4"].append(ordered.cpu().numpy())
                chunks[f"{tier}_unordered4"].append(regions.cpu().numpy())
                counts[tier].append(count.cpu().numpy())
            labels.append(yb.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    for handle in handles:
        handle.remove()
    arrays = {key: np.concatenate(value) for key, value in chunks.items()}
    all_counts = {tier: np.concatenate(value) for tier, value in counts.items()}
    coverage = {
        tier: {
            "minimum": int(value.min()),
            "quantiles": np.quantile(value, [0, .1, .5, .9, 1]).tolist(),
            "zero_effective": int((value == 0).sum()),
            "fewer_than_four": int((value < 4).sum()),
            "count": int(len(value)),
        }
        for tier, value in all_counts.items()
    }
    return arrays, np.concatenate(labels), coverage, elapsed


def per_website_accuracy(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {str(label): float((prediction[truth == label] == label).mean()) for label in range(102)}


def scored(truth: np.ndarray, prediction: np.ndarray) -> dict:
    return classification_metrics(truth, prediction) | {"per_website_accuracy": per_website_accuracy(truth, prediction)}


def cosine_predict(query: np.ndarray, reference: np.ndarray, reference_labels: np.ndarray) -> np.ndarray:
    output = []
    for start in range(0, len(query), 4096):
        output.append(reference_labels[(query[start:start + 4096] @ reference.T).argmax(1)])
    return np.concatenate(output)


def unordered_predict(query: np.ndarray, reference: np.ndarray, reference_labels: np.ndarray, device: torch.device) -> np.ndarray:
    ref = torch.from_numpy(reference).to(device)
    output = []
    with torch.inference_mode():
        for start in range(0, len(query), 256):
            q = torch.from_numpy(query[start:start + 256]).to(device)
            similarities = (q.flatten(0, 1) @ ref.flatten(0, 1).T).reshape(len(q), 4, len(ref), 4)
            winner = similarities.max(3).values.mean(1).argmax(1)
            output.append(reference_labels[winner.cpu().numpy()])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(output)


def source(name: str) -> None:
    output_path = RUN / "artifacts" / f"{name}_source_diagnostic.json"
    prediction_path = RUN / "artifacts" / f"{name}_source_predictions.json"
    refuse_existing(output_path)
    refuse_existing(prediction_path)
    checked = verify_inputs(name)
    if checked["config"]["status"] != "source_only":
        raise ValueError("Source stage requires config status=source_only")
    if not torch.cuda.is_available():
        raise RuntimeError("Full source extraction requires CUDA; do not silently change compute protocol")
    device = torch.device("cuda")
    model = make_model(name, checked["checkpoint"], device)
    sx, sy = load_npz(DATA / "train.npz")
    vx, vy = load_npz(DATA / "valid.npz")
    roles = checked["split"]["roles"]
    extracted = {}
    extraction_cost = {}
    coverage = {}
    for role, values, labels, indices in (
        ("supervised_train", sx, sy, np.asarray(roles["supervised_train"]["indices"], dtype=np.int64)),
        ("reference", sx, sy, np.asarray(roles["reference"]["indices"], dtype=np.int64)),
        ("source_holdout", sx, sy, np.asarray(roles["source_holdout"]["indices"], dtype=np.int64)),
        ("source_validation", vx, vy, None),
    ):
        reps, truth, role_coverage, elapsed = extract(model, name, device, values, labels, indices)
        extracted[role] = (reps, truth)
        coverage[role] = role_coverage
        extraction_cost[role] = elapsed
    del model
    torch.cuda.empty_cache()

    reference_labels = extracted["reference"][1]
    results = {}
    predictions = {"source_validation": {}, "source_holdout": {}}
    model_records = {}
    for representation in REPRESENTATIONS:
        train_x, train_y = extracted["supervised_train"][0][representation], extracted["supervised_train"][1]
        ref_x = extracted["reference"][0][representation]
        val_x, val_y = extracted["source_validation"][0][representation], extracted["source_validation"][1]
        hold_x, hold_y = extracted["source_holdout"][0][representation], extracted["source_holdout"][1]
        t0 = time.perf_counter()
        cosine_val = cosine_predict(val_x, ref_x, reference_labels)
        cosine_hold = cosine_predict(hold_x, ref_x, reference_labels)
        cosine_seconds = time.perf_counter() - t0
        cosine_key = representation + "__cosine_1nn"
        results[cosine_key] = {
            "representation": representation,
            "readout": "cosine_1nn_204_frozen_references",
            "source_validation": scored(val_y, cosine_val),
            "source_holdout": scored(hold_y, cosine_hold),
            "cost": {"fit_seconds": 0.0, "validation_and_holdout_predict_seconds": cosine_seconds, "reference_bytes": int(ref_x.nbytes), "dimension": int(ref_x.shape[1])},
        }
        predictions["source_validation"][cosine_key] = cosine_val.tolist()
        predictions["source_holdout"][cosine_key] = cosine_hold.tolist()

        candidates = []
        candidate_models = []
        for c_value in C_VALUES:
            classifier = LogisticRegression(
                penalty="l2", C=c_value, solver="lbfgs", max_iter=300, tol=1e-4,
                fit_intercept=True, class_weight=None, random_state=6238,
            )
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(train_x, train_y)
            fit_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            val_pred = classifier.predict(val_x)
            predict_seconds = time.perf_counter() - t0
            val_metrics = scored(val_y, val_pred)
            candidates.append({
                "C": c_value, "source_validation": val_metrics,
                "fit_seconds": fit_seconds, "validation_predict_seconds": predict_seconds,
                "n_iter": classifier.n_iter_.tolist(),
                "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in caught),
            })
            candidate_models.append(classifier)
        winner_index = max(range(len(candidates)), key=lambda i: (candidates[i]["source_validation"]["macro_f1"], -candidates[i]["C"]))
        winner = candidates[winner_index]
        classifier = candidate_models[winner_index]
        t0 = time.perf_counter()
        linear_val = classifier.predict(val_x)
        linear_hold = classifier.predict(hold_x)
        final_predict_seconds = time.perf_counter() - t0
        linear_key = representation + "__linear"
        model_path = RUN / "artifacts" / f"{name}_{representation}_linear.joblib"
        refuse_existing(model_path)
        temporary = model_path.with_suffix(".joblib.tmp")
        joblib.dump(classifier, temporary)
        temporary.replace(model_path)
        results[linear_key] = {
            "representation": representation,
            "readout": "l2_multinomial_logistic_regression_supervised_train",
            "selected_C": winner["C"],
            "source_validation": scored(val_y, linear_val),
            "source_holdout": scored(hold_y, linear_hold),
            "selection_candidates": candidates,
            "cost": {
                "selected_fit_seconds": winner["fit_seconds"],
                "all_candidate_fit_seconds": sum(row["fit_seconds"] for row in candidates),
                "validation_and_holdout_predict_seconds": final_predict_seconds,
                "model_bytes": model_path.stat().st_size,
                "dimension": int(train_x.shape[1]),
            },
        }
        predictions["source_validation"][linear_key] = linear_val.tolist()
        predictions["source_holdout"][linear_key] = linear_hold.tolist()
        model_records[representation] = {"path": str(model_path), "sha256": sha256_file(model_path), "selected_C": winner["C"]}

    for tier in ("shallow", "intermediate"):
        representation = tier + "_unordered4"
        ref_x = extracted["reference"][0][representation]
        val_x, val_y = extracted["source_validation"][0][representation], extracted["source_validation"][1]
        hold_x, hold_y = extracted["source_holdout"][0][representation], extracted["source_holdout"][1]
        t0 = time.perf_counter()
        unordered_val = unordered_predict(val_x, ref_x, reference_labels, device)
        unordered_hold = unordered_predict(hold_x, ref_x, reference_labels, device)
        elapsed = time.perf_counter() - t0
        key = representation + "__cosine_region_permutation_invariant"
        results[key] = {
            "representation": representation,
            "readout": "cosine_region_permutation_invariant_204_frozen_references",
            "source_validation": scored(val_y, unordered_val),
            "source_holdout": scored(hold_y, unordered_hold),
            "cost": {"fit_seconds": 0.0, "validation_and_holdout_predict_seconds": elapsed, "reference_bytes": int(ref_x.nbytes), "dimension": [4, int(ref_x.shape[2])]},
        }
        predictions["source_validation"][key] = unordered_val.tolist()
        predictions["source_holdout"][key] = unordered_hold.tolist()

    old = json.loads((DONOR / "artifacts" / f"{name}_evaluation.json").read_text())
    old_source = old["domains"]["source_holdout"]
    controls = {key: old_source[key] | {"per_website_accuracy": old_source["per_website_accuracy"][key]} for key in "ACDE"}
    output = {
        "model": name,
        "source_only": True,
        "checkpoint": {"path": str(checked["checkpoint_path"]), "sha256": CHECKPOINTS[name]["sha256"], "epoch": CHECKPOINTS[name]["epoch"], "history_best": checked["history_best"]},
        "split": {"path": str(SPLIT), "sha256": sha256_file(SPLIT), "role_counts": {key: value["count"] for key, value in roles.items()}},
        "layer_specs": LAYERS[name],
        "coverage": coverage,
        "extraction_seconds": extraction_cost,
        "matrix": results,
        "old_frozen_controls_not_new_repeats": controls,
        "linear_models": model_records,
        "new_backbone_training_runs": 0,
    }
    atomic_json(output_path, output)
    atomic_json(prediction_path, {"truth_scoring_only": {"source_validation": extracted["source_validation"][1].tolist(), "source_holdout": extracted["source_holdout"][1].tolist()}, "predictions": predictions})
    print(output_path)


def audit() -> None:
    output_path = RUN / "artifacts" / "layer_source_audit.json"
    refuse_existing(output_path)
    summaries = {}
    for name in CHECKPOINTS:
        checked = verify_inputs(name)
        model = make_model(name, checked["checkpoint"], torch.device("cpu"))
        shapes = {}
        captured = {}
        handles = []
        for tier, spec in LAYERS[name].items():
            def hook(module, hook_inputs, output, tier=tier):
                captured[tier] = output
            handles.append(dict(model.named_modules())[spec["name"]].register_forward_hook(hook))
        with torch.inference_mode():
            model(torch.zeros(2, 1, 5000))
        for tier, value in captured.items():
            shapes[tier] = list(value.shape)
        for handle in handles:
            handle.remove()
        sx, sy = load_npz(DATA / "train.npz")
        vx, vy = load_npz(DATA / "valid.npz")
        role_summaries = {}
        for role, values, labels, indices in (
            *((key, sx, sy, np.asarray(value["indices"], dtype=np.int64)) for key, value in checked["split"]["roles"].items()),
            ("source_validation", vx, vy, np.arange(len(vy), dtype=np.int64)),
        ):
            rows = np.sign(values[indices, :5000]).astype(np.float32)
            observed = np.where(rows != 0, np.arange(1, 5001), 0).max(1)
            layer_counts = {}
            for tier, spec in LAYERS[name].items():
                counts = []
                for start in range(0, len(rows), 256):
                    xb = torch.from_numpy(rows[start:start + 256])[:, None, :]
                    counts.extend(layer_mask(xb, spec).sum(1).tolist())
                count_array = np.asarray(counts)
                layer_counts[tier] = {
                    "minimum": int(count_array.min()), "quantiles": np.quantile(count_array, [0, .1, .5, .9, 1]).tolist(),
                    "zero_effective": int((count_array == 0).sum()), "fewer_than_four": int((count_array < 4).sum()),
                    "eligible_classes_at_least_four": int(np.unique(labels[indices][count_array >= 4]).size),
                }
            role_summaries[role] = {
                "count": int(len(rows)), "observed_span_quantiles": np.quantile(observed, [0, .1, .5, .9, 1]).tolist(),
                "shorter_than_5000": int((observed < 5000).sum()), "all_zero": int((observed == 0).sum()), "layers": layer_counts,
            }
        summaries[name] = {"hook_shapes": shapes, "roles": role_summaries}
    atomic_json(output_path, {"source_only": True, "no_samples_filtered": True, "models": summaries})
    print(output_path)


def freeze() -> None:
    output_path = RUN / "artifacts" / "frozen_selection.json"
    refuse_existing(output_path)
    config = json.loads((RUN / "config.json").read_text())
    if config["status"] != "source_only":
        raise ValueError("Freeze requires source_only status")
    selected = {}
    source_hashes = {}
    simplicity_rep = {"shallow_global": 5, "intermediate_global": 4, "shallow_ordered4": 3, "intermediate_ordered4": 2, "shallow_unordered4": 1, "intermediate_unordered4": 0}
    for name in CHECKPOINTS:
        path = RUN / "artifacts" / f"{name}_source_diagnostic.json"
        if not path.is_file():
            raise FileNotFoundError(f"Complete both source matrices before freeze: {path}")
        result = json.loads(path.read_text())
        source_hashes[name] = sha256_file(path)
        eligible = []
        for key, row in result["matrix"].items():
            score = row["source_validation"]["macro_f1"]
            if score >= config["future_gate"]["minimum_source_validation_macro_f1"]:
                readout_simple = 1 if row["readout"].startswith("cosine") else 0
                c_simple = -float(row.get("selected_C", 0.0))
                eligible.append((score, simplicity_rep[row["representation"]], readout_simple, c_simple, key, row))
        if eligible:
            *_, key, row = max(eligible)
            item = {"matrix_key": key, "representation": row["representation"], "readout": row["readout"], "source_validation": row["source_validation"], "source_holdout": row["source_holdout"]}
            if "selected_C" in row:
                model_record = result["linear_models"][row["representation"]]
                if sha256_file(Path(model_record["path"])) != model_record["sha256"]:
                    raise ValueError("Linear model hash mismatch before freeze")
                item["selected_C"] = row["selected_C"]
                item["linear_model"] = model_record
            selected[name] = item
        else:
            selected[name] = None
    atomic_json(output_path, {
        "created_after_complete_source_matrix": True,
        "selection_rule": config["future_gate"],
        "source_artifact_sha256": source_hashes,
        "selected": selected,
        "future_dates": config["future_dates"],
        "new_backbone_training_runs": 0,
    })
    print(output_path)


def extract_one_rep(model, name, device, x, y, representation):
    tier = representation.split("_")[0]
    aggregation = "unordered4" if representation.endswith("unordered4") else ("ordered4" if representation.endswith("ordered4") else "global")
    loader = DataLoader(TraceDataset(x, y), batch_size=128, shuffle=False, num_workers=0)
    module = dict(model.named_modules())[LAYERS[name][tier]["name"]]
    captured = {}
    def hook(hook_module, hook_inputs, output):
        captured["value"] = output
    handle = module.register_forward_hook(hook)
    features, labels = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            captured.clear()
            model(xb)
            glob, ordered, regions, _ = pool_representations(captured["value"], xb, LAYERS[name][tier])
            features.append((regions if aggregation == "unordered4" else (ordered if aggregation == "ordered4" else glob)).cpu().numpy())
            labels.append(yb.numpy())
    torch.cuda.synchronize()
    handle.remove()
    return np.concatenate(features), np.concatenate(labels), time.perf_counter() - start


def future(name: str) -> None:
    output_path = RUN / "artifacts" / f"{name}_future_diagnostic.json"
    refuse_existing(output_path)
    checked = verify_inputs(name)
    if checked["config"]["status"] != "future_frozen":
        raise ValueError("Future stage requires config status=future_frozen")
    frozen_path = RUN / "artifacts" / "frozen_selection.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("Run source and freeze before future")
    frozen = json.loads(frozen_path.read_text())
    source_path = RUN / "artifacts" / f"{name}_source_diagnostic.json"
    if sha256_file(source_path) != frozen["source_artifact_sha256"][name]:
        raise ValueError("Source artifact changed after freeze")
    selection = frozen["selected"][name]
    if selection is None:
        atomic_json(output_path, {"model": name, "skipped": "no source-usable frozen configuration", "new_backbone_training_runs": 0})
        print(output_path)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Future extraction requires CUDA")
    device = torch.device("cuda")
    model = make_model(name, checked["checkpoint"], device)
    representation = selection["representation"]
    classifier = None
    reference_x = reference_y = None
    if selection["readout"].startswith("l2_multinomial"):
        model_path = Path(selection["linear_model"]["path"])
        if sha256_file(model_path) != selection["linear_model"]["sha256"]:
            raise ValueError("Frozen linear model hash mismatch")
        classifier = joblib.load(model_path)
    else:
        sx, sy = load_npz(DATA / "train.npz")
        ref_indices = np.asarray(checked["split"]["roles"]["reference"]["indices"], dtype=np.int64)
        ref_reps, reference_y, _, _ = extract(model, name, device, sx, sy, ref_indices)
        reference_x = ref_reps[representation]
    old = json.loads((DONOR / "artifacts" / f"{name}_evaluation.json").read_text())
    results, predictions = {}, {}
    source_metrics = selection["source_holdout"]
    for domain in checked["config"]["future_dates"]:
        x, y = load_npz(DATA / f"{domain}.npz")
        features, truth, extraction_seconds = extract_one_rep(model, name, device, x, y, representation)
        t0 = time.perf_counter()
        prediction = (classifier.predict(features) if classifier is not None else
                      (unordered_predict(features, reference_x, reference_y, device) if representation.endswith("unordered4") else cosine_predict(features, reference_x, reference_y)))
        readout_seconds = time.perf_counter() - t0
        metrics = scored(truth, prediction)
        c_metrics = old["domains"][domain]["C"] | {"per_website_accuracy": old["domains"][domain]["per_website_accuracy"]["C"]}
        results[domain] = {
            "selected": metrics,
            "old_C_hard_control": c_metrics,
            "drop_from_source_holdout": {metric: metrics[metric] - source_metrics[metric] for metric in ("accuracy", "macro_f1")},
            "selected_minus_C": {metric: metrics[metric] - c_metrics[metric] for metric in ("accuracy", "macro_f1")},
            "cost": {"samples": int(len(truth)), "feature_extraction_seconds": extraction_seconds, "readout_seconds": readout_seconds, "dimension": list(features.shape[1:]), "representation_bytes": int(features.nbytes)},
        }
        predictions[domain] = {"truth_scoring_only": truth.tolist(), "selected_predictions": prediction.tolist()}
        del x, y, features
    atomic_json(output_path, {"model": name, "frozen_selection_sha256": sha256_file(frozen_path), "selection": selection, "domains": results, "new_backbone_training_runs": 0})
    atomic_json(RUN / "artifacts" / f"{name}_future_predictions.json", predictions)
    print(output_path)


def smoke() -> None:
    for name in CHECKPOINTS:
        checked = verify_inputs(name)
        model = make_model(name, checked["checkpoint"], torch.device("cpu"))
        x = torch.zeros(2, 1, 5000)
        modules = dict(model.named_modules())
        captured = {}
        handles = []
        for tier, spec in LAYERS[name].items():
            def hook(module, hook_inputs, output, tier=tier): captured[tier] = output
            handles.append(modules[spec["name"]].register_forward_hook(hook))
        with torch.inference_mode(): model(x)
        for tier, spec in LAYERS[name].items():
            glob, ordered, regions, counts = pool_representations(captured[tier], x, spec)
            assert glob.shape == (2, spec["channels"])
            assert ordered.shape == (2, 4 * spec["channels"])
            assert regions.shape == (2, 4, spec["channels"])
            assert torch.isfinite(glob).all() and torch.isfinite(ordered).all() and torch.isfinite(regions).all() and (counts == 0).all()
        for handle in handles: handle.remove()
        print(name, "smoke OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "audit", "source", "freeze", "future"))
    parser.add_argument("--model", choices=tuple(CHECKPOINTS))
    args = parser.parse_args()
    (RUN / "artifacts").mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    if args.stage in ("source", "future") and not args.model:
        parser.error("--model is required for source/future")
    {"smoke": smoke, "audit": audit, "source": lambda: source(args.model), "freeze": freeze, "future": lambda: future(args.model)}[args.stage]()


if __name__ == "__main__":
    main()
