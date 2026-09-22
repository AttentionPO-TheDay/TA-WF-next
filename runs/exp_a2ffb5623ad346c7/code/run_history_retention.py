#!/usr/bin/env python3
"""Historical-retention baselines on frozen donor embeddings.

Feature extraction and metric helpers are minimally adapted from
exp_b471517a3e6f41e7/code/run_recoverability.py. There is no backbone
training, backward pass, checkpoint write, or manifest construction path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.optimize import minimize
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ta_wf_next.models import DF, VarCNNDirection
from ta_wf_next.screening import TraceDataset, load_npz, sha256_file

RUN = ROOT / "runs/exp_a2ffb5623ad346c7"
DONOR = ROOT / "runs/exp_b471517a3e6f41e7"
CHECKPOINT_RUN = ROOT / "runs/exp_9121b664a1854097"
SPLIT_PATH = ROOT / "runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json"
DATA = Path("/mnt/data2/ren/datasets/TemporalDrift")
PLAN_HASH = "909850d12eed764d0f881bd7d61860d2f06d189ebdcaf200f9a69ba4d4959561"
MODELS = ("df", "varcnn_direction")
DATES = ("day14", "day90", "day270")
SHOTS = (3, 10)
SEEDS = (1729, 6238, 20260916)
CLASSES = 102
ALPHAS = (0.25, 0.50, 0.75)
LAMBDAS = (0.01, 0.10, 1.00)
CHECKPOINTS = {
    "df": ("df_best.pt", "1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae", 29),
    "varcnn_direction": ("varcnn_direction_best.pt", "fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83", 23),
}
FROZEN = {
    DONOR / "RECOVERABILITY_PLAN.md": "0db36bb1196f29a993b4b28c7fae5478d1d9d323efe81e58bd4967ed7c9bae7b",
    DONOR / "artifacts/data_isolation_audit.json": "6f1da408573f70cdf5ed64b80c9ce7b83ed56691a88208cb86a607403b3a17d6",
    DONOR / "artifacts/support_manifests.json": "cbf37c99af375c576e882e049ffb4a37946ea631099d6b15f1ba9458782725ca",
    DONOR / "artifacts/df_g_source.joblib": "895df6445a7ef7a068704a00f23f5fb84efbde4df7b6acf48aa5da4e85a27949",
    DONOR / "artifacts/df_source_linear.json": "94e863b59257f1544cb2832c441ff9709aea2b9c06f68dd961990985adce8e94",
    DONOR / "artifacts/varcnn_direction_g_source.joblib": "9e1d92c0fd5703473e4bbf4bd93e7cb6a7342100197a9e760a85b5c3d1fa777e",
    DONOR / "artifacts/varcnn_direction_source_linear.json": "232db3af1053c0ea389e5ac49a66635de1bced71bbc108e14b9a2aeaa2900872",
    SPLIT_PATH: "0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142",
    CHECKPOINT_RUN / "checkpoints/df_best.pt": CHECKPOINTS["df"][1],
    CHECKPOINT_RUN / "checkpoints/varcnn_direction_best.pt": CHECKPOINTS["varcnn_direction"][1],
}


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite formal artifact: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def list_hash(values: list[int] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def normalized(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms != 0)


def metrics(truth: np.ndarray, pred: np.ndarray, per_site: bool = True) -> dict:
    confusion = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    np.add.at(confusion, (truth, pred), 1)
    tp = np.diag(confusion).astype(np.float64)
    actual, predicted = confusion.sum(1), confusion.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall != 0)
    out = {"accuracy": float(tp.sum() / confusion.sum()), "macro_precision": float(precision.mean()),
           "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean())}
    if per_site:
        out["per_website"] = {str(c): {"support": int(actual[c]), "accuracy": float(recall[c]),
            "precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c])} for c in range(CLASSES)}
    return out


def mean_sd(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "values": a.tolist()}


def make_model(name: str, checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model = DF(CLASSES) if name == "df" else VarCNNDirection(CLASSES)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def extract(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, indices: np.ndarray | None,
            device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    loader = DataLoader(TraceDataset(x, y, indices), batch_size=128, shuffle=False, num_workers=0)
    features, logits, rows = [], [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for xb, _, row in loader:
            out, feat = model(xb.to(device))
            if feat.shape[1:] != (512,):
                raise ValueError(f"Unexpected embedding: {feat.shape}")
            features.append(feat.cpu().numpy()); logits.append(out.cpu().numpy()); rows.append(row.numpy())
    return normalized(np.concatenate(features)), np.concatenate(logits), np.concatenate(rows), time.perf_counter() - start


def load_checkpoint(name: str) -> tuple[dict, Path]:
    filename, expected, epoch = CHECKPOINTS[name]
    path = CHECKPOINT_RUN / "checkpoints" / filename
    if sha256_file(path) != expected:
        raise ValueError(f"checkpoint hash mismatch: {name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != name or checkpoint.get("seed") != 6238 or checkpoint.get("epoch") != epoch:
        raise ValueError(f"checkpoint metadata mismatch: {name}")
    history = json.loads((CHECKPOINT_RUN / "artifacts" / f"{name}_history.json").read_text())
    best = max(history, key=lambda r: (r["val_macro_f1"], -r["epoch"]))
    if best["epoch"] != epoch or best["val_macro_f1"] != checkpoint["selection_value"]:
        raise ValueError(f"checkpoint history mismatch: {name}")
    return checkpoint, path


def source_head(name: str):
    model = joblib.load(DONOR / "artifacts" / f"{name}_g_source.joblib")
    if model.coef_.shape != (CLASSES, 512) or model.intercept_.shape != (CLASSES,):
        raise ValueError(f"unexpected G-source shape: {name}")
    if not np.array_equal(model.classes_, np.arange(CLASSES)):
        raise ValueError(f"unexpected G-source classes: {name}")
    return model


def prototype(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return normalized(np.stack([x[y == c].mean(0) for c in range(CLASSES)]))


def top_margin(scores: np.ndarray) -> np.ndarray:
    top = np.partition(scores, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def shrink_objective(theta: np.ndarray, x: np.ndarray, y: np.ndarray, w0: np.ndarray,
                     b0: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    w = theta[:CLASSES * x.shape[1]].reshape(CLASSES, x.shape[1]); b = theta[CLASSES * x.shape[1]:]
    logits = x @ w.T + b
    logits -= logits.max(1, keepdims=True)
    exp = np.exp(logits); prob = exp / exp.sum(1, keepdims=True)
    loss = -np.log(prob[np.arange(len(y)), y]).mean()
    dw, db = w - w0, b - b0
    value = loss + 0.5 * lam * (np.sum(dw * dw) + np.sum(db * db))
    prob[np.arange(len(y)), y] -= 1.0
    grad_w = prob.T @ x / len(y) + lam * dw
    grad_b = prob.mean(0) + lam * db
    return float(value), np.concatenate([grad_w.ravel(), grad_b])


def fit_shrink(x: np.ndarray, y: np.ndarray, head, lam: float) -> tuple[np.ndarray, np.ndarray, dict]:
    x64 = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    w0 = np.asarray(head.coef_, dtype=np.float64); b0 = np.asarray(head.intercept_, dtype=np.float64)
    initial = np.concatenate([w0.ravel(), b0])
    start = time.perf_counter()
    result = minimize(shrink_objective, initial, args=(x64, y, w0, b0, lam), jac=True,
                      method="L-BFGS-B", options={"maxiter": 100, "maxls": 20, "ftol": 1e-9, "gtol": 1e-5})
    w = result.x[:CLASSES * x.shape[1]].reshape(CLASSES, x.shape[1]); b = result.x[CLASSES * x.shape[1]:]
    record = {"success": bool(result.success), "status": int(result.status), "message": str(result.message),
              "nit": int(result.nit), "nfev": int(result.nfev), "final_objective": float(result.fun),
              "fit_seconds": time.perf_counter() - start}
    return w, b, record


def pseudo_supports(indices: np.ndarray, labels: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    output = {}
    for seed in SEEDS:
        rng = np.random.Generator(np.random.PCG64(seed)); tens = {}
        for c in range(CLASSES):
            rows = np.sort(indices[labels == c]); tens[c] = rng.permutation(rows)[:10]
        for shot in SHOTS:
            output[(seed, shot)] = np.sort(np.concatenate([tens[c][:shot] for c in range(CLASSES)]))
    return output


def preflight() -> None:
    output = RUN / "artifacts/preflight.json"
    if sha256_file(RUN / "HISTORY_RETENTION_PLAN.md") != PLAN_HASH:
        raise ValueError("frozen plan hash mismatch")
    hashes, errors = {}, []
    for path, expected in FROZEN.items():
        actual = sha256_file(path); hashes[str(path)] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected: errors.append(f"hash mismatch: {path}")
    # Fully parse every specifically required structured donor artifact.
    summary = json.loads((DONOR / "artifacts/summary.json").read_text())
    audit = json.loads((DONOR / "artifacts/data_isolation_audit.json").read_text())
    manifest = json.loads((DONOR / "artifacts/support_manifests.json").read_text())
    integrity = json.loads((DONOR / "artifacts/integrity_check.json").read_text())
    with (DONOR / "artifacts/metrics_long.csv").open(newline="") as handle: metric_rows = list(csv.DictReader(handle))
    split = json.loads(SPLIT_PATH.read_text())
    if not integrity.get("passed") or integrity.get("errors") or not audit.get("passed") or audit.get("infeasible"):
        errors.append("donor integrity/isolation did not pass")
    if len(metric_rows) != 144 or summary.get("new_backbone_training_runs") != 0:
        errors.append("donor summary/metrics shape mismatch")
    if split.get("schema_version") != 3 or not split.get("all_102_classes_in_each_role"):
        errors.append("split schema/coverage mismatch")
    evaluated = 0
    for date in DATES:
        pool = [r["row_index"] for r in manifest["dates"][date]["eligible_rows"]]
        for seed in SEEDS:
            three = manifest["dates"][date]["configurations"][f"seed{seed}_shot3"]
            ten = manifest["dates"][date]["configurations"][f"seed{seed}_shot10"]
            if not set(three["support_rows"]) < set(ten["support_rows"]): errors.append(f"non-nested manifest {date}/{seed}")
            for shot, item in ((3, three), (10, ten)):
                support, query = item["support_rows"], item["query_rows"]
                if list_hash(support) != item["support_rows_sha256"] or list_hash(query) != item["query_rows_sha256"]:
                    errors.append(f"manifest list hash mismatch {date}/{seed}/{shot}")
                if set(query) != set(pool) - set(support): errors.append(f"manifest complement mismatch {date}/{seed}/{shot}")
                evaluated += 1
    donor_eval = integrity["evaluation_artifact_sha256"]
    for filename, expected in donor_eval.items():
        if sha256_file(DONOR / "artifacts" / filename) != expected: errors.append(f"donor evaluation hash mismatch: {filename}")
    source_shapes = {}
    for name in MODELS:
        record = json.loads((DONOR / "artifacts" / f"{name}_source_linear.json").read_text())
        head = source_head(name); source_shapes[name] = {"coef": list(head.coef_.shape), "intercept": list(head.intercept_.shape), "C": float(head.C)}
        if record["selected_C"] != 10.0 or record["new_backbone_training_runs"] != 0: errors.append(f"invalid source record: {name}")
        load_checkpoint(name)
    atomic_json(output, {"passed": not errors, "errors": errors, "frozen_hashes": hashes,
        "fully_parsed": {"summary_top_keys": sorted(summary), "metrics_rows": len(metric_rows),
        "manifest_configurations": evaluated, "eligible_rows": {d: len(manifest["dates"][d]["eligible_rows"]) for d in DATES},
        "isolation_dates": sorted(audit["dates"]), "integrity_evaluations": integrity["evaluation_artifacts_checked"],
        "split_role_rows": {r: len(v["indices"]) for r, v in split["roles"].items()}, "source_models": source_shapes},
        "new_backbone_training_runs": 0})
    if errors: raise SystemExit("; ".join(errors))
    print(output)


def select_source() -> None:
    out_path = RUN / "artifacts/source_selection.json"
    pre = json.loads((RUN / "artifacts/preflight.json").read_text())
    if not pre.get("passed"): raise ValueError("preflight not passed")
    split = json.loads(SPLIT_PATH.read_text()); sx, sy = load_npz(DATA / "train.npz"); vx, vy = load_npz(DATA / "valid.npz")
    train_idx = np.asarray(split["roles"]["supervised_train"]["indices"], dtype=np.int64)
    hold_idx = np.asarray(split["roles"]["source_holdout"]["indices"], dtype=np.int64)
    episodes = pseudo_supports(hold_idx, sy[hold_idx])
    alpha_records, lambda_records, extraction = [], [], {}
    device = torch.device("cpu")
    for name in MODELS:
        checkpoint, _ = load_checkpoint(name); model = make_model(name, checkpoint, device); head = source_head(name)
        train_z, _, train_rows, t1 = extract(model, sx, sy, train_idx, device)
        hold_z, _, hold_rows, t2 = extract(model, sx, sy, hold_idx, device)
        val_z, _, val_rows, t3 = extract(model, vx, vy, None, device)
        if not np.array_equal(train_rows, train_idx) or not np.array_equal(hold_rows, hold_idx): raise ValueError("source extraction order mismatch")
        extraction[name] = {"supervised_train": t1, "source_holdout": t2, "source_validation": t3, "device": "cpu"}
        source_proto = prototype(train_z, sy[train_rows]); np.save(RUN / "artifacts" / f"{name}_source_prototypes.npy", source_proto)
        hold_lookup = {int(row): i for i, row in enumerate(hold_rows)}
        for (seed, shot), support_rows in episodes.items():
            positions = np.asarray([hold_lookup[int(row)] for row in support_rows]); support_z = hold_z[positions]; support_y = sy[support_rows]
            current_proto = prototype(support_z, support_y)
            for alpha in ALPHAS:
                mixed = normalized((1-alpha)*source_proto + alpha*current_proto)
                pred = (val_z @ mixed.T).argmax(1)
                alpha_records.append({"model": name, "seed": seed, "shot": shot, "alpha": alpha,
                    "support_rows_sha256": list_hash(support_rows), "macro_f1": metrics(vy[val_rows], pred, False)["macro_f1"]})
            for lam in LAMBDAS:
                w, b, fit = fit_shrink(support_z, support_y, head, lam); pred = (val_z @ w.T + b).argmax(1)
                lambda_records.append({"model": name, "seed": seed, "shot": shot, "lambda": lam,
                    "support_rows_sha256": list_hash(support_rows), "macro_f1": metrics(vy[val_rows], pred, False)["macro_f1"], "fit": fit})
        del model
    alpha_summary = {str(a): mean_sd([r["macro_f1"] for r in alpha_records if r["alpha"] == a]) for a in ALPHAS}
    lambda_summary = {str(l): mean_sd([r["macro_f1"] for r in lambda_records if r["lambda"] == l]) for l in LAMBDAS}
    selected_alpha = max(ALPHAS, key=lambda a: (alpha_summary[str(a)]["mean"], -a))
    selected_lambda = max(LAMBDAS, key=lambda l: (lambda_summary[str(l)]["mean"], l))
    atomic_json(out_path, {"selection_data": "source_holdout pseudo-support; official source validation query",
        "query_dates_accessed": [], "shared_across_backbones_dates_shots_seeds": True,
        "alpha_candidates": list(ALPHAS), "lambda_candidates": list(LAMBDAS),
        "alpha_records": alpha_records, "lambda_records": lambda_records,
        "alpha_summary": alpha_summary, "lambda_summary": lambda_summary,
        "selected_alpha": selected_alpha, "selected_lambda": selected_lambda,
        "extraction_seconds": extraction, "new_backbone_training_runs": 0})
    print(json.dumps({"selected_alpha": selected_alpha, "selected_lambda": selected_lambda}, indent=2))


def evaluate(name: str, date: str) -> None:
    paths = [RUN / "artifacts" / f"{name}_{date}_seed{s}_shot{k}.json" for s in SEEDS for k in SHOTS]
    if any(p.exists() for p in paths): raise FileExistsError("refusing to overwrite formal evaluation")
    selection_path = RUN / "artifacts/source_selection.json"; selection = json.loads(selection_path.read_text())
    alpha, lam = float(selection["selected_alpha"]), float(selection["selected_lambda"])
    manifest = json.loads((DONOR / "artifacts/support_manifests.json").read_text())["dates"][date]
    checkpoint, checkpoint_path = load_checkpoint(name); device = torch.device("cpu")
    model = make_model(name, checkpoint, device); head = source_head(name); source_proto = np.load(RUN / "artifacts" / f"{name}_source_prototypes.npy")
    x, y = load_npz(DATA / f"{date}.npz"); z, _, rows, seconds = extract(model, x, y, None, device)
    if not np.array_equal(rows, np.arange(len(y))): raise ValueError("current extraction order mismatch")
    staged = []
    for seed in SEEDS:
        for shot in SHOTS:
            item = manifest["configurations"][f"seed{seed}_shot{shot}"]
            support = np.asarray(item["support_rows"], dtype=np.int64); query = np.asarray(item["query_rows"], dtype=np.int64)
            support_y = y[support]
            current_proto = prototype(z[support], support_y); mixed = normalized((1-alpha)*source_proto + alpha*current_proto)
            proto_scores = z[query] @ mixed.T; interp_pred = proto_scores.argmax(1).astype(np.int64)
            w, b, fit = fit_shrink(z[support], support_y, head, lam); shrink_scores = z[query] @ w.T + b
            shrink_pred = shrink_scores.argmax(1).astype(np.int64); g_scores = head.decision_function(z[query]); g_pred = g_scores.argmax(1).astype(np.int64)
            staged.append({"seed": seed, "shot": shot, "item": item, "support": support, "query": query,
                "new_predictions": {"prototype_interpolation": interp_pred, "shrink_to_source": shrink_pred},
                "g_pred": g_pred, "diagnostics": {"G_source_margin": top_margin(g_scores),
                    "prototype_interpolation_margin": top_margin(proto_scores), "prototype_interpolation_max_cosine": proto_scores.max(1),
                    "shrink_to_source_margin": top_margin(shrink_scores),
                    "source_current_prototype_cosine": np.sum(source_proto*current_proto, axis=1)}, "shrink_fit": fit})
    # Only now, after all new predictions for this backbone-date are fixed, access donor truth and score.
    integrity = json.loads((DONOR / "artifacts/integrity_check.json").read_text())
    for item in staged:
        seed, shot = item["seed"], item["shot"]
        donor_path = DONOR / "artifacts" / f"{name}_{date}_seed{seed}_shot{shot}.json"
        if sha256_file(donor_path) != integrity["evaluation_artifact_sha256"][donor_path.name]: raise ValueError("donor eval changed")
        donor = json.loads(donor_path.read_text()); query = item["query"]; truth = np.asarray(donor["query_truth"], dtype=np.int64)
        if donor["query_rows"] != query.tolist() or donor["support_rows"] != item["support"].tolist() or not np.array_equal(truth, y[query]):
            raise ValueError("donor/manifest/current truth mismatch")
        predictions = {"A": np.asarray(donor["predictions"]["A"]), "G_source": np.asarray(donor["predictions"]["G_source"]),
            "G_current": np.asarray(donor["predictions"]["G_current"]),
            "current_prototype": np.asarray(donor["predictions"]["simple_maintenance"]), **item["new_predictions"]}
        if not np.array_equal(predictions["G_source"], item["g_pred"]): raise ValueError("G-source recomputation mismatch")
        scored = {method: metrics(truth, pred) for method, pred in predictions.items()}
        transfers = {}
        base_ok = predictions["G_source"] == truth
        for method, pred in predictions.items():
            if method == "G_source": continue
            ok = pred == truth; corrected = (~base_ok) & ok; harmed = base_ok & (~ok)
            by_site = {str(c): {"corrected": int(corrected[truth == c].sum()), "harmed": int(harmed[truth == c].sum()),
                "net": int(corrected[truth == c].sum()-harmed[truth == c].sum())} for c in range(CLASSES)}
            transfers[method] = {"corrected": int(corrected.sum()), "harmed": int(harmed.sum()),
                "net": int(corrected.sum()-harmed.sum()), "corrected_rate": float(corrected.mean()),
                "harmed_rate": float(harmed.mean()), "net_rate": float((corrected.sum()-harmed.sum())/len(truth)), "per_website": by_site}
        diag = {k: v.tolist() for k, v in item["diagnostics"].items() if k != "source_current_prototype_cosine"}
        output = {"model": name, "date": date, "shot": shot, "seed": seed,
            "checkpoint": {"path": str(checkpoint_path), "sha256": CHECKPOINTS[name][1], "epoch": CHECKPOINTS[name][2]},
            "plan_sha256": PLAN_HASH, "source_selection_sha256": sha256_file(selection_path),
            "manifest": {k: item["item"][k] for k in ("support_rows_sha256","query_rows_sha256","support_content_hashes_sha256","query_content_hashes_sha256")},
            "support_rows": item["support"].tolist(), "query_rows": query.tolist(), "query_truth": truth.tolist(),
            "predictions": {k: v.tolist() for k, v in predictions.items()}, "metrics": scored, "transfers_vs_G_source": transfers,
            "prediction_time_diagnostics": diag,
            "source_current_prototype_cosine_by_class": item["diagnostics"]["source_current_prototype_cosine"].tolist(),
            "prototype_interpolation": {"alpha": alpha, "query_labels_used": False},
            "shrink_to_source": {"lambda": lam, "solver": "scipy_L-BFGS-B", "fit": item["shrink_fit"], "query_labels_used": False},
            "date_embedding_extraction_seconds": seconds, "feature_extraction_device": "cpu", "new_backbone_training_runs": 0}
        atomic_json(RUN / "artifacts" / f"{name}_{date}_seed{seed}_shot{shot}.json", output)
    print(f"{name}/{date}: 6 configurations")


def summarize() -> None:
    records = [json.loads((RUN / "artifacts" / f"{m}_{d}_seed{s}_shot{k}.json").read_text())
               for m in MODELS for d in DATES for s in SEEDS for k in SHOTS]
    methods = ("A", "G_source", "G_current", "current_prototype", "prototype_interpolation", "shrink_to_source")
    metric_rows, transfer_rows, site_rows = [], [], []
    for r in records:
        base = {k: r[k] for k in ("model","date","shot","seed")}
        for method in methods: metric_rows.append(base | {"method": method} | {k: r["metrics"][method][k] for k in ("accuracy","macro_precision","macro_recall","macro_f1")})
        for method, t in r["transfers_vs_G_source"].items():
            transfer_rows.append(base | {"method": method} | {k: t[k] for k in ("corrected","harmed","net","corrected_rate","harmed_rate","net_rate")})
            for site, v in t["per_website"].items(): site_rows.append(base | {"method": method, "website": int(site)} | v)
    for filename, rows in (("metrics_long.csv", metric_rows), ("error_transfers.csv", transfer_rows), ("per_website_transfers.csv", site_rows)):
        path = RUN / "artifacts" / filename
        if path.exists(): raise FileExistsError(path)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    grouped, transfer_grouped, shot_gains = {}, {}, {}
    for model in MODELS:
        grouped[model], transfer_grouped[model], shot_gains[model] = {}, {}, {}
        for date in DATES:
            grouped[model][date], transfer_grouped[model][date], shot_gains[model][date] = {}, {}, {}
            for shot in SHOTS:
                subset = [r for r in records if r["model"]==model and r["date"]==date and r["shot"]==shot]
                grouped[model][date][str(shot)] = {method: {metric: mean_sd([r["metrics"][method][metric] for r in subset]) for metric in ("accuracy","macro_f1")} for method in methods}
                transfer_grouped[model][date][str(shot)] = {method: {field: mean_sd([r["transfers_vs_G_source"][method][field] for r in subset]) for field in ("corrected","harmed","net")} for method in methods if method != "G_source"}
            for method in methods:
                shot_gains[model][date][method] = {metric: mean_sd([
                    next(r for r in records if r["model"]==model and r["date"]==date and r["seed"]==seed and r["shot"]==10)["metrics"][method][metric] -
                    next(r for r in records if r["model"]==model and r["date"]==date and r["seed"]==seed and r["shot"]==3)["metrics"][method][metric]
                    for seed in SEEDS]) for metric in ("accuracy","macro_f1")}
    judgments = {"day14_protection": {}, "late_recovery_retention": {}, "three_shot_bottleneck": {}}
    history = ("prototype_interpolation", "shrink_to_source")
    for method in history:
        protection, late, bottleneck = {}, {}, {}
        for model in MODELS:
            protect_units, late_units, gap_units = [], [], []
            for shot in SHOTS:
                cell = grouped[model]["day14"][str(shot)]; f = cell[method]["macro_f1"]["mean"]
                protect_units.append({"shot":shot,"vs_G_source":f-cell["G_source"]["macro_f1"]["mean"],
                    "vs_best_current":f-max(cell["G_current"]["macro_f1"]["mean"],cell["current_prototype"]["macro_f1"]["mean"])})
            protection[model] = {"units":protect_units,"passes":all(u["vs_G_source"]>=-.01 and u["vs_best_current"]>=.02 for u in protect_units)}
            for date in ("day90","day270"):
                for shot in SHOTS:
                    cell=grouped[model][date][str(shot)]; gs=cell["G_source"]["macro_f1"]["mean"]
                    recovery=max(cell["G_current"]["macro_f1"]["mean"],cell["current_prototype"]["macro_f1"]["mean"])-gs
                    hist=cell[method]["macro_f1"]["mean"]-gs
                    late_units.append({"date":date,"shot":shot,"current_recovery":recovery,"history_recovery":hist,
                        "retention": hist/recovery if recovery>0 else None})
            eligible=[u for u in late_units if u["retention"] is not None]
            late[model]={"units":late_units,"passes":bool(eligible) and all(u["retention"]>=.8 for u in eligible)}
            for date in DATES:
                best_gap=max(shot_gains[model][date]["G_current"]["macro_f1"]["mean"],shot_gains[model][date]["current_prototype"]["macro_f1"]["mean"])
                hist_gap=shot_gains[model][date][method]["macro_f1"]["mean"]
                f3=grouped[model][date]["3"][method]["macro_f1"]["mean"]-grouped[model][date]["3"]["G_source"]["macro_f1"]["mean"]
                gap_units.append({"date":date,"best_current_gap":best_gap,"history_gap":hist_gap,
                    "gap_reduction_fraction":(best_gap-hist_gap)/best_gap if best_gap>0 else None,"three_shot_vs_G_source":f3,
                    "qualifies":best_gap>0 and (best_gap-hist_gap)/best_gap>=.5 and f3>=-.01})
            bottleneck[model]={"units":gap_units,"passes":sum(u["qualifies"] for u in gap_units)>=2}
        judgments["day14_protection"][method]={"by_backbone":protection,"passes_both":all(v["passes"] for v in protection.values())}
        judgments["late_recovery_retention"][method]={"by_backbone":late,"passes_both":all(v["passes"] for v in late.values())}
        judgments["three_shot_bottleneck"][method]={"by_backbone":bottleneck,"passes_both":all(v["passes"] for v in bottleneck.values())}
        judgments.setdefault("simple_history_sufficient",{})[method] = judgments["day14_protection"][method]["passes_both"] and judgments["late_recovery_retention"][method]["passes_both"]
    repeat_sites = {}
    for method in history:
        repeat_sites[method] = []
        for site in range(CLASSES):
            model_date_harm = {}
            for model in MODELS:
                dates_harm=[]
                for date in DATES:
                    # aggregate both shots; a seed is harmful if summed net across shots is negative
                    harmful_seeds=sum(sum(next(row["net"] for row in site_rows if row["model"]==model and row["date"]==date and row["seed"]==seed and row["shot"]==shot and row["method"]==method and row["website"]==site) for shot in SHOTS)<0 for seed in SEEDS)
                    dates_harm.append(harmful_seeds>=2)
                model_date_harm[model]=dates_harm
            if all(sum(v)>=2 for v in model_date_harm.values()): repeat_sites[method].append({"website":site,"harmful_dates":{m:int(sum(v)) for m,v in model_date_harm.items()}})
    observable = {}
    for method in history:
        fields = ["G_source_margin", f"{method}_margin"]
        if method=="prototype_interpolation": fields.append("prototype_interpolation_max_cosine")
        observable[method]={}
        for field in fields:
            groups={"corrected":[],"harmed":[],"unchanged_error":[],"unchanged_correct":[]}
            for r in records:
                truth=np.asarray(r["query_truth"]); base=np.asarray(r["predictions"]["G_source"])==truth; ok=np.asarray(r["predictions"][method])==truth
                vals=np.asarray(r["prediction_time_diagnostics"][field])
                masks={"corrected":~base&ok,"harmed":base&~ok,"unchanged_error":~base&~ok,"unchanged_correct":base&ok}
                for key,mask in masks.items(): groups[key].extend(vals[mask].tolist())
            observable[method][field]={k:{"count":len(v),"mean":float(np.mean(v)) if v else None,"sd":float(np.std(v,ddof=1)) if len(v)>1 else None} for k,v in groups.items()}
    atomic_json(RUN/"artifacts/summary.json", {"grouped_seed_mean_sd":grouped,"transfer_seed_mean_sd":transfer_grouped,
        "paired_10shot_minus_3shot":shot_gains,"pre_registered_judgments":judgments,
        "repeated_harm_sites":repeat_sites,"observable_posthoc_groups":observable,
        "source_selection":json.loads((RUN/"artifacts/source_selection.json").read_text()),
        "rows":{"metrics_long":len(metric_rows),"error_transfers":len(transfer_rows),"per_website_transfers":len(site_rows)},
        "new_backbone_training_runs":0})
    print(RUN/"artifacts/summary.json")


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="stage",required=True)
    sub.add_parser("preflight"); sub.add_parser("select-source")
    ev=sub.add_parser("evaluate"); ev.add_argument("--model",choices=MODELS,required=True); ev.add_argument("--date",choices=DATES,required=True)
    sub.add_parser("summarize"); args=parser.parse_args()
    if args.stage=="preflight": preflight()
    elif args.stage=="select-source": select_source()
    elif args.stage=="evaluate": evaluate(args.model,args.date)
    else: summarize()


if __name__ == "__main__": main()
