#!/usr/bin/env python3
"""Frozen-checkpoint, Day0-only local error-correction diagnostic.

Stages are deliberately separated: ``prepare`` fits every local template and
fusion alpha using source-only data; only ``score`` opens future files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/exp_e0bcacc17e1b43d0"
DATA = Path("/mnt/data2/ren/datasets/TemporalDrift")
SPLIT = ROOT / "runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json"
BASE = ROOT / "runs/exp_9121b664a1854097"
CHECKPOINT = BASE / "checkpoints/df_best.pt"
sys.path.insert(0, str(ROOT / "src"))
from ta_wf_next.models import DF  # noqa: E402
from ta_wf_next.screening import sha256_file  # noqa: E402

DATES = ("day14", "day30", "day90", "day150", "day270")
FEATURES = (
    "run_mw_50_99",
    "run_mw_0_49",
    "run_mw_100_149",
    "direction_out_fraction_50_99",
    "direction_transition_density_50_99",
    "prefix150_out_fraction",
    "prefix150_transition_density",
)
PRIMARY = "run_mw_50_99"
UNSUPPORTED = (4, 8, 23, 43)
SUPPORTED = tuple(c for c in range(102) if c not in UNSUPPORTED)
SENSITIVITY = "run_mw_50_99__bypass_global_unsupported"
ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
EXPECTED_SPLIT = "0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142"
EXPECTED_CKPT = "1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae"


def refuse(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")


def atomic_json(path: Path, value) -> None:
    refuse(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    refuse(path)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        x, y = z["X"], z["y"].astype(np.int64)
    if x.ndim != 2 or x.shape[1] != 10000 or y.shape != (len(x),):
        raise ValueError(f"unexpected schema {path}: {x.shape}, {y.shape}")
    return x, y


def run_cover(direction: np.ndarray, lo: int, hi: int) -> float:
    starts = np.r_[0, np.flatnonzero(direction[1:] != direction[:-1]) + 1]
    ends = np.r_[starts[1:], len(direction)]
    lengths = ends - starts
    overlap = np.maximum(0, np.minimum(ends, hi) - np.maximum(starts, lo))
    if overlap.sum() != hi - lo:
        raise AssertionError("window is not fully covered by maximal runs")
    return float(np.sum(overlap * lengths) / overlap.sum())


def local_features(x: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    n = len(x)
    values = {name: np.full(n, np.nan, dtype=np.float64) for name in FEATURES}
    first = x[:, :500]
    eligible = np.isfinite(first).all(1) & (first != 0).all(1)
    failures = 0
    for i in np.flatnonzero(eligible):
        d = np.sign(first[i]).astype(np.int8)
        starts = np.r_[0, np.flatnonzero(d[1:] != d[:-1]) + 1]
        ends = np.r_[starts[1:], len(d)]
        runs = d[starts].astype(np.int32) * (ends - starts).astype(np.int32)
        decoded = np.repeat(np.sign(runs).astype(np.int8), np.abs(runs))
        if not (np.array_equal(decoded, d) and np.abs(runs).sum() == 500
                and np.all(np.sign(runs[1:]) != np.sign(runs[:-1]))):
            failures += 1
            continue
        values["run_mw_50_99"][i] = run_cover(d, 50, 100)
        values["run_mw_0_49"][i] = run_cover(d, 0, 50)
        values["run_mw_100_149"][i] = run_cover(d, 100, 150)
        main = d[50:100]
        prefix = d[:150]
        values["direction_out_fraction_50_99"][i] = np.mean(main > 0)
        values["direction_transition_density_50_99"][i] = np.mean(main[1:] != main[:-1])
        values["prefix150_out_fraction"][i] = np.mean(prefix > 0)
        values["prefix150_transition_density"][i] = np.mean(prefix[1:] != prefix[:-1])
    if failures:
        raise RuntimeError(f"RLE round-trip failures: {failures}")
    return values, eligible, int(eligible.sum())


class InferenceData(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return torch.from_numpy(np.sign(self.x[i, :5000]).astype(np.float32)[None, :]), i


def infer_logits(x: np.ndarray) -> tuple[np.ndarray, str, float]:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if state.get("model_name") != "df" or state.get("seed") != 6238 or state.get("epoch") != 29:
        raise ValueError("checkpoint metadata mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DF(102)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    # Match the formal exp_9121 evaluation batch size exactly. Two near-tied
    # Day14 rows changed argmax at batch 256 because of GPU numeric kernels.
    loader = DataLoader(InferenceData(x), batch_size=128, shuffle=False, num_workers=0)
    chunks, indices = [], []
    start = time.perf_counter()
    with torch.inference_mode():
        for xb, ib in loader:
            logits, _ = model(xb.to(device))
            chunks.append(logits.cpu().numpy().astype(np.float32))
            indices.extend(ib.tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if indices != list(range(len(x))):
        raise AssertionError("inference row order changed")
    return np.concatenate(chunks), str(device), elapsed


def fit_templates(features: dict[str, np.ndarray], labels: np.ndarray, indices: np.ndarray) -> dict:
    result = {}
    for name in FEATURES:
        medians, raw_scales, counts = {}, {}, []
        a = features[name]
        for c in range(102):
            vals = a[indices[labels[indices] == c]]
            vals = vals[np.isfinite(vals)]
            counts.append(int(len(vals)))
            if not len(vals):
                continue
            median = float(np.median(vals))
            medians[str(c)] = median
            raw_scales[str(c)] = float(1.4826 * np.median(np.abs(vals - median)))
        unsupported = tuple(c for c,n in enumerate(counts) if n == 0)
        if unsupported != UNSUPPORTED:
            raise ValueError(f"source support set changed for {name}: {unsupported}")
        positive = [v for v in raw_scales.values() if np.isfinite(v) and v > 0]
        floor = float(np.median(positive)) if positive else 1e-6
        scales = {str(c): max(raw_scales[str(c)], floor, 1e-6) for c in SUPPORTED}
        result[name] = {"median": medians, "raw_scale": raw_scales, "scale_floor": floor,
                        "scale": scales, "supported_classes": list(SUPPORTED),
                        "unsupported_classes": list(UNSUPPORTED), "train_eligible_per_class": counts}
    return result


def local_scores(values: np.ndarray, template: dict) -> np.ndarray:
    """Return additive local contributions; unsupported classes are exact zero."""
    supported = np.asarray(template["supported_classes"], dtype=np.int64)
    med = np.asarray([template["median"][str(c)] for c in supported])[None, :]
    scale = np.asarray([template["scale"][str(c)] for c in supported])[None, :]
    raw = -np.abs(values[:, None] - med) / scale
    calibrated = zclasses(raw)
    result = np.zeros((len(values), 102), dtype=np.float64)
    result[:, supported] = calibrated
    if not np.array_equal(result[:, np.asarray(UNSUPPORTED)], np.zeros((len(values), len(UNSUPPORTED)))):
        raise AssertionError("unsupported local contribution must be exact zero")
    return result


def zclasses(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(1, keepdims=True)
    sd = centered.std(1, keepdims=True)
    return np.divide(centered, sd, out=np.zeros_like(centered), where=sd >= 1e-12)


def fused_predictions(logits: np.ndarray, values: np.ndarray, template: dict, alpha: float) -> np.ndarray:
    pred = logits.argmax(1).astype(np.int64)
    ok = np.isfinite(values)
    if ok.any():
        local = local_scores(values[ok], template)
        fused = zclasses(logits[ok].astype(np.float64)) + alpha * local
        pred[ok] = fused.argmax(1)
    return pred


def ordinal_rank(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    true_score = scores[np.arange(len(scores)), truth]
    greater = (scores > true_score[:, None]).sum(1)
    class_ids = np.arange(scores.shape[1])[None, :]
    earlier_ties = ((scores == true_score[:, None]) & (class_ids < truth[:, None])).sum(1)
    return 1 + greater + earlier_ties


def prepare() -> None:
    if sha256_file(SPLIT) != EXPECTED_SPLIT or sha256_file(CHECKPOINT) != EXPECTED_CKPT:
        raise ValueError("formal split or checkpoint hash mismatch")
    split = json.loads(SPLIT.read_text())
    train_idx = np.asarray(split["roles"]["supervised_train"]["indices"], dtype=np.int64)
    x, y = load(DATA / "train.npz")
    train_features, _, train_eligible = local_features(x)
    templates = fit_templates(train_features, y, train_idx)
    del x, train_features
    vx, vy = load(DATA / "valid.npz")
    valid_features, _, valid_eligible = local_features(vx)
    logits, device, seconds = infer_logits(vx)
    logits_path = RUN / "artifacts/df_valid_logits.npz"
    refuse(logits_path)
    np.savez_compressed(logits_path, logits=logits, row_indices=np.arange(len(vy), dtype=np.int64))
    choices = {}
    global_accuracy = float(np.mean(logits.argmax(1) == vy))
    for name in FEATURES:
        candidates = []
        for alpha in ALPHAS:
            pred = fused_predictions(logits, valid_features[name], templates[name], alpha)
            candidates.append({"alpha": alpha, "accuracy": float(np.mean(pred == vy))})
        best = max(v["accuracy"] for v in candidates)
        chosen = min(v["alpha"] for v in candidates if v["accuracy"] == best)
        choices[name] = {"chosen_alpha": chosen, "criterion": "maximum official-valid accuracy; tie smaller alpha",
                         "candidates": candidates}
    params = {
        "schema_version": 1, "created_before_future_scoring": True,
        "split_sha256": EXPECTED_SPLIT, "checkpoint_sha256": EXPECTED_CKPT,
        "template_role": "v3 supervised_train", "template_count": len(train_idx),
        "train_file_rows_with_common_500_budget": train_eligible,
        "valid_rows": len(vy), "valid_rows_with_common_500_budget": valid_eligible,
        "valid_global_accuracy": global_accuracy, "inference_device": device,
        "valid_inference_seconds": seconds, "features": templates, "fusion": choices,
        "plan_version": 2, "plan_v2_sha256": "15a5db0df04c266e17d254f8ec681b0bbf2bb54cf7e1ef02f319ac82b34e1320",
        "supported_classes": list(SUPPORTED), "unsupported_classes": list(UNSUPPORTED),
        "fusion_form": "class-standardized global logits + alpha * local contribution; unsupported local contribution exactly zero",
        "future_files_opened": False,
    }
    atomic_json(RUN / "artifacts/frozen_day0_parameters.json", params)
    print(json.dumps({"valid_global_accuracy": global_accuracy, "alphas": {k:v["chosen_alpha"] for k,v in choices.items()},
                      "device": device, "seconds": seconds}, indent=2))


def agg_rank(date: str, subset: str, truth: np.ndarray, global_pred: np.ndarray,
             values: np.ndarray, template: dict, extra: np.ndarray | None = None) -> dict:
    base = np.ones(len(truth), dtype=bool)
    if subset == "global_wrong": base = global_pred != truth
    if subset == "global_correct": base = global_pred == truth
    if extra is not None: base &= extra
    eligible = base & np.isfinite(values)
    scores = local_scores(values[eligible], template) if eligible.any() else np.empty((0, 102))
    ranks = ordinal_rank(scores, truth[eligible]) if eligible.any() else np.empty(0)
    margin = (scores[np.arange(len(scores)), truth[eligible]] - scores[np.arange(len(scores)), global_pred[eligible]]) if eligible.any() else np.empty(0)
    return {"development_evidence": 1, "feature": PRIMARY, "date": date, "subset": subset,
            "n_total": int(base.sum()), "n_eligible": int(eligible.sum()),
            "eligible_fraction": float(eligible.sum()/max(base.sum(), 1)),
            "true_local_rank_mean": float(np.mean(ranks)) if len(ranks) else math.nan,
            "true_local_rank_median": float(np.median(ranks)) if len(ranks) else math.nan,
            "local_top1_hit": float(np.mean(ranks <= 1)) if len(ranks) else math.nan,
            "local_top5_hit": float(np.mean(ranks <= 5)) if len(ranks) else math.nan,
            "local_top10_hit": float(np.mean(ranks <= 10)) if len(ranks) else math.nan,
            "local_margin_true_minus_global_pred_mean": float(np.mean(margin)) if len(margin) else math.nan,
            "local_margin_true_minus_global_pred_median": float(np.median(margin)) if len(margin) else math.nan}


def fusion_row(feature: str, date: str, truth: np.ndarray, gp: np.ndarray, fp: np.ndarray,
               alpha: float, subset: str = "all", select: np.ndarray | None = None) -> dict:
    if select is not None:
        truth, gp, fp = truth[select], gp[select], fp[select]
    gc, fc = gp == truth, fp == truth
    w2c = int((~gc & fc).sum()); c2w = int((gc & ~fc).sum())
    wdiff = int((~gc & ~fc & (fp != gp)).sum())
    uc = int((gc & fc).sum()); uw = int((~gc & ~fc & (fp == gp)).sum())
    n = len(truth)
    return {"development_evidence": 1, "feature": feature, "date": date, "subset": subset, "alpha": alpha, "n": n,
            "wrong_to_correct": w2c, "correct_to_wrong": c2w, "wrong_to_different_wrong": wdiff,
            "unchanged_correct": uc, "unchanged_wrong": uw, "net_repair": w2c-c2w,
            "wrong_to_correct_over_global_wrong": float(w2c/max((~gc).sum(), 1)),
            "correct_to_wrong_over_global_correct": float(c2w/max(gc.sum(), 1)),
            "net_repair_over_all": float((w2c-c2w)/n), "global_accuracy": float(gc.mean()),
            "fused_accuracy": float(fc.mean()), "accuracy_delta": float(fc.mean()-gc.mean())}


def complement_rows(level: str, date: str, group_value: str, truth: np.ndarray, gp: np.ndarray,
                    logits: np.ndarray, values: np.ndarray, template: dict, select: np.ndarray) -> dict | None:
    take = select & (gp != truth) & np.isfinite(values)
    if not take.any(): return None
    ls = local_scores(values[take], template)
    lr = ordinal_rank(ls, truth[take]); gr = ordinal_rank(logits[take], truth[take])
    t, p = truth[take], gp[take]
    supports = ls[np.arange(len(ls)), t] > ls[np.arange(len(ls)), p]
    return {"development_evidence": 1, "level": level, "date": date, "group": group_value,
            "n_global_wrong_eligible": len(lr), "global_true_rank_mean": float(gr.mean()),
            "global_true_rank_median": float(np.median(gr)), "local_true_rank_mean": float(lr.mean()),
            "local_true_rank_median": float(np.median(lr)), "rank_improved": int((lr < gr).sum()),
            "rank_unchanged": int((lr == gr).sum()), "rank_worsened": int((lr > gr).sum()),
            "rank_improved_fraction": float(np.mean(lr < gr)), "pushed_into_top5": int(((gr > 5) & (lr <= 5)).sum()),
            "pushed_into_top10": int(((gr > 10) & (lr <= 10)).sum()),
            "local_supports_true_over_global_pred": int(supports.sum()),
            "local_supports_global_pred_over_true": int((~supports).sum())}


def score() -> None:
    params_path = RUN / "artifacts/frozen_day0_parameters.json"
    params = json.loads(params_path.read_text())
    if not params.get("created_before_future_scoring") or params.get("future_files_opened") is not False:
        raise ValueError("Day0 parameters are not sealed")
    templates, fusion = params["features"], params["fusion"]
    rank_rows, comp_rows, fusion_rows, site_rows = [], [], [], []
    records = []
    audit_dates = {}
    for date in DATES:
        x, y = load(DATA / f"{date}.npz")
        feats, eligible, eligible_n = local_features(x)
        logits, device, seconds = infer_logits(x)
        logits_path = RUN / "artifacts" / f"df_{date}_logits.npz"
        refuse(logits_path)
        np.savez_compressed(logits_path, logits=logits, row_indices=np.arange(len(y), dtype=np.int64))
        cache_path = BASE / "artifacts" / f"df_{date}_predictions.json"
        cache = json.loads(cache_path.read_text())
        gp = logits.argmax(1).astype(np.int64)
        checks = {
            "source_file": cache["source_file"] == f"{date}.npz",
            "row_indices": cache["row_indices"] == list(range(len(y))),
            "truth": cache["truth_scoring_only"] == y.tolist(),
            "argmax_A": cache["predictions"]["A"] == gp.tolist(),
        }
        if not all(checks.values()): raise AssertionError(f"prediction alignment failed {date}: {checks}")
        for subset in ("all", "global_correct", "global_wrong"):
            rank_rows.append(agg_rank(date, subset, y, gp, feats[PRIMARY], templates[PRIMARY]))
        rank_rows.append(agg_rank(date, "true_template_supported", y, gp, feats[PRIMARY], templates[PRIMARY], np.isin(y, SUPPORTED)))
        rank_rows.append(agg_rank(date, "true_template_unsupported", y, gp, feats[PRIMARY], templates[PRIMARY], np.isin(y, UNSUPPORTED)))
        rank_rows.append(agg_rank(date, "global_top1_supported", y, gp, feats[PRIMARY], templates[PRIMARY], np.isin(gp, SUPPORTED)))
        rank_rows.append(agg_rank(date, "global_top1_unsupported", y, gp, feats[PRIMARY], templates[PRIMARY], np.isin(gp, UNSUPPORTED)))
        base_select = np.ones(len(y), dtype=bool)
        row = complement_rows("date", date, "ALL", y, gp, logits, feats[PRIMARY], templates[PRIMARY], base_select)
        if row: comp_rows.append(row)
        for site in range(102):
            select = y == site
            row = complement_rows("site", date, str(site), y, gp, logits, feats[PRIMARY], templates[PRIMARY], select)
            if row: comp_rows.append(row)
        pairs = Counter(zip(y[gp != y].tolist(), gp[gp != y].tolist()))
        for (true_c, pred_c), count in sorted(pairs.items(), key=lambda z: (-z[1], z[0])):
            select = (y == true_c) & (gp == pred_c)
            row = complement_rows("confusion_pair", date, f"{true_c}->{pred_c}", y, gp, logits,
                                  feats[PRIMARY], templates[PRIMARY], select)
            if row: comp_rows.append(row)
        fused_by_feature = {}
        for name in FEATURES:
            alpha = float(fusion[name]["chosen_alpha"])
            fp = fused_predictions(logits, feats[name], templates[name], alpha)
            fused_by_feature[name] = fp
            fusion_rows.append(fusion_row(name, date, y, gp, fp, alpha))
        main_fp = fused_by_feature[PRIMARY]
        bypass = main_fp.copy()
        bypass[np.isin(gp, UNSUPPORTED)] = gp[np.isin(gp, UNSUPPORTED)]
        fused_by_feature[SENSITIVITY] = bypass
        fusion_rows.append(fusion_row(SENSITIVITY, date, y, gp, bypass,
                                      float(fusion[PRIMARY]["chosen_alpha"])))
        fusion_rows.append(fusion_row(PRIMARY, date, y, gp, main_fp, float(fusion[PRIMARY]["chosen_alpha"]),
                                      "true_template_supported", np.isin(y, SUPPORTED)))
        fusion_rows.append(fusion_row(PRIMARY, date, y, gp, main_fp, float(fusion[PRIMARY]["chosen_alpha"]),
                                      "true_template_unsupported", np.isin(y, UNSUPPORTED)))
        fusion_rows.append(fusion_row(PRIMARY, date, y, gp, main_fp, float(fusion[PRIMARY]["chosen_alpha"]),
                                      "global_top1_supported", np.isin(gp, SUPPORTED)))
        fusion_rows.append(fusion_row(PRIMARY, date, y, gp, main_fp, float(fusion[PRIMARY]["chosen_alpha"]),
                                      "global_top1_unsupported", np.isin(gp, UNSUPPORTED)))
        main_scores = np.full((len(y), 102), np.nan)
        if eligible.any(): main_scores[eligible] = local_scores(feats[PRIMARY][eligible], templates[PRIMARY])
        gr_all = ordinal_rank(logits, y)
        lr_all = np.full(len(y), np.nan)
        if eligible.any(): lr_all[eligible] = ordinal_rank(main_scores[eligible], y[eligible])
        for site in range(102):
            take = y == site; ew = take & eligible & (gp != y)
            gc, fc = gp[take] == y[take], main_fp[take] == y[take]
            site_rows.append({"development_evidence": 1, "date": date, "site": site, "n": int(take.sum()),
                "local_eligible": int((take & eligible).sum()), "global_wrong": int((take & (gp != y)).sum()),
                "global_wrong_local_eligible": int(ew.sum()),
                "global_true_rank_mean_on_eligible_wrong": float(np.mean(gr_all[ew])) if ew.any() else math.nan,
                "local_true_rank_mean_on_eligible_wrong": float(np.mean(lr_all[ew])) if ew.any() else math.nan,
                "rank_improved": int(np.sum(lr_all[ew] < gr_all[ew])) if ew.any() else 0,
                "rank_worsened": int(np.sum(lr_all[ew] > gr_all[ew])) if ew.any() else 0,
                "wrong_to_correct": int((~gc & fc).sum()), "correct_to_wrong": int((gc & ~fc).sum()),
                "net_repair": int((~gc & fc).sum() - (gc & ~fc).sum()),
                "global_accuracy": float(gc.mean()), "fused_accuracy": float(fc.mean()),
                "accuracy_delta": float(fc.mean()-gc.mean())})
        records.append({"date": date, "y": y, "gp": gp, "logits": logits, "features": feats,
                        "fused": fused_by_feature, "eligible": eligible})
        audit_dates[date] = {"rows": len(y), "eligible": eligible_n, "inference_device": device,
                             "inference_seconds": seconds, "cache_sha256": sha256_file(cache_path),
                             "logits_sha256": sha256_file(logits_path), "alignment": checks,
                             "global_accuracy": float(np.mean(gp == y))}
        del x
        print(date, audit_dates[date], flush=True)

    all_y = np.concatenate([r["y"] for r in records]); all_gp = np.concatenate([r["gp"] for r in records])
    all_logits = np.concatenate([r["logits"] for r in records])
    all_values = np.concatenate([r["features"][PRIMARY] for r in records])
    for subset in ("all", "global_correct", "global_wrong"):
        rank_rows.append(agg_rank("ALL", subset, all_y, all_gp, all_values, templates[PRIMARY]))
    rank_rows.append(agg_rank("ALL", "true_template_supported", all_y, all_gp, all_values, templates[PRIMARY], np.isin(all_y, SUPPORTED)))
    rank_rows.append(agg_rank("ALL", "true_template_unsupported", all_y, all_gp, all_values, templates[PRIMARY], np.isin(all_y, UNSUPPORTED)))
    rank_rows.append(agg_rank("ALL", "global_top1_supported", all_y, all_gp, all_values, templates[PRIMARY], np.isin(all_gp, SUPPORTED)))
    rank_rows.append(agg_rank("ALL", "global_top1_unsupported", all_y, all_gp, all_values, templates[PRIMARY], np.isin(all_gp, UNSUPPORTED)))
    row = complement_rows("overall", "ALL", "ALL", all_y, all_gp, all_logits, all_values,
                          templates[PRIMARY], np.ones(len(all_y), dtype=bool))
    if row: comp_rows.append(row)
    for name in FEATURES:
        all_fp = np.concatenate([r["fused"][name] for r in records])
        fusion_rows.append(fusion_row(name, "ALL", all_y, all_gp, all_fp,
                                      float(fusion[name]["chosen_alpha"])))
    all_main_fp = np.concatenate([r["fused"][PRIMARY] for r in records])
    all_bypass = np.concatenate([r["fused"][SENSITIVITY] for r in records])
    fusion_rows.append(fusion_row(SENSITIVITY, "ALL", all_y, all_gp, all_bypass,
                                  float(fusion[PRIMARY]["chosen_alpha"])))
    fusion_rows.append(fusion_row(PRIMARY, "ALL", all_y, all_gp, all_main_fp, float(fusion[PRIMARY]["chosen_alpha"]),
                                  "true_template_supported", np.isin(all_y, SUPPORTED)))
    fusion_rows.append(fusion_row(PRIMARY, "ALL", all_y, all_gp, all_main_fp, float(fusion[PRIMARY]["chosen_alpha"]),
                                  "true_template_unsupported", np.isin(all_y, UNSUPPORTED)))
    fusion_rows.append(fusion_row(PRIMARY, "ALL", all_y, all_gp, all_main_fp, float(fusion[PRIMARY]["chosen_alpha"]),
                                  "global_top1_supported", np.isin(all_gp, SUPPORTED)))
    fusion_rows.append(fusion_row(PRIMARY, "ALL", all_y, all_gp, all_main_fp, float(fusion[PRIMARY]["chosen_alpha"]),
                                  "global_top1_unsupported", np.isin(all_gp, UNSUPPORTED)))

    write_csv(RUN / "local_rank_diagnostic.csv", rank_rows)
    write_csv(RUN / "error_complementarity.csv", comp_rows)
    write_csv(RUN / "fusion_repair_table.csv", fusion_rows)
    write_csv(RUN / "per_site_date_breakdown.csv", site_rows)
    main_all = next(r for r in fusion_rows if r["feature"] == PRIMARY and r["date"] == "ALL" and r["subset"] == "all")
    controls = []
    for r in fusion_rows:
        if r["date"] not in (*DATES, "ALL") or r["subset"] != "all": continue
        controls.append({"development_evidence": 1, "feature": r["feature"],
            "feature_role": ("primary" if r["feature"] == PRIMARY else
                             "label_free_sensitivity" if r["feature"] == SENSITIVITY else "specificity_control"),
            "date": r["date"], "alpha": r["alpha"], "net_repair": r["net_repair"],
            "accuracy_delta": r["accuracy_delta"],
            "net_repair_minus_primary": r["net_repair"]-main_all["net_repair"] if r["date"] == "ALL" else "",
            "fraction_of_primary_net_repair": (r["net_repair"]/main_all["net_repair"]
                if r["date"] == "ALL" and main_all["net_repair"] != 0 else "")})
    write_csv(RUN / "specificity_controls.csv", controls)

    main_dates = [r for r in fusion_rows if r["feature"] == PRIMARY and r["date"] in DATES and r["subset"] == "all"]
    positive_dates = sum(r["net_repair"] > 0 for r in main_dates)
    per_site_net = Counter()
    for r in site_rows: per_site_net[r["site"]] += r["net_repair"]
    positive_sites = sum(v > 0 for v in per_site_net.values())
    overall_comp = next(r for r in comp_rows if r["level"] == "overall")
    control_all = [r for r in fusion_rows if r["feature"] in FEATURES and r["feature"] != PRIMARY and r["date"] == "ALL" and r["subset"] == "all"]
    best_control = max(control_all, key=lambda r: r["net_repair"])
    unsupported_all = next(r for r in fusion_rows if r["feature"] == PRIMARY and r["date"] == "ALL" and r["subset"] == "true_template_unsupported")
    bypass_all = next(r for r in fusion_rows if r["feature"] == SENSITIVITY and r["date"] == "ALL" and r["subset"] == "all")
    specificity_gap = main_all["net_repair"] - best_control["net_repair"]
    required_gap = max(10, math.ceil(0.2 * abs(main_all["net_repair"])))
    rank_gate = overall_comp["rank_improved"] > overall_comp["rank_worsened"]
    specific = all(main_all["net_repair"] > r["net_repair"] for r in control_all) and specificity_gap >= required_gap
    if main_all["net_repair"] > 0 and positive_dates >= 4 and positive_sites >= 20 and rank_gate and specific:
        verdict = "SUPPORTS_CONFLICT_RELIABILITY_FOLLOWUP"
    elif main_all["net_repair"] > 0 and positive_dates >= 3 and (not specific or best_control["net_repair"] >= 0.8*main_all["net_repair"]):
        verdict = "EARLY_TRAFFIC_ONLY"
    else:
        verdict = "STOP_LOCAL_CORRECTION_ROUTE"
    audit = {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": EXPECTED_CKPT,
             "checkpoint_metadata": {"model": "df", "seed": 6238, "epoch": 29},
             "split": str(SPLIT), "split_sha256": EXPECTED_SPLIT, "input": "sign of first 5000 raw positions",
             "prediction_source": "fresh inference of frozen checkpoint; no optimizer/backward/update",
             "formal_cache": str(BASE / "artifacts/df_<date>_predictions.json"), "dates": audit_dates,
             "all_alignment_checks_passed": all(all(v for v in d["alignment"].values()) for d in audit_dates.values())}
    atomic_json(RUN / "artifacts/global_prediction_audit.json", audit)
    summary = {"schema_version": 1, "experiment": RUN.name, "status": "completed", "verdict": verdict,
        "scope": "TemporalDrift development evidence; no training or adaptation",
        "primary_feature": PRIMARY, "chosen_alpha": fusion[PRIMARY]["chosen_alpha"],
        "global_accuracy": main_all["global_accuracy"], "fused_accuracy": main_all["fused_accuracy"],
        "accuracy_delta": main_all["accuracy_delta"], "wrong_to_correct": main_all["wrong_to_correct"],
        "correct_to_wrong": main_all["correct_to_wrong"], "wrong_to_different_wrong": main_all["wrong_to_different_wrong"],
        "net_repair": main_all["net_repair"], "positive_net_repair_dates": positive_dates,
        "positive_net_repair_sites": positive_sites, "rank_improved": overall_comp["rank_improved"],
        "rank_worsened": overall_comp["rank_worsened"], "best_control": best_control["feature"],
        "best_control_net_repair": best_control["net_repair"], "specificity_gap": specificity_gap,
        "unsupported_true_class": {"n": unsupported_all["n"], "wrong_to_correct": unsupported_all["wrong_to_correct"],
            "correct_to_wrong": unsupported_all["correct_to_wrong"], "global_correct": unsupported_all["unchanged_correct"] + unsupported_all["correct_to_wrong"],
            "net_repair": unsupported_all["net_repair"]},
        "bypass_global_unsupported_sensitivity": {"net_repair": bypass_all["net_repair"],
            "accuracy_delta": bypass_all["accuracy_delta"]},
        "required_specificity_gap": required_gap, "gates": {"rank_gate": rank_gate, "specificity_gate": specific},
        "permissions": {"training_runs": 0, "adaptation_runs": 0, "future_labels": "post-prediction development scoring only"}}
    atomic_json(RUN / "summary.json", summary)
    render_reports(params, audit, summary, fusion_rows, rank_rows, overall_comp, best_control)
    print(json.dumps(summary, indent=2))


def render_reports(params, audit, summary, fusion_rows, rank_rows, overall_comp, best_control) -> None:
    audit_md = RUN / "global_prediction_audit.md"; refuse(audit_md)
    lines = ["# Global prediction audit", "", f"- Checkpoint: `{audit['checkpoint']}`",
        f"- SHA-256: `{audit['checkpoint_sha256']}`; DF seed 6238, epoch 29.",
        f"- Split: `{audit['split']}`; SHA-256 `{audit['split_sha256']}`.",
        "- Input: sign of raw positions 0..4999. Frozen eval/inference only; no optimizer, backward, parameter update, adaptation or batch-label access.",
        "- Existing formal cache stores only top-1; fresh float32 logits were materialized in this run for rank/fusion. Every date matched cache source file, row IDs, truth array and A argmax exactly.", "",
        "| date | rows | local eligible | global accuracy | device | inference seconds |", "|---|---:|---:|---:|---|---:|"]
    for d, a in audit["dates"].items(): lines.append(f"| {d} | {a['rows']} | {a['eligible']} | {a['global_accuracy']:.6f} | {a['inference_device']} | {a['inference_seconds']:.3f} |")
    audit_md.write_text("\n".join(lines) + "\n")

    local_md = RUN / "local_scoring_definition.md"; refuse(local_md)
    alpha_lines = [f"| {name} | {params['features'][name]['scale_floor']:.8g} | {params['fusion'][name]['chosen_alpha']:.2f} | {max(x['accuracy'] for x in params['fusion'][name]['candidates']):.6f} |" for name in FEATURES]
    local_md.write_text("""# Local scoring definition

All definitions were frozen before future scoring. A trace is local-eligible only when raw positions 0..499 are finite and nonzero; otherwise local metrics are missing and fusion remains global. Exact signed maximal-run RLE of these 500 directions passed decode round-trip. The final observed run is retained exactly as in the prior primary definition and may be right-censored.

The primary scalar is the overlap-mass-weighted covering-run length on zero-based raw positions 50..99: `sum(overlap_packets * full_observed_run_length) / 50`. For each supported class, v3 supervised-train supplies the median and `1.4826*MAD`; the scale is floored by the median positive class scale and `1e-6`. Raw class score is `-|x-median_c|/scale_c`.

Classes 4, 8, 23 and 43 have no Day0 trace under the frozen 500-packet budget. No template is fabricated for them. Supported raw scores are class-standardized within the 98 supported classes; each unsupported class receives exactly zero local contribution. Fusion uses `Gz + alpha*L`, so unsupported classes retain their standardized global score. A local-ineligible trace receives all-zero local contribution and remains exactly global.

Alpha is selected only on official Day0 valid all-class accuracy from `[0,.05,.10,.20,.30,.50,.75,1]`, ties to smaller alpha. No margin gate, temperature, future batch statistic or future label is used. The source-defined bypass sensitivity disables fusion when frozen global top-1 is unsupported; it is not the primary rule.

Incoming fraction is `1-outgoing_fraction` and is not duplicated as a separate candidate. The two 50-packet neighboring run controls, the same-window direction controls and the prefix-150 direction controls use the identical template and fusion protocol.

| feature | pooled scale floor | chosen alpha | best Day0-valid accuracy |
|---|---:|---:|---:|
""" + "\n".join(alpha_lines) + "\n")

    main_dates = [r for r in fusion_rows if r["feature"] == PRIMARY and r["date"] in DATES and r["subset"] == "all"]
    result = RUN / "RESULTS.md"
    result.write_text(f"""# Local Burst Evidence Error-Correction Diagnostic

## Technical verdict: {summary['verdict']}

This is a frozen-rule TemporalDrift development diagnostic, not an external confirmation and not a Host research-route decision. No backbone or local model was trained, fine-tuned or adapted.

The formal DF source-only head had aggregate future accuracy {summary['global_accuracy']:.6f}. The Day0-only static fusion selected alpha {summary['chosen_alpha']:.2f} for the frozen 50–99 run-length scalar and reached {summary['fused_accuracy']:.6f} (delta {summary['accuracy_delta']:+.6f}). It produced {summary['wrong_to_correct']} wrong→correct, {summary['correct_to_wrong']} correct→wrong, {summary['wrong_to_different_wrong']} wrong→different-wrong and Net Repair {summary['net_repair']}. Positive Net Repair occurred on {summary['positive_net_repair_dates']}/5 dates and {summary['positive_net_repair_sites']} sites after aggregating dates.

Classes 4, 8, 23 and 43 had no Day0 template under the frozen common 500-packet budget and received exactly zero additive local contribution. Across {summary['unsupported_true_class']['n']} future samples whose true class was unsupported, fusion caused {summary['unsupported_true_class']['correct_to_wrong']} correct→wrong transitions out of {summary['unsupported_true_class']['global_correct']} originally correct predictions, repaired {summary['unsupported_true_class']['wrong_to_correct']}, and had Net Repair {summary['unsupported_true_class']['net_repair']}; all damage counts fully against the all-class result. The label-free global-unsupported bypass sensitivity had Net Repair {summary['bypass_global_unsupported_sensitivity']['net_repair']} and delta {summary['bypass_global_unsupported_sensitivity']['accuracy_delta']:+.6f}; it is not the primary rule.

On eligible global errors, true-class local rank improved for {summary['rank_improved']} samples and worsened for {summary['rank_worsened']}. The strongest specificity control was `{summary['best_control']}` with aggregate Net Repair {summary['best_control_net_repair']}; the primary-minus-control gap was {summary['specificity_gap']} against the preregistered required gap {summary['required_specificity_gap']}.

| date | wrong→correct | correct→wrong | wrong→different-wrong | Net Repair | global acc | fused acc | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {r['date']} | {r['wrong_to_correct']} | {r['correct_to_wrong']} | {r['wrong_to_different_wrong']} | {r['net_repair']} | {r['global_accuracy']:.6f} | {r['fused_accuracy']:.6f} | {r['accuracy_delta']:+.6f} |" for r in main_dates) + f"""

The verdict follows only the frozen gates in `CORRECTION_PLAN_v1.md`. Even a positive follow-up verdict would mean only that conflict reliability deserves study; it would not establish a new adaptation method. Future labels were used only after prediction for the reported development metrics and attribution. No selector, TTA or follow-on training Job was created.

Detailed ranks, confusion-pair attribution, all transition counts, specificity controls and per-site/date results are retained in the required CSV files. A limitation is that local evidence is available only to traces with the common 500-packet budget; ineligible samples retain the global decision. Scalar class templates cannot represent multimodal within-class local structure, by design of this simple diagnostic.
""")


def verify() -> None:
    required = ["CORRECTION_PLAN_v1.md", "global_prediction_audit.md", "local_scoring_definition.md",
        "local_rank_diagnostic.csv", "error_complementarity.csv", "fusion_repair_table.csv",
        "specificity_controls.csv", "per_site_date_breakdown.csv", "RESULTS.md", "summary.json"]
    missing = [name for name in required if not (RUN/name).is_file()]
    if missing: raise FileNotFoundError(missing)
    summary = json.loads((RUN/"summary.json").read_text())
    if summary["permissions"]["training_runs"] or summary["permissions"]["adaptation_runs"]:
        raise AssertionError("forbidden run count")
    with (RUN/"fusion_repair_table.csv").open() as f: rows = list(csv.DictReader(f))
    for r in rows:
        if int(r["net_repair"]) != int(r["wrong_to_correct"]) - int(r["correct_to_wrong"]):
            raise AssertionError("net repair arithmetic")
        if not math.isclose(float(r["accuracy_delta"]), int(r["net_repair"])/int(r["n"]), abs_tol=1e-12):
            raise AssertionError("accuracy delta arithmetic")
    print(json.dumps({"required_files": len(required), "fusion_rows": len(rows), "verdict": summary["verdict"], "pass": True}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "score", "verify"])
    args = parser.parse_args()
    {"prepare": prepare, "score": score, "verify": verify}[args.stage]()


if __name__ == "__main__":
    main()
