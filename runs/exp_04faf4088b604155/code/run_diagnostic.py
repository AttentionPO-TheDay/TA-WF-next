#!/usr/bin/env python3
"""Frozen 3-shot estimation-noise versus historical-coverage diagnostic.

The program deliberately separates prediction/geometry freezing from query-label
scoring. There is no optimizer, backward pass, or checkpoint write path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from ta_wf_next.models import DF, VarCNNDirection
from ta_wf_next.screening import TraceDataset, load_npz, sha256_file

RUN = ROOT / "runs/exp_04faf4088b604155"
ART = RUN / "artifacts"
DONOR = ROOT / "runs/exp_b471517a3e6f41e7"
DONOR_ART = DONOR / "artifacts"
SPLIT = ROOT / "runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json"
CKPT_RUN = ROOT / "runs/exp_9121b664a1854097"
DATA = Path("/mnt/data2/ren/datasets/TemporalDrift")
MODELS = ("df", "varcnn_direction")
DATES = ("day14", "day90", "day270")
SEEDS = (1729, 6238, 20260916)
CLASSES = 102
COMBOS = np.asarray(list(itertools.combinations(range(10), 3)), dtype=np.int64)
CHECKPOINTS = {
    "df": ("df_best.pt", "1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae", 29),
    "varcnn_direction": ("varcnn_direction_best.pt", "fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83", 23),
}


def refuse(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")


def atomic_json(path: Path, value: object) -> None:
    refuse(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    refuse(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def hash_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def normalized(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, norms, out=np.zeros_like(a), where=norms != 0)


def unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0:
        raise ValueError("zero center")
    return (v / norm).astype(np.float32)


def make_model(name: str, device: torch.device):
    filename, expected, epoch = CHECKPOINTS[name]
    path = CKPT_RUN / "checkpoints" / filename
    if sha256_file(path) != expected:
        raise ValueError(f"checkpoint hash mismatch {name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != name or checkpoint.get("seed") != 6238 or checkpoint.get("epoch") != epoch:
        raise ValueError(f"checkpoint metadata mismatch {name}")
    model = DF(CLASSES) if name == "df" else VarCNNDirection(CLASSES)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def extract_embeddings(model, x, y, indices, device):
    loader = DataLoader(TraceDataset(x, y, indices), batch_size=128, shuffle=False, num_workers=0)
    features, rows = [], []
    with torch.inference_mode():
        for xb, _, rid in loader:
            _, feat = model(xb.to(device))
            if feat.shape[1:] != (512,):
                raise ValueError(f"bad embedding {feat.shape}")
            features.append(feat.cpu().numpy().astype(np.float32, copy=False))
            rows.append(rid.numpy())
    return normalized(np.concatenate(features)), np.concatenate(rows).astype(np.int64)


def preflight() -> None:
    output = ART / "input_manifest_v1.json"
    refuse(output)
    plan = RUN / "DIAGNOSTIC_PLAN_v1.md"
    split = json.loads(SPLIT.read_text())
    donor_integrity = json.loads((DONOR_ART / "integrity_check.json").read_text())
    if not donor_integrity.get("passed") or len(donor_integrity["evaluation_artifact_sha256"]) != 36:
        raise ValueError("donor integrity is not passed/complete")
    fixed = [
        plan, RUN / "config.json", DONOR / "RECOVERABILITY_PLAN.md", DONOR / "RESULTS.md",
        DONOR / "execution_record.md", DONOR_ART / "summary.json", DONOR_ART / "metrics_long.csv",
        DONOR_ART / "support_manifests.json", DONOR_ART / "data_isolation_audit.json",
        DONOR_ART / "integrity_check.json", DONOR / "code/run_recoverability.py",
        ROOT / "src/ta_wf_next/models/df.py", ROOT / "src/ta_wf_next/models/varcnn.py",
        ROOT / "src/ta_wf_next/screening.py", SPLIT,
        ROOT / "runs/exp_a2ffb5623ad346c7/RESULTS.md",
        ROOT / "runs/exp_a2ffb5623ad346c7/artifacts/summary.json",
        ROOT / "runs/exp_a2ffb5623ad346c7/artifacts/integrity_check.json",
    ]
    files = {str(path): sha256_file(path) for path in fixed}
    for name, (filename, expected, _) in CHECKPOINTS.items():
        path = CKPT_RUN / "checkpoints" / filename
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"checkpoint mismatch {name}")
        files[str(path)] = actual
    datasets = {}
    for filename in ("train.npz", "day14.npz", "day90.npz", "day270.npz"):
        path = DATA / filename
        actual = sha256_file(path)
        expected = split["source_files_sha256"][filename]
        if actual != expected:
            raise ValueError(f"dataset mismatch {filename}")
        datasets[str(path)] = actual
    evals = {}
    for filename, expected in donor_integrity["evaluation_artifact_sha256"].items():
        path = DONOR_ART / filename
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"donor evaluation mismatch {filename}")
        evals[str(path)] = actual
    manifest = json.loads((DONOR_ART / "support_manifests.json").read_text())
    if manifest.get("seeds") != list(SEEDS) or manifest.get("shots") != [3, 10]:
        raise ValueError("unexpected donor manifest")
    combos = {
        "ordering": "formal 3 rows ascending then added 7 rows ascending within each class",
        "count": int(len(COMBOS)), "combinations": COMBOS.tolist(), "sha256": hash_array(COMBOS.astype("<i8")),
    }
    combo_path = ART / "resampling_manifest_v1.json"
    atomic_json(combo_path, combos)
    atomic_json(output, {
        "schema_version": 1, "passed": True, "created_before_current_array_load": True,
        "plan_sha256": files[str(plan)], "files": files, "datasets": datasets,
        "donor_evaluations": evals, "resampling_manifest": {"path": str(combo_path), "sha256": sha256_file(combo_path)},
        "checkpoint_training_seed": 6238, "new_backbone_training_runs": 0,
        "embedding": {"location": "forward_features output / mlp input", "dimension": 512, "normalization": "row_l2"},
    })
    print(json.dumps({"passed": True, "files": len(files), "evaluations": len(evals), "datasets": len(datasets)}))


def source_geometry(z: np.ndarray, y: np.ndarray, rows: np.ndarray):
    thresholds = np.zeros(CLASSES, np.float32)
    centers = np.zeros((CLASSES, 512), np.float32)
    exemplars = np.zeros((CLASSES, 3, 512), np.float32)
    exemplar_rows = np.zeros((CLASSES, 3), np.int64)
    nn_summary = []
    for label in range(CLASSES):
        mask = y == label
        v, r = z[mask], rows[mask]
        if len(v) < 3 or np.any(np.linalg.norm(v, axis=1) == 0):
            raise ValueError(f"insufficient/zero source class {label}")
        sim = v @ v.T
        np.fill_diagonal(sim, -np.inf)
        distances = 1.0 - sim.max(1)
        thresholds[label] = np.quantile(distances, .95, method="linear")
        centers[label] = unit(v.mean(0))
        first = int(np.argmax(v @ centers[label]))
        selected = [first]
        while len(selected) < 3:
            max_sim = (v @ v[selected].T).max(1)
            max_sim[selected] = np.inf
            candidate_values = np.where(max_sim == max_sim.min())[0]
            selected.append(int(candidate_values[np.argmin(r[candidate_values])]))
        exemplars[label] = v[selected]
        exemplar_rows[label] = r[selected]
        nn_summary.append({"site": label, "source_rows": len(v), "threshold_q95": float(thresholds[label]),
                           "nn_mean": float(distances.mean()), "nn_median": float(np.median(distances)),
                           "exemplar_rows": r[selected].tolist()})
    return thresholds, centers, exemplars, exemplar_rows, nn_summary


def extract_stage(name: str) -> None:
    if not (ART / "input_manifest_v1.json").is_file():
        raise ValueError("preflight missing")
    source_path = ART / f"{name}_source_embeddings_v1.npz"
    geom_path = ART / f"{name}_source_geometry_v1.npz"
    geom_json = ART / f"{name}_source_geometry_v1.json"
    for path in (source_path, geom_path, geom_json): refuse(path)
    for date in DATES: refuse(ART / f"{name}_{date}_embeddings_v1.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(name, device)
    split = json.loads(SPLIT.read_text())
    indices = np.asarray(split["roles"]["supervised_train"]["indices"], np.int64)
    sx, sy = load_npz(DATA / "train.npz")
    start = time.perf_counter(); sz, rows = extract_embeddings(model, sx, sy, indices, device); source_seconds = time.perf_counter()-start
    labels = sy[rows]
    atomic_npz(source_path, embeddings=sz, rows=rows, labels=labels)
    thresholds, centers, exemplars, exemplar_rows, summary = source_geometry(sz, labels, rows)
    atomic_npz(geom_path, thresholds=thresholds, centers=centers, exemplars=exemplars, exemplar_rows=exemplar_rows)
    atomic_json(geom_json, {"model": name, "source_role": "supervised_train", "threshold_rule": "q95 leave-one-out same-class nearest cosine distance",
                            "quantile_method": "linear", "sites": summary, "source_cache_sha256": sha256_file(source_path)})
    del sx, sy, sz
    dates = {}
    for date in DATES:
        x, y = load_npz(DATA / f"{date}.npz")
        start = time.perf_counter(); z, current_rows = extract_embeddings(model, x, y, None, device); elapsed = time.perf_counter()-start
        if not np.array_equal(current_rows, np.arange(len(y))): raise ValueError("current row order")
        path = ART / f"{name}_{date}_embeddings_v1.npy"
        np.save(path, z, allow_pickle=False)
        dates[date] = {"rows": len(z), "seconds": elapsed, "sha256": sha256_file(path)}
        del x, y, z
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    atomic_json(ART / f"{name}_embedding_cache_v1.json", {"model": name, "device": str(device), "source_seconds": source_seconds,
                "source": {"rows": len(rows), "sha256": sha256_file(source_path)}, "dates": dates,
                "checkpoint_sha256": CHECKPOINTS[name][1], "new_backbone_training_runs": 0, "current_labels_cached": False})
    print(json.dumps({"model": name, "device": str(device), "source_rows": len(rows), "dates": {k:v["rows"] for k,v in dates.items()}}))


def class_support_order(manifest, date, seed):
    c3 = manifest["dates"][date]["configurations"][f"seed{seed}_shot3"]
    c10 = manifest["dates"][date]["configurations"][f"seed{seed}_shot10"]
    label_by_row = {int(x["row_index"]): int(x["label"]) for x in manifest["dates"][date]["eligible_rows"]}
    three, ten = set(c3["support_rows"]), set(c10["support_rows"])
    if not three < ten: raise ValueError("non-nested formal support")
    order = np.zeros((CLASSES, 10), np.int64)
    for label in range(CLASSES):
        a = sorted(row for row in three if label_by_row[row] == label)
        b = sorted(row for row in ten-three if label_by_row[row] == label)
        if len(a) != 3 or len(b) != 7: raise ValueError(f"bad class support {label}")
        order[label] = a+b
    return order, np.asarray(c10["query_rows"], np.int64)


def predict_all_subsets(query_z, support_z):
    # support_z [C,10,D], output [120,Q]; dot products are computed once.
    proto_norm = np.linalg.norm(support_z[:, COMBOS, :].sum(2), axis=2).T  # [120,C]
    if np.any(proto_norm == 0): raise ValueError("zero subset prototype")
    out = np.empty((len(COMBOS), len(query_z)), np.int16)
    ten_norm = np.linalg.norm(support_z.sum(1), axis=1)
    ten_out = np.empty(len(query_z), np.int16)
    flat = support_z.reshape(CLASSES*10, 512)
    for start in range(0, len(query_z), 256):
        q = query_z[start:start+256]
        sims = (q @ flat.T).reshape(len(q), CLASSES, 10)
        for cstart in range(0, len(COMBOS), 12):
            cb = COMBOS[cstart:cstart+12]
            scores = sims[:, :, cb].sum(3).transpose(0, 2, 1) / proto_norm[cstart:cstart+len(cb)][None, :, :]
            out[cstart:cstart+len(cb), start:start+len(q)] = scores.argmax(2).T.astype(np.int16)
        ten_out[start:start+len(q)] = (sims.sum(2) / ten_norm[None, :]).argmax(1).astype(np.int16)
    return out, ten_out


def support_metrics_for_subset(source_z, source_y, thresholds, centers, support):
    metrics = np.zeros((len(COMBOS), CLASSES, 4), np.float32)
    for label in range(CLASSES):
        sv = source_z[source_y == label]
        sup = support[label]
        src_sim = sv @ sup.T
        for ci, combo in enumerate(COMBOS):
            v = sup[combo]
            coverage = np.mean((1.0-src_sim[:,combo]).min(1) <= thresholds[label])
            center = unit(v.mean(0))
            pair = 1.0 - np.asarray([v[0]@v[1], v[0]@v[2], v[1]@v[2]])
            instability = max(1.0-unit(v[[1,2]].mean(0))@center,
                              1.0-unit(v[[0,2]].mean(0))@center,
                              1.0-unit(v[[0,1]].mean(0))@center)
            metrics[ci,label] = (coverage, 1.0-center@centers[label], pair.mean(), instability)
    return metrics


def freeze_stage(name: str, date: str) -> None:
    if not (ART / f"{name}_embedding_cache_v1.json").is_file(): raise ValueError("embedding cache missing")
    manifest = json.loads((DONOR_ART / "support_manifests.json").read_text())
    current = np.load(ART / f"{name}_{date}_embeddings_v1.npy", mmap_mode="r")
    source = np.load(ART / f"{name}_source_embeddings_v1.npz")
    geom = np.load(ART / f"{name}_source_geometry_v1.npz")
    sz, sy = source["embeddings"], source["labels"]
    thresholds, centers, exemplars = geom["thresholds"], geom["centers"], geom["exemplars"]
    for seed in SEEDS:
        out_path = ART / f"frozen_{name}_{date}_seed{seed}_v1.npz"; refuse(out_path)
        order, query = class_support_order(manifest, date, seed)
        support = np.asarray(current[order], np.float32)
        subset_pred, ten_pred = predict_all_subsets(np.asarray(current[query], np.float32), support)
        support_metrics = support_metrics_for_subset(sz, sy, thresholds, centers, support)
        added_category = np.zeros((CLASSES,7), np.int8)
        added_d3 = np.zeros((CLASSES,7), np.float32); added_dsrc = np.zeros((CLASSES,7), np.float32)
        center_shift = np.zeros(CLASSES, np.float32); scale3 = np.zeros(CLASSES,np.float32); scale10=np.zeros(CLASSES,np.float32)
        for label in range(CLASSES):
            v3, v10, va = support[label,:3], support[label], support[label,3:]
            d3 = 1.0-(va@v3.T).max(1); dsrc=1.0-(va@sz[sy==label].T).max(1)
            added_d3[label], added_dsrc[label] = d3, dsrc
            added_category[label] = np.where(d3<=thresholds[label],0,np.where(dsrc<=thresholds[label],1,2))
            c3, c10 = unit(v3.mean(0)), unit(v10.mean(0)); center_shift[label]=1.0-c3@c10
            scale3[label]=(1.0-v3@c3).mean(); scale10[label]=(1.0-v10@c10).mean()
        # One fixed historical multi-prototype probe: 3 source exemplars + current prototype.
        current_proto = np.stack([unit(support[c,:3].mean(0)) for c in range(CLASSES)])
        protos = np.concatenate([exemplars, current_proto[:,None,:]], axis=1).reshape(CLASSES*4,512)
        probe = np.empty(len(query), np.int16)
        for start in range(0,len(query),256):
            scores=(np.asarray(current[query[start:start+256]])@protos.T).reshape(-1,CLASSES,4).max(2)
            probe[start:start+len(scores)]=scores.argmax(1).astype(np.int16)
        atomic_npz(out_path, query_rows=query, support_order=order, subset_predictions=subset_pred,
                   ten_predictions=ten_pred, multiprototype_predictions=probe,
                   support_metrics=support_metrics, added_category=added_category,
                   added_d3=added_d3, added_dsrc=added_dsrc, center_shift=center_shift,
                   scale3=scale3, scale10=scale10)
        print(out_path)


def seal() -> None:
    output = ART / "firewall_seal_v1.json"; refuse(output)
    paths = [ART / "input_manifest_v1.json", ART / "resampling_manifest_v1.json"]
    for name in MODELS:
        paths += [ART/f"{name}_source_embeddings_v1.npz", ART/f"{name}_source_geometry_v1.npz", ART/f"{name}_source_geometry_v1.json", ART/f"{name}_embedding_cache_v1.json"]
        paths += [ART/f"{name}_{date}_embeddings_v1.npy" for date in DATES]
        paths += [ART/f"frozen_{name}_{date}_seed{seed}_v1.npz" for date in DATES for seed in SEEDS]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing: raise ValueError(f"missing freeze artifacts: {missing}")
    atomic_json(output,{"schema_version":1,"sealed_before_query_label_access":True,"files":{str(p):sha256_file(p) for p in paths},
                        "models":list(MODELS),"dates":list(DATES),"seeds":list(SEEDS),"combinations":len(COMBOS),
                        "new_backbone_training_runs":0,"query_labels_used_for_rules_predictions_or_metrics":False})
    print(output)


def confusion_metrics(truth, pred):
    conf=np.zeros((CLASSES,CLASSES),np.int64); np.add.at(conf,(truth,pred),1)
    tp=np.diag(conf).astype(float); actual=conf.sum(1); predicted=conf.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(CLASSES),where=predicted!=0)
    recall=np.divide(tp,actual,out=np.zeros(CLASSES),where=actual!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(CLASSES),where=precision+recall!=0)
    return float(tp.sum()/conf.sum()),float(f1.mean()),recall,f1


def write_csv(path, rows, fields):
    refuse(path)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def score() -> None:
    seal_path=ART/"firewall_seal_v1.json"
    seal_record=json.loads(seal_path.read_text())
    for path,expected in seal_record["files"].items():
        if sha256_file(Path(path))!=expected: raise ValueError(f"seal mismatch {path}")
    manifest=json.loads((DONOR_ART/"support_manifests.json").read_text())
    resampling=[]; site_rows=[]; support_rows=[]; error_rows=[]; probe_rows=[]; unit_summaries=[]
    metric_names=("source_coverage","center_to_source","within_support_dispersion","jackknife_center_instability")
    all_truth={}
    for date in DATES:
        _, y=load_npz(DATA/f"{date}.npz"); all_truth[date]=y
    for name in MODELS:
      current_by_date={d:np.load(ART/f"{name}_{d}_embeddings_v1.npy",mmap_mode="r") for d in DATES}
      for date in DATES:
       y=all_truth[date]; current=current_by_date[date]
       for seed in SEEDS:
        p=ART/f"frozen_{name}_{date}_seed{seed}_v1.npz"; a=np.load(p)
        query=a["query_rows"]; truth=y[query]; preds=a["subset_predictions"]; ten=a["ten_predictions"]
        donor3=json.loads((DONOR_ART/f"{name}_{date}_seed{seed}_shot3.json").read_text())
        donor10=json.loads((DONOR_ART/f"{name}_{date}_seed{seed}_shot10.json").read_text())
        map3={r:v for r,v in zip(donor3["query_rows"],donor3["predictions"]["simple_maintenance"])}
        expected3=np.asarray([map3[int(r)] for r in query],np.int16)
        if not np.array_equal(preds[0],expected3) or not np.array_equal(ten,np.asarray(donor10["predictions"]["simple_maintenance"],np.int16)):
            raise ValueError(f"donor prediction mismatch {name}/{date}/{seed}")
        f1s=[]; site_acc=np.zeros((len(COMBOS),CLASSES)); site_f1=np.zeros_like(site_acc)
        for ci,pred in enumerate(preds):
            acc,f1,rec,pf1=confusion_metrics(truth,pred); f1s.append(f1); site_acc[ci]=rec; site_f1[ci]=pf1
            resampling.append({"backbone":name,"date":date,"seed":seed,"subset_id":ci,"positions":"-".join(map(str,COMBOS[ci])),"is_formal3":ci==0,"accuracy":acc,"macro_f1":f1})
        f1s=np.asarray(f1s); acc10,f110,rec10,pf110=confusion_metrics(truth,ten)
        f13=f1s[0]; gap=f110-f13
        q=np.quantile(f1s,[.05,.1,.5,.9,.95],method="linear")
        percentile=float((np.sum(f1s<f13)+.5*np.sum(f1s==f13))/len(f1s))
        unit_summary={"backbone":name,"date":date,"seed":seed,"formal3_common_f1":float(f13),"ten_f1":f110,"gap":gap,
          "subset_mean":float(f1s.mean()),"subset_sd":float(f1s.std(ddof=1)),"min":float(f1s.min()),"q05":float(q[0]),"q10":float(q[1]),"median":float(q[2]),"q90":float(q[3]),"q95":float(q[4]),"max":float(f1s.max()),
          "formal3_percentile":percentile,"p_within_2pp_ten":float(np.mean(f1s>=f110-.02)),"sampling_spread_q90_q10":float(q[3]-q[1]),
          "oracle_explain":None if gap<=0 else float(np.clip((q[4]-f13)/gap,0,1)),"mean_residual":float(f110-f1s.mean())}
        order=a["support_order"]; categories=a["added_category"]
        # site-level rows and deployable metrics
        for ci in range(len(COMBOS)):
          for label in range(CLASSES):
            m=a["support_metrics"][ci,label]
            row={"backbone":name,"date":date,"seed":seed,"subset_id":ci,"site":label,"site_accuracy":site_acc[ci,label],"site_f1":site_f1[ci,label]}
            row.update(dict(zip(metric_names,map(float,m)))); support_rows.append(row)
            site_rows.append({"backbone":name,"date":date,"seed":seed,"subset_id":ci,"site":label,"accuracy":site_acc[ci,label],"f1":site_f1[ci,label]})
        # query-label error attribution on common query
        formal=preds[0]; repaired=(formal!=truth)&(ten==truth); harmed=(formal==truth)&(ten!=truth)
        closer_all=[]; closer_repaired=[]; repaired_cats=[]
        for qi,rowid in enumerate(query):
            label=int(truth[qi]); qz=np.asarray(current[rowid]); old=support= np.asarray(current[order[label,:3]])
            add=np.asarray(current[order[label,3:]])
            d_old=float(1.0-(old@qz).max()); add_i=int(np.argmax(add@qz)); d_add=float(1.0-add[add_i]@qz); closer=d_add<d_old
            closer_all.append(closer)
            if repaired[qi]: closer_repaired.append(closer); repaired_cats.append(int(categories[label,add_i]))
            if repaired[qi] or harmed[qi]:
                error_rows.append({"backbone":name,"date":date,"seed":seed,"query_row":int(rowid),"site":label,"event":"repaired" if repaired[qi] else "harmed","pred3":int(formal[qi]),"pred10":int(ten[qi]),"truth":label,"d_original3":d_old,"d_added7":d_add,"added_is_closer":closer,"nearest_added_category":int(categories[label,add_i])})
        probe_acc,probe_f1,_,_=confusion_metrics(truth,a["multiprototype_predictions"])
        explained=None if gap<=0 else float((probe_f1-f13)/gap)
        probe_rows.append({"backbone":name,"date":date,"seed":seed,"method":"source_3_farthest_first_plus_current_prototype","accuracy":probe_acc,"macro_f1":probe_f1,"gap_explained":explained})
        counts=np.bincount(categories.ravel(),minlength=3)
        unit_summary.update({"repaired":int(repaired.sum()),"harmed":int(harmed.sum()),"all_query_added_closer":float(np.mean(closer_all)),
          "repaired_added_closer":None if not closer_repaired else float(np.mean(closer_repaired)),"repaired_added_closer_enrichment":None if not closer_repaired else float(np.mean(closer_repaired)-np.mean(closer_all)),
          "repaired_nearest_category_counts":np.bincount(repaired_cats,minlength=3).tolist(),"added_support_category_counts":counts.tolist(),
          "center_shift_mean":float(a["center_shift"].mean()),"scale3_mean":float(a["scale3"].mean()),"scale10_mean":float(a["scale10"].mean()),"probe_gap_explained":explained})
        unit_summaries.append(unit_summary)
    # post-hoc identifiability, with predeclared risk directions
    ident=[]
    for name in MODELS:
      for date in DATES:
        rows=[r for r in support_rows if r["backbone"]==name and r["date"]==date]
        outcome=np.asarray([r["site_accuracy"] for r in rows]); groups={(r["seed"],r["subset_id"]) for r in rows}
        failure=np.zeros(len(rows),dtype=int)
        for group in groups:
            idx=np.asarray([i for i,r in enumerate(rows) if (r["seed"],r["subset_id"])==group]); cutoff=np.quantile(outcome[idx],.25,method="linear"); failure[idx]=(outcome[idx]<=cutoff)
        for metric in metric_names:
            raw=np.asarray([r[metric] for r in rows]); risk=1-raw if metric=="source_coverage" else raw
            rho=float(spearmanr(risk,outcome).statistic); auc=float(roc_auc_score(failure,risk)) if len(np.unique(failure))==2 else None
            ident.append({"backbone":name,"date":date,"metric":metric,"n":len(rows),"spearman_risk_vs_site_accuracy":rho,"bottom_quartile_failure_auc":auc})
    # repeatability of source-supported/current-new additions
    repeat=[]
    for name in MODELS:
      for date in DATES:
       values={cat:[] for cat in (1,2)}
       for label in range(CLASSES):
        for cat in (1,2):
          hits=0
          for seed in SEEDS:
            a=np.load(ART/f"frozen_{name}_{date}_seed{seed}_v1.npz"); hits+=bool(np.any(a["added_category"][label]==cat))
          if hits>=2: values[cat].append(label)
       repeat.append({"backbone":name,"date":date,"source_supported_repeat_sites":values[1],"current_new_repeat_sites":values[2]})
    # preregistered machine adjudication
    a_oracle=sum((u["oracle_explain"] or 0)>=.5 for u in unit_summaries); a_near=sum(u["p_within_2pp_ten"]>=.10 for u in unit_summaries)
    link=sum(u["repaired_added_closer_enrichment"] is not None and u["repaired_added_closer"]>=u["all_query_added_closer"] and u["repaired_added_closer_enrichment"]>=.10 for u in unit_summaries)
    repeated_dates=0
    for date in DATES:
        if all(next(r for r in repeat if r["backbone"]==m and r["date"]==date)["source_supported_repeat_sites"] for m in MODELS): repeated_dates+=1
    identifiable=any(all(any(i["metric"]==metric and (i["spearman_risk_vs_site_accuracy"]<=-.20 or i["bottom_quartile_failure_auc"]>=.65) for i in ident if i["backbone"]==m) for m in MODELS) for metric in metric_names)
    b_pass=link>=12 and repeated_dates>=2 and identifiable
    a_pass=a_oracle>=12 and a_near>=9 and not b_pass
    verdict="A_estimation_noise_dominant" if a_pass else ("B_historical_coverage_dominant" if b_pass and not (a_oracle>=12 and a_near>=9) else ("C_mixed" if b_pass else "C_insufficient"))
    history_counts=np.sum([u["added_support_category_counts"] for u in unit_summaries],axis=0).tolist()
    stop_multi=all(sum((p["gap_explained"] is not None and p["gap_explained"]>=.75) for p in probe_rows if p["backbone"]==m and p["date"]==d)>=2 for m in MODELS for d in DATES)
    summary={"schema_version":1,"unit_summaries":unit_summaries,"identifiability":ident,"repeatability":repeat,
      "adjudication":{"verdict":verdict,"A_oracle_units":a_oracle,"A_near10_units":a_near,"B_linkage_units":link,"B_repeated_dates_both_backbones":repeated_dates,"B_identifiable":identifiable,"A_pass":a_pass,"B_pass":b_pass},
      "added_support_categories":{"near_original3":history_counts[0],"far3_source_supported":history_counts[1],"far3_source_unsupported":history_counts[2]},
      "multiprototype_stop_condition":stop_multi,"prior_shrinkage_context":"exp_a2ffb5623ad346c7: gap shrinkage was mainly under-adaptation and did not retain late recovery","new_backbone_training_runs":0}
    write_csv(ART/"resampling_results.csv",resampling,list(resampling[0])); write_csv(ART/"per_site_stats.csv",site_rows,list(site_rows[0])); write_csv(ART/"support_only_metrics.csv",support_rows,list(support_rows[0])); write_csv(ART/"error_attribution.csv",error_rows,list(error_rows[0])); write_csv(ART/"probe_results.csv",probe_rows,list(probe_rows[0]))
    atomic_json(ART/"summary.json",summary); atomic_json(ART/"error_attribution_summary.json",{"units":unit_summaries}); atomic_json(ART/"probe_results.json",{"rows":probe_rows,"stop_condition":stop_multi})
    print(json.dumps(summary["adjudication"],indent=2))


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="stage",required=True)
    sub.add_parser("preflight")
    e=sub.add_parser("extract"); e.add_argument("--model",choices=MODELS,required=True)
    f=sub.add_parser("freeze"); f.add_argument("--model",choices=MODELS,required=True); f.add_argument("--date",choices=DATES,required=True)
    sub.add_parser("seal"); sub.add_parser("score")
    a=p.parse_args()
    if a.stage=="preflight": preflight()
    elif a.stage=="extract": extract_stage(a.model)
    elif a.stage=="freeze": freeze_stage(a.model,a.date)
    elif a.stage=="seal": seal()
    else: score()

if __name__=="__main__": main()

