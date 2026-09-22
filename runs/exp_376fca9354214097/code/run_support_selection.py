#!/usr/bin/env python3
"""Support-internal CV on frozen donor embeddings; no backbone training path."""
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
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from ta_wf_next.models import DF, VarCNNDirection
from ta_wf_next.screening import TraceDataset, load_npz, sha256_file

RUN = ROOT / "runs/exp_376fca9354214097"
DONOR = ROOT / "runs/exp_b471517a3e6f41e7"
HISTORY = ROOT / "runs/exp_a2ffb5623ad346c7"
CHECKPOINT_RUN = ROOT / "runs/exp_9121b664a1854097"
SPLIT = ROOT / "runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json"
DATA = Path("/mnt/data2/ren/datasets/TemporalDrift")
PLAN_HASH = "89c3059fefc78fe47151e8a25e8107f64171542597d87af335539e401e3791dc"
MODELS = ("df", "varcnn_direction")
DATES = ("day14", "day90", "day270")
SHOTS = (3, 10)
SEEDS = (1729, 6238, 20260916)
CLASSES = 102
ALPHAS = (0.25, 0.50, 0.75)
LAMBDAS = (1.00, 0.10, 0.01)
PROTO = ("G_source", "proto_alpha_0.25", "proto_alpha_0.50", "proto_alpha_0.75", "current_prototype")
LINEAR = ("G_source", "shrink_lambda_1.00", "shrink_lambda_0.10", "shrink_lambda_0.01", "G_current")
GLOBAL = ("G_source", "proto_alpha_0.25", "shrink_lambda_1.00", "proto_alpha_0.50",
          "shrink_lambda_0.10", "proto_alpha_0.75", "shrink_lambda_0.01", "current_prototype", "G_current")
CHECKPOINTS = {
    "df": ("df_best.pt", "1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae", 29),
    "varcnn_direction": ("varcnn_direction_best.pt", "fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83", 23),
}
FROZEN = {
    DONOR / "artifacts/support_manifests.json": "cbf37c99af375c576e882e049ffb4a37946ea631099d6b15f1ba9458782725ca",
    DONOR / "artifacts/data_isolation_audit.json": "6f1da408573f70cdf5ed64b80c9ce7b83ed56691a88208cb86a607403b3a17d6",
    DONOR / "artifacts/df_g_source.joblib": "895df6445a7ef7a068704a00f23f5fb84efbde4df7b6acf48aa5da4e85a27949",
    DONOR / "artifacts/varcnn_direction_g_source.joblib": "9e1d92c0fd5703473e4bbf4bd93e7cb6a7342100197a9e760a85b5c3d1fa777e",
    DONOR / "artifacts/integrity_check.json": "72dbf528ac40abb526df9827360f58277f70d674f30ecf281f53a6983a09f1d7",
    HISTORY / "artifacts/integrity_check.json": "acc699e61fdfa6d157bffc20a0154ccc17f884c923975346a49d47a3d79594c5",
    HISTORY / "artifacts/summary.json": "c6867b8b14c8590a96a014703de1525173167c227c3ee0a5214db3c05a01f468",
    HISTORY / "artifacts/metrics_long.csv": "cf07cbeeeb9ccaf6566830d7037afc4a205987148779d956f24c28645640991e",
    SPLIT: "0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142",
    CHECKPOINT_RUN / "checkpoints/df_best.pt": CHECKPOINTS["df"][1],
    CHECKPOINT_RUN / "checkpoints/varcnn_direction_best.pt": CHECKPOINTS["varcnn_direction"][1],
}


def atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite formal artifact: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def list_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def normalized(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, n, out=np.zeros_like(a), where=n != 0)


def metric(truth: np.ndarray, pred: np.ndarray, per_site: bool = False) -> dict:
    cm = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    np.add.at(cm, (truth, pred), 1)
    tp = np.diag(cm).astype(float); actual = cm.sum(1); guessed = cm.sum(0)
    p = np.divide(tp, guessed, out=np.zeros_like(tp), where=guessed != 0)
    r = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    f = np.divide(2*p*r, p+r, out=np.zeros_like(tp), where=p+r != 0)
    out = {"accuracy": float(tp.sum()/cm.sum()), "macro_precision": float(p.mean()),
           "macro_recall": float(r.mean()), "macro_f1": float(f.mean())}
    if per_site:
        out["per_website"] = {str(c): {"support": int(actual[c]), "accuracy": float(r[c]),
            "precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c])} for c in range(CLASSES)}
    return out


def mean_sd(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "values": a.tolist()}


def make_model(name, checkpoint):
    model = DF(CLASSES) if name == "df" else VarCNNDirection(CLASSES)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.cpu().eval()


def load_checkpoint(name):
    filename, expected, epoch = CHECKPOINTS[name]; path = CHECKPOINT_RUN / "checkpoints" / filename
    if sha256_file(path) != expected: raise ValueError(f"checkpoint hash mismatch: {name}")
    cp = torch.load(path, map_location="cpu", weights_only=True)
    history = json.loads((CHECKPOINT_RUN / "artifacts" / f"{name}_history.json").read_text())
    best = max(history, key=lambda r: (r["val_macro_f1"], -r["epoch"]))
    if cp.get("model_name") != name or cp.get("seed") != 6238 or cp.get("epoch") != epoch:
        raise ValueError(f"checkpoint metadata mismatch: {name}")
    if best["epoch"] != epoch or best["val_macro_f1"] != cp["selection_value"]:
        raise ValueError(f"checkpoint/history mismatch: {name}")
    return cp


def extract(model, x, y, indices=None):
    loader = DataLoader(TraceDataset(x, y, indices), batch_size=128, shuffle=False, num_workers=0)
    zs, rows = [], []; start = time.perf_counter()
    with torch.inference_mode():
        for xb, _, row in loader:
            _, z = model(xb)
            if z.shape[1:] != (512,): raise ValueError(f"bad embedding {z.shape}")
            zs.append(z.numpy()); rows.append(row.numpy())
    return normalized(np.concatenate(zs)), np.concatenate(rows), time.perf_counter()-start


def source_head(name):
    h = joblib.load(DONOR / "artifacts" / f"{name}_g_source.joblib")
    if h.coef_.shape != (CLASSES, 512) or h.intercept_.shape != (CLASSES,) or not np.array_equal(h.classes_, np.arange(CLASSES)):
        raise ValueError(f"bad source head {name}")
    return h


def prototype(x, y):
    if set(np.unique(y)) != set(range(CLASSES)): raise ValueError("prototype fit missing class")
    return normalized(np.stack([x[y == c].mean(0) for c in range(CLASSES)]))


def shrink_objective(theta, x, y, w0, b0, lam):
    w = theta[:CLASSES*x.shape[1]].reshape(CLASSES, x.shape[1]); b = theta[CLASSES*x.shape[1]:]
    logits = x @ w.T + b; logits -= logits.max(1, keepdims=True)
    prob = np.exp(logits); prob /= prob.sum(1, keepdims=True)
    loss = -np.log(prob[np.arange(len(y)), y]).mean(); dw, db = w-w0, b-b0
    value = loss + .5*lam*(np.sum(dw*dw)+np.sum(db*db))
    prob[np.arange(len(y)), y] -= 1
    grad = np.concatenate([(prob.T@x/len(y)+lam*dw).ravel(), prob.mean(0)+lam*db])
    return float(value), grad


def fit_shrink(x, y, head, lam):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    w0 = np.asarray(head.coef_, dtype=np.float64); b0 = np.asarray(head.intercept_, dtype=np.float64)
    start = time.perf_counter()
    r = minimize(shrink_objective, np.concatenate([w0.ravel(), b0]), args=(x,y,w0,b0,lam), jac=True,
                 method="L-BFGS-B", options={"maxiter":100,"maxls":20,"ftol":1e-9,"gtol":1e-5})
    w = r.x[:CLASSES*x.shape[1]].reshape(CLASSES,x.shape[1]); b = r.x[CLASSES*x.shape[1]:]
    record = {"success":bool(r.success),"status":int(r.status),"message":str(r.message),"nit":int(r.nit),
              "nfev":int(r.nfev),"final_objective":float(r.fun),"fit_seconds":time.perf_counter()-start}
    if not np.isfinite(r.fun) or not np.isfinite(r.x).all(): raise ValueError("non-finite shrink fit")
    return w, b, record


def fit_current(x, y):
    clf = LogisticRegression(penalty="l2", C=10.0, solver="lbfgs", max_iter=300, tol=1e-4,
                             fit_intercept=True, class_weight=None, random_state=6238)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning); clf.fit(x, y)
    rec = {"fit_seconds":time.perf_counter()-start,"n_iter":clf.n_iter_.tolist(),
           "convergence_warning":any(issubclass(v.category,ConvergenceWarning) for v in caught)}
    return clf, rec


def candidate_predictions(train_z, train_y, test_z, head, source_proto):
    cur = prototype(train_z, train_y); out = {"G_source": head.predict(test_z).astype(np.int64)}; fits = {}
    for a in ALPHAS:
        key = f"proto_alpha_{a:.2f}"; mixed = normalized((1-a)*source_proto+a*cur)
        out[key] = (test_z@mixed.T).argmax(1).astype(np.int64)
    out["current_prototype"] = (test_z@cur.T).argmax(1).astype(np.int64)
    for lam in LAMBDAS:
        key = f"shrink_lambda_{lam:.2f}"; w,b,rec = fit_shrink(train_z,train_y,head,lam)
        out[key] = (test_z@w.T+b).argmax(1).astype(np.int64); fits[key] = rec
    clf,rec = fit_current(train_z,train_y); out["G_current"] = clf.predict(test_z).astype(np.int64); fits["G_current"] = rec
    return out, fits


def choose(stats, candidates):
    valid = [c for c in candidates if stats[c]["valid"]]
    if not valid: return None
    priority = {c:i for i,c in enumerate(candidates)}
    return min(valid, key=lambda c:(-stats[c]["macro_f1_mean"],-stats[c]["accuracy_mean"],
                                    stats[c]["macro_f1_sd"],priority[c]))


def preflight():
    if sha256_file(RUN/"SUPPORT_SELECTION_PLAN.md") != PLAN_HASH: raise ValueError("plan hash mismatch")
    errors=[]; hashes={}
    for path,expected in FROZEN.items():
        actual=sha256_file(path); hashes[str(path)]={"expected":expected,"actual":actual,"match":actual==expected}
        if actual!=expected: errors.append(f"hash mismatch {path}")
    # Full parse/read of all required donor records after plan freeze.
    manifest=json.loads((DONOR/"artifacts/support_manifests.json").read_text())
    audit=json.loads((DONOR/"artifacts/data_isolation_audit.json").read_text())
    dint=json.loads((DONOR/"artifacts/integrity_check.json").read_text())
    dsum=json.loads((DONOR/"artifacts/summary.json").read_text())
    hint=json.loads((HISTORY/"artifacts/integrity_check.json").read_text())
    hsum=json.loads((HISTORY/"artifacts/summary.json").read_text())
    with (DONOR/"artifacts/metrics_long.csv").open(newline="") as f: dm=list(csv.DictReader(f))
    with (HISTORY/"artifacts/metrics_long.csv").open(newline="") as f: hm=list(csv.DictReader(f))
    split=json.loads(SPLIT.read_text())
    if not audit.get("passed") or audit.get("infeasible") or not dint.get("passed") or not hint.get("passed"): errors.append("donor integrity/isolation failed")
    if len(dm)!=144 or len(hm)!=216 or dsum.get("new_backbone_training_runs")!=0 or hsum.get("new_backbone_training_runs")!=0: errors.append("donor metric/summary shape")
    if split.get("schema_version")!=3 or not split.get("all_102_classes_in_each_role"): errors.append("split schema/coverage")
    for fn,expected in dint["evaluation_artifact_sha256"].items():
        if sha256_file(DONOR/"artifacts"/fn)!=expected: errors.append(f"donor evaluation changed {fn}")
    configs=0; common={}
    for date in DATES:
        eligible=[r["row_index"] for r in manifest["dates"][date]["eligible_rows"]]; common[date]={}
        for seed in SEEDS:
            three=manifest["dates"][date]["configurations"][f"seed{seed}_shot3"]
            ten=manifest["dates"][date]["configurations"][f"seed{seed}_shot10"]
            if not set(three["support_rows"]) < set(ten["support_rows"]): errors.append(f"not nested {date}/{seed}")
            if set(ten["query_rows"]) != set(eligible)-set(ten["support_rows"]): errors.append(f"bad common query {date}/{seed}")
            if list_hash(ten["query_rows"])!=ten["query_rows_sha256"]: errors.append(f"query hash {date}/{seed}")
            common[date][str(seed)]={"query_rows":ten["query_rows"],"query_rows_sha256":ten["query_rows_sha256"],
                "excluded_full_10shot_support":ten["support_rows"],"excluded_support_sha256":ten["support_rows_sha256"]}
            configs += 2
    for name in MODELS: load_checkpoint(name); source_head(name)
    atomic_json(RUN/"artifacts/preflight.json",{"passed":not errors,"errors":errors,"frozen_hashes":hashes,
        "fully_parsed":{"donor_metrics":len(dm),"history_metrics":len(hm),"manifest_configurations":configs,
        "donor_evaluations":len(dint["evaluation_artifact_sha256"]),"split_roles":{k:len(v["indices"]) for k,v in split["roles"].items()}},
        "new_backbone_training_runs":0,"backbone_finetuning_runs":0})
    atomic_json(RUN/"artifacts/common_query_manifests.json",{"schema_version":1,"rule":"eligible canonical pool minus full seed-specific 10-shot support","dates":common})
    if errors: raise SystemExit("; ".join(errors))
    print("preflight passed")


def prepare_source():
    pre=json.loads((RUN/"artifacts/preflight.json").read_text())
    if not pre["passed"]: raise ValueError("preflight failed")
    split=json.loads(SPLIT.read_text()); x,y=load_npz(DATA/"train.npz")
    idx=np.asarray(split["roles"]["supervised_train"]["indices"],dtype=np.int64); records={}
    for name in MODELS:
        model=make_model(name,load_checkpoint(name)); z,rows,seconds=extract(model,x,y,idx)
        if not np.array_equal(rows,idx): raise ValueError("source extraction order")
        p=prototype(z,y[rows]); path=RUN/"artifacts"/f"{name}_source_prototypes.npy"
        if path.exists(): raise FileExistsError(path)
        np.save(path,p); records[name]={"rows":len(rows),"seconds":seconds,"sha256":sha256_file(path)}
    atomic_json(RUN/"artifacts/source_preparation.json",{"fit_role":"source supervised_train only","models":records,"new_backbone_training_runs":0})
    print("source prototypes prepared")


def evaluate(name,date):
    outputs=[RUN/"artifacts"/f"{name}_{date}_seed{s}_shot{k}.json" for s in SEEDS for k in SHOTS]
    if any(p.exists() for p in outputs): raise FileExistsError("formal output exists")
    manifest=json.loads((DONOR/"artifacts/support_manifests.json").read_text())["dates"][date]
    common=json.loads((RUN/"artifacts/common_query_manifests.json").read_text())["dates"][date]
    model=make_model(name,load_checkpoint(name)); head=source_head(name); source_proto=np.load(RUN/"artifacts"/f"{name}_source_prototypes.npy")
    x,y=load_npz(DATA/f"{date}.npz"); z,rows,seconds=extract(model,x,y,None)
    if not np.array_equal(rows,np.arange(len(y))): raise ValueError("date extraction order")
    staged=[]
    for seed in SEEDS:
        query=np.asarray(common[str(seed)]["query_rows"],dtype=np.int64)
        for shot in SHOTS:
            item=manifest["configurations"][f"seed{seed}_shot{shot}"]; support=np.asarray(item["support_rows"],dtype=np.int64)
            sy=y[support]; K=3 if shot==3 else 5
            fold=np.empty(len(support),dtype=np.int64)
            for c in range(CLASSES):
                pos=np.flatnonzero(sy==c); pos=pos[np.argsort(support[pos])]
                if len(pos)!=shot: raise ValueError("support class count")
                fold[pos]=np.arange(shot)%K
            fold_scores={c:[] for c in GLOBAL}; fold_fits=[]
            for f in range(K):
                tr=fold!=f; va=fold==f
                preds,fits=candidate_predictions(z[support[tr]],sy[tr],z[support[va]],head,source_proto)
                fold_fits.append(fits)
                for c,p in preds.items(): fold_scores[c].append(metric(sy[va],p,False))
            stats={}
            for c,scores in fold_scores.items():
                f1=np.asarray([v["macro_f1"] for v in scores]); acc=np.asarray([v["accuracy"] for v in scores])
                stats[c]={"valid":bool(np.isfinite(f1).all() and np.isfinite(acc).all()),"folds":scores,
                    "macro_f1_mean":float(f1.mean()),"macro_f1_sd":float(f1.std(ddof=1)),
                    "accuracy_mean":float(acc.mean()),"accuracy_sd":float(acc.std(ddof=1))}
            selected_proto=choose(stats,PROTO); selected_linear=choose(stats,LINEAR); selected_global=choose(stats,GLOBAL)
            final_pred,final_fits=candidate_predictions(z[support],sy,z[query],head,source_proto)
            # Fold-wise winner agreement and winner-runner uncertainty are frozen diagnostics.
            for family,cands,winner in (("prototype",PROTO,selected_proto),("linear",LINEAR,selected_linear),("global",GLOBAL,selected_global)):
                priority={c:i for i,c in enumerate(cands)}
                fold_winners=[]
                for f in range(K):
                    fold_winners.append(min(cands,key=lambda c:(-stats[c]["folds"][f]["macro_f1"],-stats[c]["folds"][f]["accuracy"],priority[c])))
                ranked=sorted(cands,key=lambda c:(-stats[c]["macro_f1_mean"],-stats[c]["accuracy_mean"],stats[c]["macro_f1_sd"],priority[c]))
                first,second=ranked[:2]; se=float(np.sqrt(stats[first]["macro_f1_sd"]**2/K+stats[second]["macro_f1_sd"]**2/K))
                stats.setdefault("family_diagnostics",{})[family]={"winner":winner,"runner_up":second,
                    "winner_runner_macro_f1_margin":stats[first]["macro_f1_mean"]-stats[second]["macro_f1_mean"],
                    "winner_runner_difference_se":se,"fold_winners":fold_winners,
                    "fold_winner_agreement":float(sum(v==winner for v in fold_winners)/K)}
            aliases={"support_selected_prototype":final_pred[selected_proto],"support_selected_linear":final_pred[selected_linear],
                     "support_selected_baseline":final_pred[selected_global]}
            staged.append({"seed":seed,"shot":shot,"support":support,"query":query,"item":item,"stats":stats,
                "selected":{"prototype":selected_proto,"linear":selected_linear,"global":selected_global},
                "predictions":final_pred|aliases,"fold_fit_records":fold_fits,"final_fit_records":final_fits})
    # Query truth is first indexed only after every prediction for this backbone/date is fixed.
    donor_integrity=json.loads((DONOR/"artifacts/integrity_check.json").read_text())
    for s in staged:
        donor_path=DONOR/"artifacts"/f"{name}_{date}_seed{s['seed']}_shot10.json"
        if sha256_file(donor_path)!=donor_integrity["evaluation_artifact_sha256"][donor_path.name]: raise ValueError("donor eval changed")
        donor=json.loads(donor_path.read_text()); truth=np.asarray(donor["query_truth"],dtype=np.int64)
        if donor["query_rows"]!=s["query"].tolist() or not np.array_equal(truth,y[s["query"]]): raise ValueError("common query truth mismatch")
        if not np.array_equal(s["predictions"]["G_source"],np.asarray(donor["predictions"]["G_source"])): raise ValueError("G-source mismatch")
        if s["shot"]==10:
            if not np.array_equal(s["predictions"]["G_current"],np.asarray(donor["predictions"]["G_current"])): raise ValueError("G-current mismatch")
            if not np.array_equal(s["predictions"]["current_prototype"],np.asarray(donor["predictions"]["simple_maintenance"])): raise ValueError("prototype mismatch")
        scores={c:metric(truth,p,True) for c,p in s["predictions"].items()}
        base=s["predictions"]["G_source"]==truth; transfers={}
        for c,p in s["predictions"].items():
            if c=="G_source": continue
            ok=p==truth; corrected=(~base)&ok; harmed=base&(~ok)
            transfers[c]={"corrected":int(corrected.sum()),"harmed":int(harmed.sum()),"net":int(corrected.sum()-harmed.sum()),
                "corrected_rate":float(corrected.mean()),"harmed_rate":float(harmed.mean()),"net_rate":float((corrected.sum()-harmed.sum())/len(truth))}
        oracle={}
        for family,cands in (("prototype",PROTO),("linear",LINEAR),("global",GLOBAL)):
            priority={c:i for i,c in enumerate(cands)}
            winner=min(cands,key=lambda c:(-scores[c]["macro_f1"],priority[c])); selected=s["selected"][family]
            oracle[family]={"candidate":winner,"macro_f1":scores[winner]["macro_f1"],"selected_candidate":selected,
                "selected_macro_f1":scores[selected]["macro_f1"],"regret":scores[winner]["macro_f1"]-scores[selected]["macro_f1"]}
        out={"model":name,"date":date,"seed":s["seed"],"shot":s["shot"],"plan_sha256":PLAN_HASH,
            "support_rows":s["support"].tolist(),"common_query_rows":s["query"].tolist(),"common_query_truth":truth.tolist(),
            "support_manifest_hashes":{k:s["item"][k] for k in ("support_rows_sha256","support_content_hashes_sha256")},
            "common_query_rows_sha256":list_hash(s["query"]),"cv":{"fold_count":3 if s["shot"]==3 else 5,
                "fold_rule":"within-class ascending row position modulo K","candidate_stats":s["stats"],"selected":s["selected"],
                "fold_fit_records":s["fold_fit_records"],"query_labels_used":False},
            "predictions":{k:v.tolist() for k,v in s["predictions"].items()},"metrics":scores,
            "transfers_vs_G_source":transfers,"posthoc_oracle":oracle,"final_fit_records":s["final_fit_records"],
            "fixed_rules":{"prototype":"proto_alpha_0.25","linear":"shrink_lambda_0.10"},
            "date_embedding_extraction_seconds":seconds,"feature_extraction_device":"cpu",
            "new_backbone_training_runs":0,"backbone_finetuning_runs":0}
        atomic_json(RUN/"artifacts"/f"{name}_{date}_seed{s['seed']}_shot{s['shot']}.json",out)
    print(f"{name}/{date}: 6 configurations")


def summarize():
    records=[json.loads((RUN/"artifacts"/f"{m}_{d}_seed{s}_shot{k}.json").read_text()) for m in MODELS for d in DATES for s in SEEDS for k in SHOTS]
    report_methods=("G_source","current_prototype","G_current","proto_alpha_0.25","shrink_lambda_0.10",
                    "support_selected_prototype","support_selected_linear","support_selected_baseline")
    metric_rows=[]; transfer_rows=[]; selection_rows=[]
    for r in records:
        base={k:r[k] for k in ("model","date","shot","seed")}
        for method in report_methods:
            metric_rows.append(base|{"method":method}|{k:r["metrics"][method][k] for k in ("accuracy","macro_precision","macro_recall","macro_f1")})
            if method!="G_source": transfer_rows.append(base|{"method":method}|r["transfers_vs_G_source"][method])
        for fam in ("prototype","linear","global"):
            d=r["cv"]["candidate_stats"]["family_diagnostics"][fam]; o=r["posthoc_oracle"][fam]; win=d["winner"]
            selection_rows.append(base|{"family":fam,"selected":win,"oracle":o["candidate"],"oracle_regret":o["regret"],
                "cv_macro_f1_mean":r["cv"]["candidate_stats"][win]["macro_f1_mean"],"cv_macro_f1_sd":r["cv"]["candidate_stats"][win]["macro_f1_sd"],
                "winner_runner_margin":d["winner_runner_macro_f1_margin"],"difference_se":d["winner_runner_difference_se"],
                "fold_winner_agreement":d["fold_winner_agreement"]})
    for fn,rows in (("metrics_long.csv",metric_rows),("error_transfers.csv",transfer_rows),("selection_diagnostics.csv",selection_rows)):
        path=RUN/"artifacts"/fn
        if path.exists(): raise FileExistsError(path)
        with path.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    grouped={}
    for m in MODELS:
        grouped[m]={}
        for d in DATES:
            grouped[m][d]={}
            for shot in SHOTS:
                rs=[r for r in records if r["model"]==m and r["date"]==d and r["shot"]==shot]
                grouped[m][d][str(shot)]={method:{x:mean_sd([r["metrics"][method][x] for r in rs]) for x in ("accuracy","macro_f1")} for method in report_methods}
    judgment={"by_backbone":{}}
    for m in MODELS:
        d14=grouped[m]["day14"]["10"]; f=d14["support_selected_baseline"]["macro_f1"]["mean"]
        protect={"vs_G_source":f-d14["G_source"]["macro_f1"]["mean"],
                 "vs_best_current":f-max(d14["G_current"]["macro_f1"]["mean"],d14["current_prototype"]["macro_f1"]["mean"])}
        protect["passes"]=protect["vs_G_source"]>=-.01 and protect["vs_best_current"]>=.02
        late=[]
        for d in ("day90","day270"):
            cell=grouped[m][d]["10"]; gs=cell["G_source"]["macro_f1"]["mean"]
            R=max(cell["G_current"]["macro_f1"]["mean"],cell["current_prototype"]["macro_f1"]["mean"])-gs
            H=cell["support_selected_baseline"]["macro_f1"]["mean"]-gs
            late.append({"date":d,"current_recovery":R,"selected_recovery":H,"retention":H/R if R>0 else None})
        late_pass=all(v["retention"]>=.8 for v in late if v["retention"] is not None)
        judgment["by_backbone"][m]={"day14":protect,"late":late,"late_passes":late_pass}
    judgment["passes_both_day14"]=all(v["day14"]["passes"] for v in judgment["by_backbone"].values())
    judgment["passes_both_late"]=all(v["late_passes"] for v in judgment["by_backbone"].values())
    judgment["solves_tradeoff"]=judgment["passes_both_day14"] and judgment["passes_both_late"]
    noise={}
    for shot in SHOTS:
        noise[str(shot)]={}
        for fam in ("prototype","linear","global"):
            rows=[r for r in selection_rows if r["shot"]==shot and r["family"]==fam]
            substantial=[r for r in rows if r["selected"]!=r["oracle"] and float(r["oracle_regret"])>.01]
            unstable=[r for r in rows if float(r["cv_macro_f1_sd"])>=.05 or float(r["fold_winner_agreement"])<.6 or float(r["winner_runner_margin"])<=float(r["difference_se"])]
            noise[str(shot)][fam]={"configurations":len(rows),"selected_oracle_match_rate":float(np.mean([r["selected"]==r["oracle"] for r in rows])),
                "oracle_regret":mean_sd([float(r["oracle_regret"]) for r in rows]),"substantial_misselections":len(substantial),
                "unstable_configurations":len(unstable),"selection_frequencies":{c:sum(r["selected"]==c for r in rows) for c in GLOBAL}}
    fixed_improvement={}
    for m in MODELS:
        fixed_improvement[m]={}
        for d in DATES:
            fixed_improvement[m][d]={}
            for shot in SHOTS:
                cell=grouped[m][d][str(shot)]; selected=cell["support_selected_baseline"]["macro_f1"]["mean"]
                fixed_improvement[m][d][str(shot)]={"selected_minus_fixed_alpha":selected-cell["proto_alpha_0.25"]["macro_f1"]["mean"],
                    "selected_minus_fixed_lambda":selected-cell["shrink_lambda_0.10"]["macro_f1"]["mean"],
                    "selected_minus_better_fixed":selected-max(cell["proto_alpha_0.25"]["macro_f1"]["mean"],cell["shrink_lambda_0.10"]["macro_f1"]["mean"])}
    atomic_json(RUN/"artifacts/summary.json",{"grouped_seed_mean_sd":grouped,"pre_registered_10shot_judgment":judgment,
        "selection_noise":noise,"fixed_rule_improvement":fixed_improvement,"rows":{"metrics":len(metric_rows),"transfers":len(transfer_rows),"selection":len(selection_rows)},
        "new_backbone_training_runs":0,"backbone_finetuning_runs":0})
    print("summary written")


def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="stage",required=True)
    s.add_parser("preflight"); s.add_parser("prepare-source"); ev=s.add_parser("evaluate")
    ev.add_argument("--model",choices=MODELS,required=True); ev.add_argument("--date",choices=DATES,required=True); s.add_parser("summarize")
    a=p.parse_args()
    if a.stage=="preflight": preflight()
    elif a.stage=="prepare-source": prepare_source()
    elif a.stage=="evaluate": evaluate(a.model,a.date)
    else: summarize()


if __name__ == "__main__": main()
