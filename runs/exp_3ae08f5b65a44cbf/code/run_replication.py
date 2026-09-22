#!/usr/bin/env python3
"""Frozen backbone-seed replication for exp_3ae08f5b65a44cbf.

Training follows scripts/run_temporal_screening.py; support selection follows
exp_376fca9354214097/code/run_support_selection.py.  This copy is isolated in
the replication run and generalized only over the preregistered training seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
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
from ta_wf_next.screening import TraceDataset, classification_metrics, load_npz, seed_everything, sha256_file

RUN = ROOT / "runs/exp_3ae08f5b65a44cbf"
DONOR = ROOT / "runs/exp_b471517a3e6f41e7"
SELECT = ROOT / "runs/exp_376fca9354214097"
ORIGINAL = ROOT / "runs/exp_9121b664a1854097"
SPLIT = ROOT / "runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json"
DATA = Path("/mnt/data2/ren/datasets/TemporalDrift")
PLAN_HASH = "e3745039e54eed4e531c839671a2daf2c288fd3fe7d3aab71ffbb77c6f638614"
MODELS = ("df", "varcnn_direction")
TRAIN_SEEDS = (1013, 2024)
SUPPORT_SEEDS = (1729, 6238, 20260916)
DATES = ("day14", "day90", "day270")
SHOTS = (3, 10)
CLASSES = 102
ALPHAS = (0.25, 0.50, 0.75)
LAMBDAS = (1.00, 0.10, 0.01)
PROTO = ("G_source", "proto_alpha_0.25", "proto_alpha_0.50", "proto_alpha_0.75", "current_prototype")
LINEAR = ("G_source", "shrink_lambda_1.00", "shrink_lambda_0.10", "shrink_lambda_0.01", "G_current")
GLOBAL = ("G_source", "proto_alpha_0.25", "shrink_lambda_1.00", "proto_alpha_0.50",
          "shrink_lambda_0.10", "proto_alpha_0.75", "shrink_lambda_0.01", "current_prototype", "G_current")
FROZEN = {
    SPLIT: "0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142",
    DONOR / "artifacts/support_manifests.json": "cbf37c99af375c576e882e049ffb4a37946ea631099d6b15f1ba9458782725ca",
    DONOR / "artifacts/data_isolation_audit.json": "6f1da408573f70cdf5ed64b80c9ce7b83ed56691a88208cb86a607403b3a17d6",
    SELECT / "SUPPORT_SELECTION_PLAN.md": "89c3059fefc78fe47151e8a25e8107f64171542597d87af335539e401e3791dc",
    SELECT / "artifacts/common_query_manifests.json": "161ebc6e71885258808a865c97c346abdc77fba3b823e595ba510e8159e531ed",
    SELECT / "artifacts/integrity_check.json": "87276c08c8888844841cb32ad203cd35f3c2c44f4964967d60eab476c111c9ca",
    ORIGINAL / "checkpoints/df_best.pt": "1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae",
    ORIGINAL / "checkpoints/varcnn_direction_best.pt": "fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83",
}
DATA_HASHES = {
    "train.npz": "2994271abc43bc3e3513924367da475e0a724b2cdfce9212968f6dc05763c882",
    "valid.npz": "e0f3ea5b170ecd9476c70dbc24c712266f014727b43fb8a377a546eed5c7ab62",
    "day14.npz": "eaa52ab83aea590ebc336cac1e8c2c6df8ea37ecc5ea26d5686e951b18460333",
    "day90.npz": "eaa25c23657f22509ab3ac9018a63a12736fab007db45cfa2ea05cbb6c0b25a6",
    "day270.npz": "35e965b9eed0239ca9b4959228defb85626235c88ef77b9b83c842a32bab9488",
}


def atomic_json(path: Path, value: object, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite formal artifact: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def model_for(name: str):
    if name == "df": return DF(CLASSES)
    if name == "varcnn_direction": return VarCNNDirection(CLASSES)
    raise ValueError(name)


def checkpoint_path(name: str, seed: int) -> Path:
    return RUN / "checkpoints" / f"{name}_seed{seed}_best.pt"


def training_config(name: str, seed: int) -> dict:
    return {"architecture": name, "training_seed": seed, "classes": CLASSES,
        "model_definition": str(ROOT / ("src/ta_wf_next/models/df.py" if name == "df" else "src/ta_wf_next/models/varcnn.py")),
        "model_definition_sha256": sha256_file(ROOT / ("src/ta_wf_next/models/df.py" if name == "df" else "src/ta_wf_next/models/varcnn.py")),
        "split_path": str(SPLIT), "split_sha256": FROZEN[SPLIT], "train_npz_sha256": DATA_HASHES["train.npz"],
        "valid_npz_sha256": DATA_HASHES["valid.npz"], "train_role": "v3 supervised_train",
        "input": "float32 sign(X[:5000]), shape [B,1,5000]", "optimizer": "AdamW", "learning_rate": 0.001,
        "weight_decay": 0.0001, "batch_size": 64, "validation_batch_size": 128, "epochs": 30,
        "criterion": "CrossEntropyLoss", "selection": "maximum official source-validation macro_f1; earliest exact tie",
        "cudnn_deterministic": True, "cudnn_benchmark": False}


def preflight() -> None:
    errors=[]; hashes={}
    if sha256_file(RUN/"REPLICATION_PLAN.md") != PLAN_HASH: errors.append("replication plan hash mismatch")
    for path, expected in FROZEN.items():
        actual=sha256_file(path); hashes[str(path)]={"expected":expected,"actual":actual,"match":actual==expected}
        if actual != expected: errors.append(f"frozen hash mismatch: {path}")
    for name, expected in DATA_HASHES.items():
        actual=sha256_file(DATA/name); hashes[str(DATA/name)]={"expected":expected,"actual":actual,"match":actual==expected}
        if actual != expected: errors.append(f"data hash mismatch: {name}")
    split=json.loads(SPLIT.read_text()); manifest=json.loads((DONOR/"artifacts/support_manifests.json").read_text())
    audit=json.loads((DONOR/"artifacts/data_isolation_audit.json").read_text())
    common=json.loads((SELECT/"artifacts/common_query_manifests.json").read_text())
    old_integrity=json.loads((SELECT/"artifacts/integrity_check.json").read_text())
    summary=json.loads((SELECT/"artifacts/summary.json").read_text())
    posthoc=json.loads((SELECT/"artifacts/posthoc_analysis.json").read_text())
    with (SELECT/"artifacts/selection_diagnostics.csv").open(newline="") as f: diagnostics=list(csv.DictReader(f))
    if not audit.get("passed") or audit.get("infeasible") or not old_integrity.get("passed"): errors.append("old isolation/integrity failed")
    if len(diagnostics)!=108 or summary.get("new_backbone_training_runs")!=0 or posthoc.get("failure_attribution") is None: errors.append("old formal artifact shape")
    for filename, expected in old_integrity["evaluation_artifact_sha256"].items():
        if sha256_file(SELECT/"artifacts"/filename)!=expected: errors.append(f"old evaluation changed: {filename}")
    cells=[]
    for date in DATES:
        eligible={r["row_index"] for r in manifest["dates"][date]["eligible_rows"]}
        for seed in SUPPORT_SEEDS:
            three=manifest["dates"][date]["configurations"][f"seed{seed}_shot3"]
            ten=manifest["dates"][date]["configurations"][f"seed{seed}_shot10"]
            cq=common["dates"][date][str(seed)]
            if not set(three["support_rows"]) < set(ten["support_rows"]): errors.append(f"not nested {date}/{seed}")
            if set(cq["query_rows"]) != eligible-set(ten["support_rows"]): errors.append(f"bad common query {date}/{seed}")
            if cq["query_rows"]!=ten["query_rows"] or list_hash(cq["query_rows"])!=cq["query_rows_sha256"]: errors.append(f"query hash {date}/{seed}")
            cells.append({"date":date,"support_seed":seed,"query_count":len(cq["query_rows"]),"query_sha256":cq["query_rows_sha256"]})
    configs={f"{m}_seed{s}": {"config":training_config(m,s), "sha256":canonical_hash(training_config(m,s))}
             for m in MODELS for s in TRAIN_SEEDS}
    out={"passed":not errors,"errors":errors,"frozen_hashes":hashes,"fully_parsed":{"split_roles":{k:v["count"] for k,v in split["roles"].items()},
        "support_manifest_bytes":(DONOR/"artifacts/support_manifests.json").stat().st_size,"old_evaluations":len(old_integrity["evaluation_artifact_sha256"]),
        "old_summary_rows":len(summary["rows"]),"selection_diagnostics_rows":len(diagnostics),"common_query_cells":cells},
        "training_configs":configs,"new_training_budget":4,"new_training_started":False}
    atomic_json(RUN/"artifacts/preflight.json",out)
    atomic_json(RUN/"artifacts/training_configs.json",configs)
    if errors: raise SystemExit("; ".join(errors))
    print(json.dumps({"passed":True,"training_config_hashes":{k:v["sha256"] for k,v in configs.items()}},indent=2))


def train(name: str, seed: int) -> None:
    if name not in MODELS or seed not in TRAIN_SEEDS: raise ValueError("not preregistered")
    pre=json.loads((RUN/"artifacts/preflight.json").read_text())
    if not pre["passed"] or sha256_file(RUN/"REPLICATION_PLAN.md")!=PLAN_HASH: raise ValueError("preflight/plan")
    out=checkpoint_path(name,seed); history_path=RUN/"artifacts"/f"{name}_seed{seed}_history.json"
    record_path=RUN/"artifacts"/f"{name}_seed{seed}_checkpoint.json"
    if out.exists() or history_path.exists() or record_path.exists(): raise FileExistsError("training output already exists")
    if not torch.cuda.is_available(): raise RuntimeError("Full training requires CUDA")
    split=json.loads(SPLIT.read_text()); idx=np.asarray(split["roles"]["supervised_train"]["indices"],dtype=np.int64)
    x,y=load_npz(DATA/"train.npz"); vx,vy=load_npz(DATA/"valid.npz")
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(TraceDataset(x,y,idx),batch_size=64,shuffle=True,num_workers=0,generator=generator)
    val_loader=DataLoader(TraceDataset(vx,vy),batch_size=128,shuffle=False,num_workers=0)
    seed_everything(seed); device=torch.device("cuda"); model=model_for(name).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); criterion=torch.nn.CrossEntropyLoss()
    history=[]; best=-1.0; started=time.perf_counter(); config=training_config(name,seed); config_hash=canonical_hash(config)
    for epoch in range(1,31):
        model.train(); total=correct=0; loss_sum=0.0
        for xb,yb,_ in train_loader:
            xb=xb.to(device); yb=yb.to(device); optimizer.zero_grad(set_to_none=True)
            logits,_=model(xb); loss=criterion(logits,yb); loss.backward(); optimizer.step()
            total+=len(yb); correct+=(logits.argmax(1)==yb).sum().item(); loss_sum+=loss.item()*len(yb)
        model.eval(); truth=[]; pred=[]
        with torch.inference_mode():
            for xb,yb,_ in val_loader:
                logits,_=model(xb.to(device)); truth.extend(yb.tolist()); pred.extend(logits.argmax(1).cpu().tolist())
        met=classification_metrics(np.asarray(truth),np.asarray(pred))
        row={"epoch":epoch,"train_loss":loss_sum/total,"train_accuracy":correct/total,
             **{f"val_{k}":v for k,v in met.items()},"elapsed_seconds":time.perf_counter()-started}
        history.append(row); print(json.dumps(row),flush=True)
        if met["macro_f1"]>best:
            best=met["macro_f1"]
            tmp=out.with_suffix(".pt.tmp")
            torch.save({"model":model.state_dict(),"model_name":name,"seed":seed,"epoch":epoch,
                "selection_metric":"source_validation_macro_f1","selection_value":best,"training_config_sha256":config_hash},tmp)
            tmp.replace(out)
    atomic_json(history_path,history)
    best_row=max(history,key=lambda r:(r["val_macro_f1"],-r["epoch"]))
    record={"architecture":name,"training_seed":seed,"best_epoch":best_row["epoch"],
        "source_validation_accuracy":best_row["val_accuracy"],"source_validation_macro_f1":best_row["val_macro_f1"],
        "checkpoint_path":str(out.relative_to(ROOT)),"checkpoint_sha256":sha256_file(out),
        "training_config_sha256":config_hash,"training_config":config,"history_path":str(history_path.relative_to(ROOT)),
        "history_sha256":sha256_file(history_path)}
    atomic_json(record_path,record); print(json.dumps(record,indent=2))


def load_new_checkpoint(name: str, seed: int):
    rec=json.loads((RUN/"artifacts"/f"{name}_seed{seed}_checkpoint.json").read_text()); path=checkpoint_path(name,seed)
    if rec["checkpoint_sha256"]!=sha256_file(path) or rec["training_config_sha256"]!=canonical_hash(training_config(name,seed)): raise ValueError("checkpoint/config hash")
    cp=torch.load(path,map_location="cpu",weights_only=True)
    if cp["model_name"]!=name or cp["seed"]!=seed or cp["epoch"]!=rec["best_epoch"]: raise ValueError("checkpoint metadata")
    return cp,rec


def extract(model, x, y, device, indices=None):
    loader=DataLoader(TraceDataset(x,y,indices),batch_size=128,shuffle=False,num_workers=0)
    zs=[]; rows=[]; started=time.perf_counter()
    model.to(device).eval()
    with torch.inference_mode():
        for xb,_,row in loader:
            _,z=model(xb.to(device)); zs.append(z.cpu().numpy()); rows.append(row.numpy())
    return normalized(np.concatenate(zs)),np.concatenate(rows),time.perf_counter()-started


def prototype(x,y):
    if set(np.unique(y))!=set(range(CLASSES)): raise ValueError("prototype missing class")
    return normalized(np.stack([x[y==c].mean(0) for c in range(CLASSES)]))


def prepare_source(name: str, seed: int) -> None:
    output=RUN/"artifacts"/f"{name}_seed{seed}_source.json"
    if output.exists(): raise FileExistsError(output)
    cp,rec=load_new_checkpoint(name,seed); model=model_for(name); model.load_state_dict(cp["model"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split=json.loads(SPLIT.read_text()); idx=np.asarray(split["roles"]["supervised_train"]["indices"],dtype=np.int64)
    x,y=load_npz(DATA/"train.npz"); vx,vy=load_npz(DATA/"valid.npz")
    z,rows,train_s=extract(model,x,y,device,idx); vz,vrows,valid_s=extract(model,vx,vy,device,None)
    if not np.array_equal(rows,idx) or not np.array_equal(vrows,np.arange(len(vy))): raise ValueError("extraction order")
    scores={}; fits={}
    for C in (0.1,1.0,10.0):
        clf=LogisticRegression(penalty="l2",C=C,solver="lbfgs",max_iter=300,tol=1e-4,fit_intercept=True,class_weight=None,random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always",ConvergenceWarning); clf.fit(z,y[rows])
        scores[str(C)]={"metrics":metric(vy,clf.predict(vz)),"n_iter":clf.n_iter_.tolist(),
            "convergence_warning":any(issubclass(v.category,ConvergenceWarning) for v in caught)}; fits[C]=clf
    selected=min((0.1,1.0,10.0),key=lambda C:(-scores[str(C)]["metrics"]["macro_f1"],C))
    head_path=RUN/"artifacts"/f"{name}_seed{seed}_g_source.joblib"; proto_path=RUN/"artifacts"/f"{name}_seed{seed}_source_prototypes.npy"
    joblib.dump(fits[selected],head_path); np.save(proto_path,prototype(z,y[rows]))
    atomic_json(output,{"architecture":name,"training_seed":seed,"checkpoint_sha256":rec["checkpoint_sha256"],
        "fit_role":"v3 supervised_train only","selection_role":"official source validation only","C_candidates":[0.1,1.0,10.0],
        "selected_C":selected,"candidate_scores":scores,"head_path":str(head_path.relative_to(ROOT)),"head_sha256":sha256_file(head_path),
        "prototype_path":str(proto_path.relative_to(ROOT)),"prototype_sha256":sha256_file(proto_path),
        "train_rows":len(rows),"valid_rows":len(vrows),"extraction_seconds":{"train":train_s,"valid":valid_s}})
    print(json.dumps({"selected_C":selected,"source_valid":scores[str(selected)]["metrics"]},indent=2))


def shrink_objective(theta,x,y,w0,b0,lam):
    w=theta[:CLASSES*x.shape[1]].reshape(CLASSES,x.shape[1]); b=theta[CLASSES*x.shape[1]:]
    logits=x@w.T+b; logits-=logits.max(1,keepdims=True); prob=np.exp(logits); prob/=prob.sum(1,keepdims=True)
    loss=-np.log(prob[np.arange(len(y)),y]).mean(); dw=w-w0; db=b-b0
    value=loss+.5*lam*(np.sum(dw*dw)+np.sum(db*db)); prob[np.arange(len(y)),y]-=1
    grad=np.concatenate([(prob.T@x/len(y)+lam*dw).ravel(),prob.mean(0)+lam*db])
    return float(value),grad


def fit_shrink(x,y,head,lam):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.int64); w0=np.asarray(head.coef_,dtype=np.float64); b0=np.asarray(head.intercept_,dtype=np.float64)
    started=time.perf_counter(); r=minimize(shrink_objective,np.concatenate([w0.ravel(),b0]),args=(x,y,w0,b0,lam),jac=True,
        method="L-BFGS-B",options={"maxiter":100,"maxls":20,"ftol":1e-9,"gtol":1e-5})
    if not np.isfinite(r.fun) or not np.isfinite(r.x).all(): raise ValueError("nonfinite shrink")
    w=r.x[:CLASSES*x.shape[1]].reshape(CLASSES,x.shape[1]); b=r.x[CLASSES*x.shape[1]:]
    return w,b,{"success":bool(r.success),"status":int(r.status),"message":str(r.message),"nit":int(r.nit),"nfev":int(r.nfev),"final_objective":float(r.fun),"fit_seconds":time.perf_counter()-started}


def fit_current(x,y,C,seed):
    clf=LogisticRegression(penalty="l2",C=C,solver="lbfgs",max_iter=300,tol=1e-4,fit_intercept=True,class_weight=None,random_state=seed)
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning); clf.fit(x,y)
    return clf,{"fit_seconds":time.perf_counter()-started,"n_iter":clf.n_iter_.tolist(),"convergence_warning":any(issubclass(v.category,ConvergenceWarning) for v in caught)}


def candidate_predictions(train_z,train_y,test_z,head,source_proto,C,seed):
    cur=prototype(train_z,train_y); out={"G_source":head.predict(test_z).astype(np.int64)}; fits={}
    for a in ALPHAS:
        mixed=normalized((1-a)*source_proto+a*cur); out[f"proto_alpha_{a:.2f}"]=(test_z@mixed.T).argmax(1).astype(np.int64)
    out["current_prototype"]=(test_z@cur.T).argmax(1).astype(np.int64)
    for lam in LAMBDAS:
        w,b,r=fit_shrink(train_z,train_y,head,lam); out[f"shrink_lambda_{lam:.2f}"]=(test_z@w.T+b).argmax(1).astype(np.int64); fits[f"shrink_lambda_{lam:.2f}"]=r
    clf,r=fit_current(train_z,train_y,C,seed); out["G_current"]=clf.predict(test_z).astype(np.int64); fits["G_current"]=r
    return out,fits


def choose(stats,candidates):
    priority={c:i for i,c in enumerate(candidates)}
    return min(candidates,key=lambda c:(-stats[c]["macro_f1_mean"],-stats[c]["accuracy_mean"],stats[c]["macro_f1_sd"],priority[c]))


def evaluate(name: str, train_seed: int, date: str) -> None:
    outputs=[RUN/"artifacts"/f"{name}_trainseed{train_seed}_{date}_supportseed{s}_shot{k}.json" for s in SUPPORT_SEEDS for k in SHOTS]
    if any(p.exists() for p in outputs): raise FileExistsError("formal evaluation exists")
    cp,rec=load_new_checkpoint(name,train_seed); model=model_for(name); model.load_state_dict(cp["model"])
    source=json.loads((RUN/"artifacts"/f"{name}_seed{train_seed}_source.json").read_text())
    head_path=ROOT/source["head_path"]; proto_path=ROOT/source["prototype_path"]
    if sha256_file(head_path)!=source["head_sha256"] or sha256_file(proto_path)!=source["prototype_sha256"]: raise ValueError("source artifact hash")
    head=joblib.load(head_path); source_proto=np.load(proto_path); C=float(source["selected_C"])
    manifest=json.loads((DONOR/"artifacts/support_manifests.json").read_text())["dates"][date]
    common=json.loads((SELECT/"artifacts/common_query_manifests.json").read_text())["dates"][date]
    x,y=load_npz(DATA/f"{date}.npz"); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z,rows,extract_s=extract(model,x,y,device,None)
    if not np.array_equal(rows,np.arange(len(y))): raise ValueError("date extraction order")
    staged=[]
    for support_seed in SUPPORT_SEEDS:
        query=np.asarray(common[str(support_seed)]["query_rows"],dtype=np.int64)
        for shot in SHOTS:
            item=manifest["configurations"][f"seed{support_seed}_shot{shot}"]; support=np.asarray(item["support_rows"],dtype=np.int64); sy=y[support]; K=3 if shot==3 else 5
            fold=np.empty(len(support),dtype=np.int64)
            for c in range(CLASSES):
                pos=np.flatnonzero(sy==c); pos=pos[np.argsort(support[pos])]
                if len(pos)!=shot: raise ValueError("support class count")
                fold[pos]=np.arange(shot)%K
            fold_scores={c:[] for c in GLOBAL}; fold_fits=[]
            for f in range(K):
                tr=fold!=f; va=fold==f; preds,fits=candidate_predictions(z[support[tr]],sy[tr],z[support[va]],head,source_proto,C,train_seed); fold_fits.append(fits)
                for c,p in preds.items(): fold_scores[c].append(metric(sy[va],p))
            stats={}
            for c,scores in fold_scores.items():
                f1=np.asarray([v["macro_f1"] for v in scores]); acc=np.asarray([v["accuracy"] for v in scores])
                stats[c]={"valid":bool(np.isfinite(f1).all() and np.isfinite(acc).all()),"folds":scores,"macro_f1_mean":float(f1.mean()),
                    "macro_f1_sd":float(f1.std(ddof=1)),"accuracy_mean":float(acc.mean()),"accuracy_sd":float(acc.std(ddof=1))}
            selected={"prototype":choose(stats,PROTO),"linear":choose(stats,LINEAR),"global":choose(stats,GLOBAL)}
            final_pred,final_fits=candidate_predictions(z[support],sy,z[query],head,source_proto,C,train_seed)
            for family,cands,winner in (("prototype",PROTO,selected["prototype"]),("linear",LINEAR,selected["linear"]),("global",GLOBAL,selected["global"])):
                priority={c:i for i,c in enumerate(cands)}; fold_winners=[]
                for f in range(K): fold_winners.append(min(cands,key=lambda c:(-stats[c]["folds"][f]["macro_f1"],-stats[c]["folds"][f]["accuracy"],priority[c])))
                ranked=sorted(cands,key=lambda c:(-stats[c]["macro_f1_mean"],-stats[c]["accuracy_mean"],stats[c]["macro_f1_sd"],priority[c])); first,second=ranked[:2]
                se=float(np.sqrt(stats[first]["macro_f1_sd"]**2/K+stats[second]["macro_f1_sd"]**2/K))
                stats.setdefault("family_diagnostics",{})[family]={"winner":winner,"runner_up":second,"winner_runner_macro_f1_margin":stats[first]["macro_f1_mean"]-stats[second]["macro_f1_mean"],
                    "winner_runner_difference_se":se,"fold_winners":fold_winners,"fold_winner_agreement":float(sum(v==winner for v in fold_winners)/K)}
            final_pred.update({"support_selected_prototype":final_pred[selected["prototype"]],"support_selected_linear":final_pred[selected["linear"]],"support_selected_baseline":final_pred[selected["global"]]})
            staged.append({"support_seed":support_seed,"shot":shot,"support":support,"query":query,"item":item,"stats":stats,"selected":selected,
                "predictions":final_pred,"fold_fit_records":fold_fits,"final_fit_records":final_fits})
    # Query truth is first indexed only after all six configurations are fixed.
    for s in staged:
        truth=y[s["query"]]; scores={c:metric(truth,p,True) for c,p in s["predictions"].items()}; base=s["predictions"]["G_source"]==truth; transfers={}
        for c,p in s["predictions"].items():
            if c=="G_source": continue
            ok=p==truth; corrected=(~base)&ok; harmed=base&(~ok); transfers[c]={"corrected":int(corrected.sum()),"harmed":int(harmed.sum()),"net":int(corrected.sum()-harmed.sum()),
                "corrected_rate":float(corrected.mean()),"harmed_rate":float(harmed.mean()),"net_rate":float((corrected.sum()-harmed.sum())/len(truth))}
        oracle={}
        for family,cands in (("prototype",PROTO),("linear",LINEAR),("global",GLOBAL)):
            priority={c:i for i,c in enumerate(cands)}; winner=min(cands,key=lambda c:(-scores[c]["macro_f1"],-scores[c]["accuracy"],priority[c])); selected=s["selected"][family]
            oracle[family]={"candidate":winner,"selected_candidate":selected,"regret_macro_f1":scores[winner]["macro_f1"]-scores[selected]["macro_f1"]}
        output=RUN/"artifacts"/f"{name}_trainseed{train_seed}_{date}_supportseed{s['support_seed']}_shot{s['shot']}.json"
        atomic_json(output,{"architecture":name,"training_seed":train_seed,"support_seed":s["support_seed"],"date":date,"shot":s["shot"],
            "checkpoint_sha256":rec["checkpoint_sha256"],"training_config_sha256":rec["training_config_sha256"],"source_artifact_sha256":sha256_file(RUN/"artifacts"/f"{name}_seed{train_seed}_source.json"),
            "support_rows":s["support"].tolist(),"support_rows_sha256":s["item"]["support_rows_sha256"],"query_rows":s["query"].tolist(),
            "query_rows_sha256":list_hash(s["query"]),"query_truth":truth.tolist(),"query_labels_used_only_after_all_predictions_fixed":True,
            "selected":s["selected"],"cv":s["stats"],"fold_fit_records":s["fold_fit_records"],"final_fit_records":s["final_fit_records"],
            "predictions":{k:v.tolist() for k,v in s["predictions"].items()},"metrics":scores,"transfers_vs_G_source":transfers,"posthoc_oracle":oracle,
            "date_embedding_extraction_seconds":extract_s})
    print(json.dumps({"architecture":name,"training_seed":train_seed,"date":date,"outputs":len(staged)},indent=2))


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("preflight")
    for cmd in ("train","prepare-source"):
        q=sub.add_parser(cmd); q.add_argument("--model",choices=MODELS,required=True); q.add_argument("--training-seed",type=int,choices=TRAIN_SEEDS,required=True)
    q=sub.add_parser("evaluate"); q.add_argument("--model",choices=MODELS,required=True); q.add_argument("--training-seed",type=int,choices=TRAIN_SEEDS,required=True); q.add_argument("--date",choices=DATES,required=True)
    a=p.parse_args()
    if a.command=="preflight": preflight()
    elif a.command=="train": train(a.model,a.training_seed)
    elif a.command=="prepare-source": prepare_source(a.model,a.training_seed)
    else: evaluate(a.model,a.training_seed,a.date)


if __name__ == "__main__": main()
