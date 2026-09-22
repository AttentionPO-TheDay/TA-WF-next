#!/usr/bin/env python3
"""Independent integrity verification of the replication artifacts."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]; RUN=ROOT/"runs/exp_3ae08f5b65a44cbf"; OLD=ROOT/"runs/exp_376fca9354214097"
DONOR=ROOT/"runs/exp_b471517a3e6f41e7"; MODELS=("df","varcnn_direction"); TRAIN=(1013,2024)
DATES=("day14","day90","day270"); SUPPORT=(1729,6238,20260916); SHOTS=(3,10); CLASSES=102
GLOBAL=("G_source","proto_alpha_0.25","shrink_lambda_1.00","proto_alpha_0.50","shrink_lambda_0.10","proto_alpha_0.75","shrink_lambda_0.01","current_prototype","G_current")

def sha(p):
 d=hashlib.sha256()
 with p.open("rb") as f:
  while b:=f.read(8*1024*1024): d.update(b)
 return d.hexdigest()
def lhash(v): return hashlib.sha256(np.asarray(v,dtype="<i8").tobytes()).hexdigest()
def metric(y,p):
 cm=np.zeros((CLASSES,CLASSES),dtype=np.int64); np.add.at(cm,(y,p),1); tp=np.diag(cm).astype(float); actual=cm.sum(1); guessed=cm.sum(0)
 pr=np.divide(tp,guessed,out=np.zeros_like(tp),where=guessed!=0); re=np.divide(tp,actual,out=np.zeros_like(tp),where=actual!=0); f=np.divide(2*pr*re,pr+re,out=np.zeros_like(tp),where=pr+re!=0)
 return {"accuracy":float(tp.sum()/cm.sum()),"macro_precision":float(pr.mean()),"macro_recall":float(re.mean()),"macro_f1":float(f.mean())}
def close(a,b): return abs(float(a)-float(b))<=1e-12

errors=[]; hashes={}; predictions=0; files=0
manifest=json.loads((DONOR/"artifacts/support_manifests.json").read_text()); common=json.loads((OLD/"artifacts/common_query_manifests.json").read_text())
oldint=json.loads((OLD/"artifacts/integrity_check.json").read_text())
for fn,expected in oldint["evaluation_artifact_sha256"].items():
 actual=sha(OLD/"artifacts"/fn)
 if actual!=expected: errors.append(f"old evaluation changed {fn}")
for m in MODELS:
 for t in TRAIN:
  recp=RUN/"artifacts"/f"{m}_seed{t}_checkpoint.json"; rec=json.loads(recp.read_text()); cp=ROOT/rec["checkpoint_path"]
  if sha(cp)!=rec["checkpoint_sha256"]: errors.append(f"checkpoint hash {m}/{t}")
  obj=torch.load(cp,map_location="cpu",weights_only=True)
  if obj.get("model_name")!=m or obj.get("seed")!=t or obj.get("epoch")!=rec["best_epoch"]: errors.append(f"checkpoint metadata {m}/{t}")
  hist=json.loads((ROOT/rec["history_path"]).read_text()); best=max(hist,key=lambda r:(r["val_macro_f1"],-r["epoch"]))
  if len(hist)!=30 or best["epoch"]!=rec["best_epoch"] or not close(best["val_macro_f1"],rec["source_validation_macro_f1"]): errors.append(f"history {m}/{t}")
  srcp=RUN/"artifacts"/f"{m}_seed{t}_source.json"; src=json.loads(srcp.read_text())
  if sha(ROOT/src["head_path"])!=src["head_sha256"] or sha(ROOT/src["prototype_path"])!=src["prototype_sha256"]: errors.append(f"source hashes {m}/{t}")
  for d in DATES:
   for s in SUPPORT:
    objs={}
    for k in SHOTS:
     p=RUN/"artifacts"/f"{m}_trainseed{t}_{d}_supportseed{s}_shot{k}.json"; o=json.loads(p.read_text()); files+=1; hashes[p.name]=sha(p); objs[k]=o
     expected_support=manifest["dates"][d]["configurations"][f"seed{s}_shot{k}"]["support_rows"]
     q=common["dates"][d][str(s)]["query_rows"]
     if o["support_rows"]!=expected_support or o["query_rows"]!=q or lhash(q)!=o["query_rows_sha256"]: errors.append(f"manifest/query {p.name}")
     if not o.get("query_labels_used_only_after_all_predictions_fixed"): errors.append(f"label permission {p.name}")
     y=np.asarray(o["query_truth"],dtype=np.int64)
     if len(set(y))!=CLASSES: errors.append(f"class coverage {p.name}")
     for method,predlist in o["predictions"].items():
      pred=np.asarray(predlist,dtype=np.int64); predictions+=len(pred)
      if len(pred)!=len(y): errors.append(f"prediction length {p.name}/{method}"); continue
      met=metric(y,pred)
      for key,val in met.items():
       if not close(val,o["metrics"][method][key]): errors.append(f"metric {p.name}/{method}/{key}")
      if len(o["metrics"][method].get("per_website",{}))!=CLASSES: errors.append(f"per-site {p.name}/{method}")
      if method!="G_source":
       base=np.asarray(o["predictions"]["G_source"])==y; ok=pred==y; c=int(((~base)&ok).sum()); h=int((base&(~ok)).sum()); tr=o["transfers_vs_G_source"][method]
       if (c,h,c-h)!=(tr["corrected"],tr["harmed"],tr["net"]): errors.append(f"transfer {p.name}/{method}")
     for fam,alias in (("prototype","support_selected_prototype"),("linear","support_selected_linear"),("global","support_selected_baseline")):
      if o["predictions"][alias]!=o["predictions"][o["selected"][fam]]: errors.append(f"selected alias {p.name}/{fam}")
    if objs[3]["query_rows"]!=objs[10]["query_rows"] or objs[3]["query_truth"]!=objs[10]["query_truth"]: errors.append(f"common query between shots {m}/{t}/{d}/{s}")
    if not set(objs[3]["support_rows"])<set(objs[10]["support_rows"]) or set(objs[10]["support_rows"])&set(objs[10]["query_rows"]): errors.append(f"support nesting/isolation {m}/{t}/{d}/{s}")

expected_checkpoints={f"{m}_seed{t}_best.pt" for m in MODELS for t in TRAIN}; actual_checkpoints={p.name for p in (RUN/"checkpoints").glob("*.pt")}
if actual_checkpoints!=expected_checkpoints: errors.append(f"checkpoint set {sorted(actual_checkpoints)}")
summary=json.loads((RUN/"artifacts/summary.json").read_text()); randomness=json.loads((RUN/"artifacts/randomness_decomposition.json").read_text())
if summary.get("new_backbone_training_runs")!=4 or summary.get("metrics_long_rows")!=1296: errors.append("summary shape")
if len(randomness.get("two_factor",{}))!=288: errors.append("variance decomposition shape")
result={"passed":not errors,"errors":errors,"new_training_runs":4,"new_checkpoints":len(actual_checkpoints),"new_evaluations_checked":files,
 "old_evaluations_hash_checked":len(oldint["evaluation_artifact_sha256"]),"prediction_rows_checked":predictions,"all_common_queries_shared_between_shots":not any("common query" in e for e in errors),
 "all_metrics_and_transfers_recomputed":not any(e.startswith(("metric","transfer")) for e in errors),"evaluation_artifact_sha256":hashes,
 "summary_sha256":sha(RUN/"artifacts/summary.json"),"randomness_decomposition_sha256":sha(RUN/"artifacts/randomness_decomposition.json"),"metrics_long_sha256":sha(RUN/"artifacts/metrics_long.csv")}
out=RUN/"artifacts/integrity_check.json"
if out.exists(): raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="evaluation_artifact_sha256"},indent=2))
if errors: raise SystemExit(1)
