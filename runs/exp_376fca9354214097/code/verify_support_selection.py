#!/usr/bin/env python3
"""Independent integrity verifier for support-internal selection artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]; RUN=ROOT/"runs/exp_376fca9354214097"; ART=RUN/"artifacts"
DONOR=ROOT/"runs/exp_b471517a3e6f41e7"; MODELS=("df","varcnn_direction"); DATES=("day14","day90","day270"); SEEDS=(1729,6238,20260916); SHOTS=(3,10); CLASSES=102
GLOBAL=("G_source","proto_alpha_0.25","shrink_lambda_1.00","proto_alpha_0.50","shrink_lambda_0.10","proto_alpha_0.75","shrink_lambda_0.01","current_prototype","G_current")
EXPECTED={RUN/"SUPPORT_SELECTION_PLAN.md":"89c3059fefc78fe47151e8a25e8107f64171542597d87af335539e401e3791dc",
 DONOR/"artifacts/support_manifests.json":"cbf37c99af375c576e882e049ffb4a37946ea631099d6b15f1ba9458782725ca",
 DONOR/"artifacts/data_isolation_audit.json":"6f1da408573f70cdf5ed64b80c9ce7b83ed56691a88208cb86a607403b3a17d6",
 DONOR/"artifacts/df_g_source.joblib":"895df6445a7ef7a068704a00f23f5fb84efbde4df7b6acf48aa5da4e85a27949",
 DONOR/"artifacts/varcnn_direction_g_source.joblib":"9e1d92c0fd5703473e4bbf4bd93e7cb6a7342100197a9e760a85b5c3d1fa777e"}
def sha(p):
 d=hashlib.sha256()
 with p.open("rb") as f:
  while b:=f.read(8*1024*1024): d.update(b)
 return d.hexdigest()
def calc(t,p):
 cm=np.zeros((CLASSES,CLASSES),dtype=np.int64); np.add.at(cm,(t,p),1); tp=np.diag(cm).astype(float); a=cm.sum(1); g=cm.sum(0)
 pr=np.divide(tp,g,out=np.zeros_like(tp),where=g!=0); re=np.divide(tp,a,out=np.zeros_like(tp),where=a!=0); f=np.divide(2*pr*re,pr+re,out=np.zeros_like(tp),where=pr+re!=0)
 return float(tp.sum()/cm.sum()),float(f.mean())
errors=[]; hashes={}; manifest=json.loads((DONOR/"artifacts/support_manifests.json").read_text()); dint=json.loads((DONOR/"artifacts/integrity_check.json").read_text())
for p,e in EXPECTED.items():
 a=sha(p); hashes[str(p)]={"expected":e,"actual":a,"match":a==e}
 if a!=e: errors.append(f"hash {p}")
checked=0; pred_rows=0; artifact_hashes={}; nonconverged=[]; convergence=[]
for m in MODELS:
 for d in DATES:
  for seed in SEEDS:
   rec={}
   for shot in SHOTS:
    p=ART/f"{m}_{d}_seed{seed}_shot{shot}.json"; r=json.loads(p.read_text()); rec[shot]=r; artifact_hashes[p.name]=sha(p)
    item=manifest["dates"][d]["configurations"][f"seed{seed}_shot{shot}"]; ten=manifest["dates"][d]["configurations"][f"seed{seed}_shot10"]
    if r["support_rows"]!=item["support_rows"] or r["common_query_rows"]!=ten["query_rows"]: errors.append(f"row mismatch {p.name}")
    if set(r["common_query_rows"])&set(ten["support_rows"]): errors.append(f"query/support overlap {p.name}")
    if r["cv"]["query_labels_used"] or r["new_backbone_training_runs"] or r["backbone_finetuning_runs"]: errors.append(f"permission {p.name}")
    truth=np.asarray(r["common_query_truth"]); chosen=r["cv"]["selected"]
    if r["predictions"]["support_selected_prototype"]!=r["predictions"][chosen["prototype"]] or r["predictions"]["support_selected_linear"]!=r["predictions"][chosen["linear"]] or r["predictions"]["support_selected_baseline"]!=r["predictions"][chosen["global"]]: errors.append(f"selected alias {p.name}")
    for method,pred0 in r["predictions"].items():
     pred=np.asarray(pred0); pred_rows+=len(pred)
     if len(pred)!=len(truth): errors.append(f"length {p.name}/{method}"); continue
     acc,f1=calc(truth,pred); saved=r["metrics"][method]
     if abs(acc-saved["accuracy"])>1e-12 or abs(f1-saved["macro_f1"])>1e-12 or len(saved["per_website"])!=CLASSES: errors.append(f"metric {p.name}/{method}")
    for fit in r["final_fit_records"].values():
     if "success" in fit and not fit["success"]: nonconverged.append(p.name)
     if fit.get("convergence_warning"): convergence.append(p.name)
    if shot==10:
     dp=DONOR/"artifacts"/p.name; dr=json.loads(dp.read_text())
     if sha(dp)!=dint["evaluation_artifact_sha256"][p.name]: errors.append(f"donor hash {p.name}")
     for ours,theirs in (("G_source","G_source"),("G_current","G_current"),("current_prototype","simple_maintenance")):
      if r["predictions"][ours]!=dr["predictions"][theirs]: errors.append(f"donor endpoint {p.name}/{ours}")
    checked+=1
   if rec[3]["common_query_rows"]!=rec[10]["common_query_rows"] or rec[3]["common_query_truth"]!=rec[10]["common_query_truth"]: errors.append(f"3/10 common query {m}/{d}/{seed}")
   if not set(rec[3]["support_rows"])<set(rec[10]["support_rows"]): errors.append(f"nested support {m}/{d}/{seed}")
for fn,n in (("metrics_long.csv",288),("error_transfers.csv",252),("selection_diagnostics.csv",108)):
 with (ART/fn).open(newline="") as f: rows=list(csv.DictReader(f))
 if len(rows)!=n: errors.append(f"{fn} rows {len(rows)}")
if list(RUN.glob("checkpoints/*")): errors.append("checkpoint output")
if list(RUN.rglob("*.tmp")): errors.append("temporary files")
result={"passed":not errors,"errors":errors,"warnings":{"nonconverged_shrink_fits":sorted(set(nonconverged)),"G_current_convergence_warnings":sorted(set(convergence))},
 "frozen_hashes":hashes,"evaluation_artifact_sha256":artifact_hashes,"evaluations_checked":checked,"prediction_rows_checked":pred_rows,
 "all_common_queries_shared_between_shots":not any("common query" in e for e in errors),"all_10shot_endpoints_match_donor":not any("donor endpoint" in e for e in errors),
 "new_backbone_training_runs":0,"backbone_finetuning_runs":0,"checkpoint_outputs":0}
path=ART/"integrity_check.json"
if path.exists(): raise FileExistsError(path)
tmp=path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); tmp.replace(path)
print(json.dumps({k:result[k] for k in ("passed","errors","warnings","evaluations_checked","prediction_rows_checked")},indent=2))
if errors: raise SystemExit(1)
