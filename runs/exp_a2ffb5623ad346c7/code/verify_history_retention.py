#!/usr/bin/env python3
"""Independent verifier for historical-retention formal artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/"runs/exp_a2ffb5623ad346c7"; DONOR=ROOT/"runs/exp_b471517a3e6f41e7"; ART=RUN/"artifacts"
MODELS=("df","varcnn_direction"); DATES=("day14","day90","day270"); SEEDS=(1729,6238,20260916); SHOTS=(3,10); CLASSES=102
METHODS=("A","G_source","G_current","current_prototype","prototype_interpolation","shrink_to_source")
EXPECTED={
 RUN/"HISTORY_RETENTION_PLAN.md":"909850d12eed764d0f881bd7d61860d2f06d189ebdcaf200f9a69ba4d4959561",
 DONOR/"artifacts/support_manifests.json":"cbf37c99af375c576e882e049ffb4a37946ea631099d6b15f1ba9458782725ca",
 DONOR/"artifacts/data_isolation_audit.json":"6f1da408573f70cdf5ed64b80c9ce7b83ed56691a88208cb86a607403b3a17d6",
 DONOR/"artifacts/df_g_source.joblib":"895df6445a7ef7a068704a00f23f5fb84efbde4df7b6acf48aa5da4e85a27949",
 DONOR/"artifacts/varcnn_direction_g_source.joblib":"9e1d92c0fd5703473e4bbf4bd93e7cb6a7342100197a9e760a85b5c3d1fa777e",
 ROOT/"runs/exp_9121b664a1854097/checkpoints/df_best.pt":"1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae",
 ROOT/"runs/exp_9121b664a1854097/checkpoints/varcnn_direction_best.pt":"fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83",
 ROOT/"runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json":"0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142",
}

def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as h:
  while block:=h.read(8*1024*1024): d.update(block)
 return d.hexdigest()

def metric(truth,pred):
 cm=np.zeros((CLASSES,CLASSES),dtype=np.int64); np.add.at(cm,(truth,pred),1); tp=np.diag(cm).astype(float)
 actual=cm.sum(1); predicted=cm.sum(0); p=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted!=0); r=np.divide(tp,actual,out=np.zeros_like(tp),where=actual!=0)
 f=np.divide(2*p*r,p+r,out=np.zeros_like(tp),where=p+r!=0); return float(tp.sum()/cm.sum()),float(f.mean())

def main():
 output=ART/"integrity_check.json"
 if output.exists(): raise FileExistsError(output)
 errors=[]; hashes={}
 for path,expected in EXPECTED.items():
  actual=sha(path); hashes[str(path)]={"expected":expected,"actual":actual,"match":actual==expected}
  if actual!=expected: errors.append(f"hash mismatch {path}")
 pre=json.loads((ART/"preflight.json").read_text()); selection=json.loads((ART/"source_selection.json").read_text()); manifest=json.loads((DONOR/"artifacts/support_manifests.json").read_text())
 donor_integrity=json.loads((DONOR/"artifacts/integrity_check.json").read_text())
 if not pre.get("passed") or pre.get("errors"): errors.append("preflight failed")
 if selection.get("query_dates_accessed")!=[] or not selection.get("shared_across_backbones_dates_shots_seeds"): errors.append("selection permission mismatch")
 if selection.get("selected_alpha") not in (.25,.5,.75) or selection.get("selected_lambda") not in (.01,.1,1.): errors.append("selected candidate invalid")
 if len(selection.get("alpha_records",[]))!=36 or len(selection.get("lambda_records",[]))!=36: errors.append("selection record count mismatch")
 checked=0; prediction_rows=0; artifact_hashes={}; nonconverged=[]
 for model in MODELS:
  for date in DATES:
   for seed in SEEDS:
    for shot in SHOTS:
     path=ART/f"{model}_{date}_seed{seed}_shot{shot}.json"; r=json.loads(path.read_text()); artifact_hashes[path.name]=sha(path)
     donor_path=DONOR/"artifacts"/path.name; donor=json.loads(donor_path.read_text())
     if sha(donor_path)!=donor_integrity["evaluation_artifact_sha256"][path.name]: errors.append(f"donor changed {path.name}")
     item=manifest["dates"][date]["configurations"][f"seed{seed}_shot{shot}"]
     if r["query_rows"]!=item["query_rows"] or r["support_rows"]!=item["support_rows"] or r["query_truth"]!=donor["query_truth"]: errors.append(f"row/truth mismatch {path.name}")
     if r["new_backbone_training_runs"]!=0 or r["prototype_interpolation"]["query_labels_used"] or r["shrink_to_source"]["query_labels_used"]: errors.append(f"permission/training mismatch {path.name}")
     if r["prototype_interpolation"]["alpha"]!=selection["selected_alpha"] or r["shrink_to_source"]["lambda"]!=selection["selected_lambda"]: errors.append(f"hyperparameter mismatch {path.name}")
     if not r["shrink_to_source"]["fit"]["success"]: nonconverged.append(path.name)
     truth=np.asarray(r["query_truth"]); base=np.asarray(r["predictions"]["G_source"]); base_ok=base==truth
     for method in METHODS:
      pred=np.asarray(r["predictions"][method]); prediction_rows+=len(pred)
      if len(pred)!=len(truth): errors.append(f"prediction length {path.name}/{method}"); continue
      acc,f1=metric(truth,pred); saved=r["metrics"][method]
      if abs(acc-saved["accuracy"])>1e-12 or abs(f1-saved["macro_f1"])>1e-12 or len(saved["per_website"])!=CLASSES: errors.append(f"metric mismatch {path.name}/{method}")
      donor_key="simple_maintenance" if method=="current_prototype" else method
      if method in ("A","G_source","G_current","current_prototype") and r["predictions"][method]!=donor["predictions"][donor_key]: errors.append(f"donor prediction mismatch {path.name}/{method}")
      if method!="G_source":
       ok=pred==truth; corrected=(~base_ok)&ok; harmed=base_ok&(~ok); t=r["transfers_vs_G_source"][method]
       if (t["corrected"],t["harmed"],t["net"])!=(int(corrected.sum()),int(harmed.sum()),int(corrected.sum()-harmed.sum())): errors.append(f"transfer mismatch {path.name}/{method}")
       for c in range(CLASSES):
        mask=truth==c; expected=(int(corrected[mask].sum()),int(harmed[mask].sum()),int(corrected[mask].sum()-harmed[mask].sum())); got=t["per_website"][str(c)]
        if expected!=(got["corrected"],got["harmed"],got["net"]): errors.append(f"site transfer mismatch {path.name}/{method}/{c}")
     for key,values in r["prediction_time_diagnostics"].items():
      if len(values)!=len(truth) or not np.isfinite(values).all(): errors.append(f"diagnostic mismatch {path.name}/{key}")
     checked+=1
 for filename,count in (("metrics_long.csv",216),("error_transfers.csv",180),("per_website_transfers.csv",18360)):
  with (ART/filename).open(newline="") as h: rows=list(csv.DictReader(h))
  if len(rows)!=count: errors.append(f"{filename} rows {len(rows)} != {count}")
 if list((RUN/"checkpoints").glob("*")): errors.append("checkpoint output exists")
 if list(RUN.rglob("*.tmp")): errors.append("temporary files exist")
 result={"passed":not errors,"errors":errors,"warnings":{"nonconverged_shrink_fits":nonconverged},"frozen_hashes":hashes,
  "evaluation_artifact_sha256":artifact_hashes,"evaluations_checked":checked,"prediction_rows_checked":prediction_rows,
  "all_donor_predictions_match":not any("donor prediction" in e for e in errors),"all_metrics_and_transfers_recomputed":not any("metric mismatch" in e or "transfer mismatch" in e for e in errors),
  "selection_used_no_current_query":selection.get("query_dates_accessed")==[],"selected_alpha":selection["selected_alpha"],"selected_lambda":selection["selected_lambda"],
  "new_backbone_training_runs":0,"checkpoint_outputs":0}
 tmp=output.with_suffix(".json.tmp"); tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); tmp.replace(output)
 print(json.dumps({k:result[k] for k in ("passed","errors","warnings","evaluations_checked","prediction_rows_checked","selected_alpha","selected_lambda")},indent=2))
 if errors: raise SystemExit(1)

if __name__=="__main__": main()
