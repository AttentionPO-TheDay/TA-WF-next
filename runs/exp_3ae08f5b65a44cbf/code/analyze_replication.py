#!/usr/bin/env python3
"""Aggregate all three backbone training seeds and decompose two random factors."""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/"runs/exp_3ae08f5b65a44cbf"; OLD=ROOT/"runs/exp_376fca9354214097"
MODELS=("df","varcnn_direction"); TRAIN=(6238,1013,2024); SUPPORT=(1729,6238,20260916)
DATES=("day14","day90","day270"); SHOTS=(3,10)
METHODS=("G_source","proto_alpha_0.25","shrink_lambda_1.00","proto_alpha_0.50","shrink_lambda_0.10",
         "proto_alpha_0.75","shrink_lambda_0.01","current_prototype","G_current","support_selected_prototype",
         "support_selected_linear","support_selected_baseline")

def atomic(path,value):
    if path.exists(): raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); tmp.replace(path)

def path_for(m,t,d,s,k):
    return (OLD/"artifacts"/f"{m}_{d}_seed{s}_shot{k}.json") if t==6238 else (RUN/"artifacts"/f"{m}_trainseed{t}_{d}_supportseed{s}_shot{k}.json")

def mean_sd_range(v):
    a=np.asarray(v,float); return {"mean":float(a.mean()),"sd":float(a.std(ddof=1)),"min":float(a.min()),"max":float(a.max()),"range":float(a.max()-a.min()),"values":a.tolist()}

records={}; rows=[]
for m in MODELS:
  for t in TRAIN:
    for d in DATES:
      for s in SUPPORT:
        for k in SHOTS:
          o=json.loads(path_for(m,t,d,s,k).read_text()); records[m,t,d,s,k]=o
          for method in METHODS:
            met=o["metrics"][method]
            tr=None if method=="G_source" else o["transfers_vs_G_source"][method]
            rows.append({"architecture":m,"training_seed":t,"date":d,"shot":k,"support_seed":s,"method":method,
              "accuracy":met["accuracy"],"macro_precision":met["macro_precision"],"macro_recall":met["macro_recall"],"macro_f1":met["macro_f1"],
              "corrected":"" if tr is None else tr["corrected"],"harmed":"" if tr is None else tr["harmed"],"net":"" if tr is None else tr["net"]})

csv_path=RUN/"artifacts/metrics_long.csv"
with csv_path.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# Support-seed means used by preregistered judgments.
cells={}; judgments=[]
for m in MODELS:
  for t in TRAIN:
    for d in DATES:
      for k in SHOTS:
        key=f"{m}/train{t}/{d}/shot{k}"; cells[key]={}
        for method in METHODS:
          cells[key][method]={metric:mean_sd_range([records[m,t,d,s,k]["metrics"][method][metric] for s in SUPPORT]) for metric in ("accuracy","macro_f1")}
        if k==10:
          g=cells[key]["G_source"]["macro_f1"]["mean"]; cur=max(cells[key]["G_current"]["macro_f1"]["mean"],cells[key]["current_prototype"]["macro_f1"]["mean"])
          selected=cells[key]["support_selected_baseline"]["macro_f1"]["mean"]
          if d=="day14":
            passed=selected>=g-.01 and selected>=cur+.02
            judgments.append({"architecture":m,"training_seed":t,"date":d,"kind":"day14_protection","G_source":g,"stronger_current":cur,"selected":selected,
                "selected_minus_G_source":selected-g,"selected_minus_stronger_current":selected-cur,"passed":bool(passed)})
          else:
            R=cur-g; H=selected-g; rate=H/R if R>0 else None; passed=R>0 and rate>=.8
            judgments.append({"architecture":m,"training_seed":t,"date":d,"kind":"late_recovery","G_source":g,"stronger_current":cur,"selected":selected,
                "R":R,"H":H,"retention_rate":rate,"passed":bool(passed)})

# A: fixed support seed, compare training seeds; B: fixed training seed, compare support seeds.
training_randomness={}; support_randomness={}; two_factor={}
for m in MODELS:
  for d in DATES:
    for k in SHOTS:
      for method in METHODS:
        for metric in ("accuracy","macro_f1"):
          base=f"{m}/{d}/shot{k}/{method}/{metric}"
          for s in SUPPORT:
            training_randomness[f"{base}/support{s}"]=mean_sd_range([records[m,t,d,s,k]["metrics"][method][metric] for t in TRAIN])
          for t in TRAIN:
            support_randomness[f"{base}/train{t}"]=mean_sd_range([records[m,t,d,s,k]["metrics"][method][metric] for s in SUPPORT])
          a=np.asarray([[records[m,t,d,s,k]["metrics"][method][metric] for s in SUPPORT] for t in TRAIN],float)
          grand=a.mean(); tm=a.mean(1); sm=a.mean(0); fitted=grand+(tm-grand)[:,None]+(sm-grand)[None,:]
          ss_t=float(len(SUPPORT)*np.sum((tm-grand)**2)); ss_s=float(len(TRAIN)*np.sum((sm-grand)**2)); ss_e=float(np.sum((a-fitted)**2)); ss_total=float(np.sum((a-grand)**2))
          two_factor[base]={"cells":a.tolist(),"training_seeds":list(TRAIN),"support_seeds":list(SUPPORT),"grand_mean":float(grand),
            "ss_training":ss_t,"ss_support":ss_s,"ss_interaction_residual":ss_e,"ss_total":ss_total,
            "share_training":ss_t/ss_total if ss_total else 0.0,"share_support":ss_s/ss_total if ss_total else 0.0,
            "share_interaction_residual":ss_e/ss_total if ss_total else 0.0,"ms_training":ss_t/2,"ms_support":ss_s/2,"ms_interaction_residual":ss_e/4}

# Selection diagnostics, including the formally reused original seed.
noise={str(k):{"count":0,"oracle_match":0,"regrets":[],"selected":Counter()} for k in SHOTS}
for (m,t,d,s,k),o in records.items():
    q=noise[str(k)]; q["count"]+=1; oracle_record=o["posthoc_oracle"]["global"]
    selected=o["selected"]["global"] if "selected" in o else oracle_record["selected_candidate"]
    oracle=oracle_record["candidate"]; regret=oracle_record.get("regret_macro_f1",oracle_record.get("regret"))
    q["oracle_match"]+=selected==oracle; q["regrets"].append(regret); q["selected"][selected]+=1
for k,q in noise.items():
    q["oracle_match_rate"]=q["oracle_match"]/q["count"]; q["regret_macro_f1"]=mean_sd_range(q.pop("regrets")); q["selected"]=dict(q["selected"])

checkpoint_records=[]
for m in MODELS:
  oldcp=json.loads((ORIGINAL:=ROOT/"runs/exp_9121b664a1854097"/"artifacts"/f"{m}_history.json").read_text())
  best=max(oldcp,key=lambda r:(r["val_macro_f1"],-r["epoch"]))
  checkpoint_records.append({"architecture":m,"training_seed":6238,"best_epoch":best["epoch"],"source_validation_accuracy":best["val_accuracy"],"source_validation_macro_f1":best["val_macro_f1"],
    "checkpoint_path":f"runs/exp_9121b664a1854097/checkpoints/{m}_best.pt"})
  for t in (1013,2024): checkpoint_records.append(json.loads((RUN/"artifacts"/f"{m}_seed{t}_checkpoint.json").read_text()))

strict=all(x["passed"] for x in judgments)
source_quality={}
for m in MODELS:
    vals=[x["source_validation_macro_f1"] for x in checkpoint_records if x["architecture"]==m]; mean=float(np.mean(vals)); rng=max(vals)-min(vals)
    source_quality[m]={"macro_f1":mean_sd_range(vals),"range_exceeds_4pp":rng>.04,"any_deviation_from_mean_exceeds_2pp":any(abs(x-mean)>.02 for x in vals)}

summary={"new_backbone_training_runs":4,"architectures":list(MODELS),"training_seeds":list(TRAIN),"support_seeds":list(SUPPORT),
 "checkpoint_records":checkpoint_records,"support_seed_mean_cells":cells,"preregistered_10shot_judgments":judgments,
 "strict_replication_success":strict,"passed_cells":sum(x["passed"] for x in judgments),"total_primary_cells":len(judgments),
 "selection_noise":noise,"source_checkpoint_quality":source_quality,"metrics_long_rows":len(rows)}
atomic(RUN/"artifacts/summary.json",summary)
atomic(RUN/"artifacts/randomness_decomposition.json",{"backbone_training_randomness":training_randomness,"support_sampling_randomness":support_randomness,"two_factor":two_factor})
print(json.dumps({"strict_replication_success":strict,"passed_cells":summary["passed_cells"],"total":len(judgments),"noise":noise},indent=2))
