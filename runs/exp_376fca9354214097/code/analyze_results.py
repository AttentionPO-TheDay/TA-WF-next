#!/usr/bin/env python3
"""Preregistered posthoc attribution; never changes predictions."""
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]; RUN=ROOT/"runs/exp_376fca9354214097"; ART=RUN/"artifacts"
MODELS=("df","varcnn_direction"); DATES=("day14","day90","day270"); SHOTS=(3,10)
records=[json.loads(Path(p).read_text()) for p in glob.glob(str(ART/"*_day*_seed*_shot*.json"))]
if len(records)!=36: raise ValueError(len(records))

def avg(rs,method,metric): return float(np.mean([r["metrics"][method][metric] for r in rs]))

out={"selection_frequencies":{},"oracle_envelope":{},"fixed_to_selected":{},"noise_by_cell":{}}
for shot in SHOTS:
    out["selection_frequencies"][str(shot)]={}
    for model in MODELS:
        out["selection_frequencies"][str(shot)][model]={}
        for date in DATES:
            rs=[r for r in records if r["shot"]==shot and r["model"]==model and r["date"]==date]
            out["selection_frequencies"][str(shot)][model][date]={fam:dict(Counter(r["cv"]["selected"][fam] for r in rs)) for fam in ("prototype","linear","global")}

for model in MODELS:
    out["oracle_envelope"][model]={}; out["fixed_to_selected"][model]={}; out["noise_by_cell"][model]={}
    for date in DATES:
        out["oracle_envelope"][model][date]={}; out["fixed_to_selected"][model][date]={}; out["noise_by_cell"][model][date]={}
        for shot in SHOTS:
            rs=[r for r in records if r["shot"]==shot and r["model"]==model and r["date"]==date]
            oracle=float(np.mean([r["posthoc_oracle"]["global"]["macro_f1"] for r in rs]))
            out["oracle_envelope"][model][date][str(shot)]={"macro_f1":oracle}
            out["fixed_to_selected"][model][date][str(shot)]={}
            for metric in ("accuracy","macro_f1"):
                selected=avg(rs,"support_selected_baseline",metric); alpha=avg(rs,"proto_alpha_0.25",metric); lam=avg(rs,"shrink_lambda_0.10",metric)
                out["fixed_to_selected"][model][date][str(shot)][metric]={"selected":selected,"fixed_alpha":alpha,"fixed_lambda":lam,
                    "selected_minus_fixed_alpha":selected-alpha,"selected_minus_fixed_lambda":selected-lam,"selected_minus_better_fixed":selected-max(alpha,lam)}
            out["noise_by_cell"][model][date][str(shot)]={fam:{
                "oracle_match":sum(r["cv"]["selected"][fam]==r["posthoc_oracle"][fam]["candidate"] for r in rs),
                "mean_regret":float(np.mean([r["posthoc_oracle"][fam]["regret"] for r in rs])),
                "substantial_misselections":sum(r["cv"]["selected"][fam]!=r["posthoc_oracle"][fam]["candidate"] and r["posthoc_oracle"][fam]["regret"]>.01 for r in rs)} for fam in ("prototype","linear","global")}

# Apply the exact preregistered 10-shot thresholds to the per-configuration query oracle envelope.
oracle_judgment={"by_backbone":{}}
for model in MODELS:
    d14=[r for r in records if r["model"]==model and r["date"]=="day14" and r["shot"]==10]
    oracle14=float(np.mean([r["posthoc_oracle"]["global"]["macro_f1"] for r in d14])); gs=avg(d14,"G_source","macro_f1")
    best=max(avg(d14,"G_current","macro_f1"),avg(d14,"current_prototype","macro_f1"))
    protection={"oracle_macro_f1":oracle14,"vs_G_source":oracle14-gs,"vs_best_current":oracle14-best}
    protection["passes"]=protection["vs_G_source"]>=-.01 and protection["vs_best_current"]>=.02
    late=[]
    for date in ("day90","day270"):
        rs=[r for r in records if r["model"]==model and r["date"]==date and r["shot"]==10]
        gs=avg(rs,"G_source","macro_f1"); recovery=max(avg(rs,"G_current","macro_f1"),avg(rs,"current_prototype","macro_f1"))-gs
        oracle=float(np.mean([r["posthoc_oracle"]["global"]["macro_f1"] for r in rs]))-gs
        late.append({"date":date,"current_recovery":recovery,"oracle_recovery":oracle,"retention":oracle/recovery if recovery>0 else None})
    late_pass=all(v["retention"]>=.8 for v in late if v["retention"] is not None)
    oracle_judgment["by_backbone"][model]={"day14":protection,"late":late,"late_passes":late_pass}
oracle_judgment["passes_both_day14"]=all(v["day14"]["passes"] for v in oracle_judgment["by_backbone"].values())
oracle_judgment["passes_both_late"]=all(v["late_passes"] for v in oracle_judgment["by_backbone"].values())
oracle_judgment["solves_tradeoff"]=oracle_judgment["passes_both_day14"] and oracle_judgment["passes_both_late"]
selected=json.loads((ART/"summary.json").read_text())["pre_registered_10shot_judgment"]
if selected["solves_tradeoff"]: attribution="none_selected_passed"
elif oracle_judgment["solves_tradeoff"]: attribution="A_selection_failure"
else: attribution="B_candidate_family_failure_or_mixed"
out["oracle_10shot_judgment"]=oracle_judgment; out["failure_attribution"]=attribution

path=ART/"posthoc_analysis.json"
if path.exists(): raise FileExistsError(path)
tmp=path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); tmp.replace(path)
print(json.dumps({"oracle_judgment":oracle_judgment,"failure_attribution":attribution},indent=2))
