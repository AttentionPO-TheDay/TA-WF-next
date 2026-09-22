#!/usr/bin/env python3
"""Preregistered posthoc site repeatability and visible-score separation."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[3]; RUN=ROOT/"runs/exp_a2ffb5623ad346c7"; ART=RUN/"artifacts"
MODELS=("df","varcnn_direction"); DATES=("day14","day90","day270"); SEEDS=(1729,6238,20260916); SHOTS=(3,10)
METHODS=("prototype_interpolation","shrink_to_source")

def auc(harmed,corrected):
    values=np.concatenate([harmed,corrected]); ranks=rankdata(values,method="average"); n1=len(harmed); n0=len(corrected)
    return float((ranks[:n1].sum()-n1*(n1+1)/2)/(n1*n0))

def main():
    output=ART/"posthoc_repeatability.json"
    if output.exists(): raise FileExistsError(output)
    with (ART/"per_website_transfers.csv").open(newline="") as h: rows=list(csv.DictReader(h))
    records={(m,d,s,k):json.loads((ART/f"{m}_{d}_seed{s}_shot{k}.json").read_text()) for m in MODELS for d in DATES for s in SEEDS for k in SHOTS}
    result={}
    for method in METHODS:
        aggregate=defaultdict(lambda:{"corrected":0,"harmed":0,"net":0})
        for row in rows:
            if row["method"]==method:
                a=aggregate[int(row["website"])]
                for key in a: a[key]+=int(row[key])
        repeated={"benefit":[],"harm":[]}
        for site in range(102):
            for direction,positive in (("benefit",True),("harm",False)):
                by_model={}
                for model in MODELS:
                    date_flags=[]
                    for date in DATES:
                        seed_flags=[]
                        for seed in SEEDS:
                            net=sum(next(int(r["net"]) for r in rows if r["model"]==model and r["date"]==date and int(r["seed"])==seed and int(r["shot"])==shot and r["method"]==method and int(r["website"])==site) for shot in SHOTS)
                            seed_flags.append(net>0 if positive else net<0)
                        date_flags.append(sum(seed_flags)>=2)
                    by_model[model]=int(sum(date_flags))
                if all(v>=2 for v in by_model.values()): repeated[direction].append({"website":site,"qualifying_dates_by_backbone":by_model,"aggregate":aggregate[site]})
        fields=["G_source_margin",f"{method}_margin"]
        if method=="prototype_interpolation": fields.append("prototype_interpolation_max_cosine")
        visible={field:{model:[] for model in MODELS} for field in fields}
        for (model,date,seed,shot),r in records.items():
            truth=np.asarray(r["query_truth"]); base=np.asarray(r["predictions"]["G_source"])==truth; current=np.asarray(r["predictions"][method])==truth
            harmed=base&~current; corrected=~base&current
            for field in fields:
                values=np.asarray(r["prediction_time_diagnostics"][field])
                if harmed.any() and corrected.any(): visible[field][model].append(auc(values[harmed],values[corrected]))
        visible_summary={field:{model:{"harm_vs_corrected_auc_mean":float(np.mean(v)),"sd":float(np.std(v,ddof=1)),"min":float(np.min(v)),"max":float(np.max(v)),"configurations":len(v)} for model,v in by.items()} for field,by in visible.items()}
        alignment={}
        harm_ids={x["website"] for x in repeated["harm"]}
        for model in MODELS:
            harm_values=[]; other_values=[]
            for (m,date,seed,shot),r in records.items():
                if m!=model: continue
                values=np.asarray(r["source_current_prototype_cosine_by_class"])
                harm_values.extend(values[list(harm_ids)].tolist()); other_values.extend(values[[i for i in range(102) if i not in harm_ids]].tolist())
            alignment[model]={"repeated_harm_sites_mean":float(np.mean(harm_values)) if harm_values else None,
                              "other_sites_mean":float(np.mean(other_values)) if other_values else None}
        result[method]={"repeatability_rule":"net sign in >=2/3 seeds after summing shots; >=2/3 dates in both backbones",
            "repeated_sites":repeated,
            "top_aggregate_corrected":sorted(({"website":k,**v} for k,v in aggregate.items()),key=lambda x:(-x["corrected"],x["website"]))[:15],
            "top_aggregate_harmed":sorted(({"website":k,**v} for k,v in aggregate.items()),key=lambda x:(-x["harmed"],x["website"]))[:15],
            "top_aggregate_net_gain":sorted(({"website":k,**v} for k,v in aggregate.items()),key=lambda x:(-x["net"],x["website"]))[:15],
            "top_aggregate_net_harm":sorted(({"website":k,**v} for k,v in aggregate.items()),key=lambda x:(x["net"],x["website"]))[:15],
            "visible_harm_vs_corrected_auc":visible_summary,"prototype_alignment":alignment}
    tmp=output.with_suffix(".json.tmp"); tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); tmp.replace(output); print(output)

if __name__=="__main__": main()
