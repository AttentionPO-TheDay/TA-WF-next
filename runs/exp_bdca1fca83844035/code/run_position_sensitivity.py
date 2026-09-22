#!/usr/bin/env python3
"""Fixed 500-packet position sensitivity; no learning or model code."""
from __future__ import annotations
import csv, importlib.util, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve(); spec=importlib.util.spec_from_file_location("burst",HERE.with_name("run_burst_diagnostic.py")); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
RUN=b.RUN; BUDGET=500; K=10

def packet_pos(direction,runs):
    out=defaultdict(lambda:np.full(K,np.nan)); starts=np.r_[0,np.cumsum(np.abs(runs))[:-1]]; ends=starts+np.abs(runs)
    for j in range(K):
        lo,hi=j*BUDGET//K,(j+1)*BUDGET//K
        if lo>=len(direction): continue
        d=direction[lo:min(hi,len(direction))]
        out["outgoing_fraction"][j]=np.mean(d>0); out["transition_density"][j]=np.mean(d[1:]!=d[:-1]) if len(d)>1 else 0.0
        take=(starts>=lo)&(starts<hi)
        if take.any(): out["start_run_length_mean"][j]=np.mean(np.abs(runs)[take])
        overlap=np.maximum(0,np.minimum(ends,hi)-np.maximum(starts,lo))
        if overlap.sum(): out["cover_run_length_mass_weighted"][j]=np.sum(overlap*np.abs(runs))/overlap.sum()
    return out

def write(rows):
    p=RUN/"artifacts/position_equal500_sensitivity.csv"
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    split=json.loads(b.SPLIT.read_text()); day0=sorted(sum((split["roles"][r]["indices"] for r in ("supervised_train","reference","source_holdout")),[]))
    values=defaultdict(list); counts=defaultdict(int)
    for date in b.DATES:
        x=b.npz_memmap(b.DATA/b.FILES[date],"X.npy"); y=b.npz_memmap(b.DATA/b.FILES[date],"y.npy"); indices=day0 if date=="day0" else range(len(y))
        for i in indices:
            row=x[i,:b.L]; where=np.flatnonzero(np.isfinite(row)&(row!=0)); valid=int(where[-1])+1 if len(where) else 0
            if valid<BUDGET: continue
            d=np.sign(row[:BUDGET]).astype(np.int8); runs=b.signed_runs(d); counts[(date,int(y[i]))]+=1
            scenarios=(("equal500",d,runs),("equal500_exclude_terminal",b.decode_runs(runs[:-1]),runs[:-1]))
            for scenario,sd,sr in scenarios:
                if not len(sr): continue
                for coord,obj in (("raw_packet_index",packet_pos(sd,sr)),("relative_burst_order",b.burst_position(sr,K))):
                    for metric,a in obj.items():
                        for j,v in enumerate(a):
                            if np.isfinite(v): values[(scenario,date,int(y[i]),coord,metric,j)].append(float(v))
        del x,y
        print(date,sum(counts[(date,s)] for s in range(102)),flush=True)
    keys=sorted({(scenario,coord,metric,j) for scenario,date,site,coord,metric,j in values}); common=sorted(set.intersection(*({s for s in range(102) if counts[(d,s)]} for d in b.DATES)))
    rows=[]
    for scenario,coord,metric,j in keys:
        for date in b.DATES[1:]:
            future_centers=[]; shifts=[]; stable=[]
            for site in common:
                a=np.asarray(values[(scenario,"day0",site,coord,metric,j)]); z=np.asarray(values[(scenario,date,site,coord,metric,j)])
                if not len(a) or not len(z): continue
                rng=np.random.default_rng(b.SEED+site+j*1009+sum(map(ord,metric))); diffs=[]
                for _ in range(b.RESAMPLES):
                    p=rng.permutation(len(a)); h=len(a)//2
                    if h: diffs.append(abs(np.median(a[p[:h]])-np.median(a[p[h:2*h]])))
                shift=abs(np.median(z)-np.median(a)); shifts.append(shift); stable.append(shift<=b.q(diffs,.95)); future_centers.append(np.median(z))
            fc=np.asarray(future_centers); between=np.abs(fc[:,None]-fc[None,:])[np.triu_indices(len(fc),1)]
            rows.append({"scenario":scenario,"coordinate":coord,"bins":K,"bin":j,"metric":metric,"date":date,"common_sites":len(common),
                "median_same_site_temporal_shift":b.q(shifts,.5),"median_different_site_distance":b.q(between,.5),
                "separability_ratio":b.q(between,.5)/max(b.q(shifts,.5),b.EPS),"fraction_not_exceeding_day0_q95":float(np.mean(stable))})
    write(rows); print(json.dumps({"rows":len(rows),"common_sites":len(common),"counts":{d:sum(counts[(d,s)] for s in range(102)) for d in b.DATES}}))
if __name__=="__main__": main()
