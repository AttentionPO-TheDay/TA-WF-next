#!/usr/bin/env python3
"""Training-free TemporalDrift burst structure diagnostic.

This program never imports torch, opens a checkpoint, fits a classifier, or
changes an input dataset. Large uncompressed NPZ members are memory-mapped.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/exp_bdca1fca83844035"
DATA = Path("/mnt/data2/ren/datasets/TemporalDrift")
SPLIT = ROOT / "runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json"
DATES = ("day0", "day14", "day30", "day90", "day150", "day270")
FILES = {"day0": "train.npz", **{f"day{x}": f"day{x}.npz" for x in (14, 30, 90, 150, 270)}}
L = 5000
SEED = 20260918
RESAMPLES = 200
EPS = 1e-9

ATTRS = (
    "valid_length", "padding_fraction", "burst_count", "transition_density",
    "outgoing_packet_fraction", "run_mean", "run_median", "run_p90", "run_max",
    "incoming_run_mean", "incoming_run_p90", "outgoing_run_mean", "outgoing_run_p90",
    "out_in_mean_log_ratio", "adjacent_log_length_corr", "adjacent_out_in_log_ratio",
)


def npz_memmap(path: Path, member: str) -> np.memmap:
    """Map an uncompressed .npy member without loading a multi-GB NPZ."""
    with path.open("rb") as raw, zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{member} is compressed; refusing large implicit load")
        raw.seek(info.header_offset)
        header = raw.read(30)
        sig, = struct.unpack("<I", header[:4])
        if sig != 0x04034B50:
            raise ValueError("bad local ZIP header")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        npy_start = info.header_offset + 30 + name_len + extra_len
        raw.seek(npy_start)
        version = np.lib.format.read_magic(raw)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(raw)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(raw)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_3_0(raw)
        data_offset = raw.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=data_offset,
                     shape=shape, order="F" if fortran else "C")


def signed_runs(direction: np.ndarray) -> np.ndarray:
    if direction.ndim != 1 or len(direction) == 0:
        return np.empty(0, dtype=np.int32)
    if np.any((direction != 1) & (direction != -1)):
        raise ValueError("runs require nonzero +/-1 directions")
    starts = np.r_[0, np.flatnonzero(direction[1:] != direction[:-1]) + 1]
    ends = np.r_[starts[1:], len(direction)]
    return (direction[starts].astype(np.int32) * (ends - starts).astype(np.int32))


def decode_runs(runs: np.ndarray) -> np.ndarray:
    if len(runs) == 0:
        return np.empty(0, dtype=np.int8)
    return np.repeat(np.sign(runs).astype(np.int8), np.abs(runs).astype(np.int64))


def q(values, p):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, p)) if len(a) else math.nan


def mean_or_nan(a):
    return float(np.mean(a)) if len(a) else math.nan


def run_attributes(runs: np.ndarray, observed_length: int) -> dict[str, float]:
    lengths = np.abs(runs).astype(np.float64)
    signs = np.sign(runs)
    incoming = lengths[signs < 0]
    outgoing = lengths[signs > 0]
    n = float(lengths.sum())
    result = {
        "valid_length": float(observed_length),
        "padding_fraction": float(1 - observed_length / L),
        "burst_count": float(len(runs)),
        "transition_density": float((len(runs)-1) / (n-1)) if n > 1 and len(runs) else 0.0,
        "outgoing_packet_fraction": float(outgoing.sum()/n) if n else math.nan,
        "run_mean": mean_or_nan(lengths), "run_median": q(lengths, .5),
        "run_p90": q(lengths, .9), "run_max": float(lengths.max()) if len(lengths) else math.nan,
        "incoming_run_mean": mean_or_nan(incoming), "incoming_run_p90": q(incoming, .9),
        "outgoing_run_mean": mean_or_nan(outgoing), "outgoing_run_p90": q(outgoing, .9),
        "out_in_mean_log_ratio": (float(np.log1p(outgoing.mean())-np.log1p(incoming.mean()))
                                   if len(incoming) and len(outgoing) else math.nan),
        "adjacent_log_length_corr": math.nan,
        "adjacent_out_in_log_ratio": math.nan,
    }
    if len(lengths) >= 3 and np.std(np.log1p(lengths[:-1])) > 0 and np.std(np.log1p(lengths[1:])) > 0:
        result["adjacent_log_length_corr"] = float(np.corrcoef(np.log1p(lengths[:-1]), np.log1p(lengths[1:]))[0, 1])
    ratios = []
    for i in range(len(runs)-1):
        if runs[i] > 0:
            ratios.append(np.log1p(lengths[i])-np.log1p(lengths[i+1]))
        else:
            ratios.append(np.log1p(lengths[i+1])-np.log1p(lengths[i]))
    if ratios:
        result["adjacent_out_in_log_ratio"] = float(np.mean(ratios))
    return result


def packet_position(direction: np.ndarray, runs: np.ndarray, windows=10):
    out = defaultdict(lambda: np.full(windows, np.nan, dtype=np.float64))
    run_starts = np.r_[0, np.cumsum(np.abs(runs))[:-1]] if len(runs) else np.empty(0, int)
    run_ends = run_starts + np.abs(runs)
    for b in range(windows):
        lo, hi = b * L // windows, (b+1) * L // windows
        if lo >= len(direction):
            continue
        d = direction[lo:min(hi, len(direction))]
        out["outgoing_fraction"][b] = np.mean(d > 0)
        out["transition_density"][b] = np.mean(d[1:] != d[:-1]) if len(d) > 1 else 0.0
        start_lengths = np.abs(runs)[(run_starts >= lo) & (run_starts < hi)]
        if len(start_lengths):
            out["start_run_length_mean"][b] = np.mean(start_lengths)
            out["start_run_length_median"][b] = np.median(start_lengths)
        overlap = np.maximum(0, np.minimum(run_ends, hi) - np.maximum(run_starts, lo))
        if overlap.sum():
            out["cover_run_length_mass_weighted"][b] = np.sum(overlap*np.abs(runs))/overlap.sum()
    return out


def burst_position(runs: np.ndarray, bins: int):
    out = defaultdict(lambda: np.full(bins, np.nan, dtype=np.float64))
    if not len(runs):
        return out
    lengths = np.abs(runs).astype(float); signs = np.sign(runs); total = lengths.sum()
    ids = np.minimum(bins-1, np.arange(len(runs))*bins//len(runs))
    for b in range(bins):
        take = ids == b
        if not take.any(): continue
        ll, ss = lengths[take], signs[take]
        out["packet_mass_fraction"][b] = ll.sum()/total
        out["outgoing_packet_fraction"][b] = ll[ss > 0].sum()/ll.sum()
        out["run_length_mean"][b] = ll.mean()
        if np.any(ss < 0): out["incoming_run_length_mean"][b] = ll[ss < 0].mean()
        if np.any(ss > 0): out["outgoing_run_length_mean"][b] = ll[ss > 0].mean()
    return out


def stable_hash(direction_padded: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(direction_padded, dtype=np.int8).tobytes()).hexdigest()


def site_centers(rows, attrs=ATTRS):
    grouped = defaultdict(list)
    for r in rows: grouped[int(r["site"])].append(r)
    centers = {}
    for site, rr in grouped.items():
        centers[site] = {a: q([x[a] for x in rr], .5) for a in attrs}
    return centers, grouped


def baseline_distributions(day0_grouped):
    result = defaultdict(dict)
    for site, rr in sorted(day0_grouped.items()):
        n = len(rr); rng = np.random.default_rng(SEED + site)
        for a in ATTRS:
            values = np.asarray([r[a] for r in rr], float); values = values[np.isfinite(values)]
            diffs=[]
            if len(values) >= 4:
                for _ in range(RESAMPLES):
                    perm=rng.permutation(len(values)); h=len(values)//2
                    diffs.append(abs(np.median(values[perm[:h]])-np.median(values[perm[h:2*h]])))
            result[site][a] = np.asarray(diffs, float)
    return result


def analyze_core(all_rows):
    centers={}; grouped={}
    for d in DATES: centers[d],grouped[d]=site_centers(all_rows[d])
    common_sites=sorted(set.intersection(*(set(centers[d]) for d in DATES)))
    if not common_sites:
        raise ValueError("no sites shared by all dates in this sensitivity")
    baseline=baseline_distributions(grouped["day0"])
    detail=[]; joint=[]
    for d in DATES[1:]:
        for site in common_sites:
            for a in ATTRS:
                b=baseline[site][a]; shift=abs(centers[d][site][a]-centers["day0"][site][a])
                detail.append({"site":site,"date":d,"attribute":a,
                    "day0_center":centers["day0"][site][a],"future_center":centers[d][site][a],
                    "temporal_abs_shift":shift,"day0_split_median_abs_diff":q(b,.5),
                    "day0_split_q95_abs_diff":q(b,.95),
                    "shift_over_day0_split_median":shift/max(q(b,.5),EPS),
                    "exceeds_day0_q95":int(shift>q(b,.95))})
        for a in ATTRS:
            same=np.asarray([x["temporal_abs_shift"] for x in detail if x["date"]==d and x["attribute"]==a],float)
            vals=np.asarray([centers[d][s][a] for s in common_sites],float); vals=vals[np.isfinite(vals)]
            diff=np.abs(vals[:,None]-vals[None,:]); between=diff[np.triu_indices(len(vals),1)]
            subset=[x for x in detail if x["date"]==d and x["attribute"]==a]
            joint.append({"attribute":a,"date":d,"sites":len(vals),
                "median_same_site_temporal_shift":q(same,.5),
                "median_different_site_distance":q(between,.5),
                "separability_ratio":q(between,.5)/max(q(same,.5),EPS),
                "fraction_not_exceeding_day0_q95":float(np.mean([not x["exceeds_day0_q95"] for x in subset])),
                "common_site_coverage":len(common_sites)/102})
    for a in ATTRS:
        rr=[r for r in joint if r["attribute"]==a]
        joint.append({"attribute":a,"date":"ALL","sites":len(common_sites),
            "median_same_site_temporal_shift":q([r["median_same_site_temporal_shift"] for r in rr],.5),
            "median_different_site_distance":q([r["median_different_site_distance"] for r in rr],.5),
            "separability_ratio":q([r["separability_ratio"] for r in rr],.5),
            "fraction_not_exceeding_day0_q95":float(np.mean([r["fraction_not_exceeding_day0_q95"] for r in rr])),
            "common_site_coverage":len(common_sites)/102})
    return detail,joint,centers,grouped


def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--self-test",action="store_true"); parser.add_argument("--resume",action="store_true"); args=parser.parse_args()
    if args.self_test:
        for d in (np.array([1,1,-1,-1,-1,1],np.int8),np.array([-1],np.int8)):
            r=signed_runs(d); assert np.array_equal(decode_runs(r),d); assert np.all(np.sign(r[1:])!=np.sign(r[:-1]))
        assert np.array_equal(signed_runs(np.array([1,1,-1],np.int8)),np.array([2,-1]))
        print("self-test passed"); return
    started=time.time(); split=json.loads(SPLIT.read_text())
    day0_indices=sorted(sum((split["roles"][r]["indices"] for r in ("supervised_train","reference","source_holdout")),[]))
    all_rows={}; packet_pos={}; burst_pos={5:{},10:{},20:{}}; audits={}; roundtrip={}
    timing={}; sensitivity={"exclude_terminal":{},"equal500":{}}
    for date in DATES:
        path=DATA/FILES[date]; x=npz_memmap(path,"X.npy"); y=npz_memmap(path,"y.npy")
        candidates=day0_indices if date=="day0" else range(len(y)); seen={}; canonical=[]; conflicts=[]
        zero_stats=Counter(); label_counts=Counter(); raw_label_counts=Counter(map(int,y))
        for i in candidates:
            row=x[i,:L]; finite=np.isfinite(row); nz=finite & (row!=0); last=int(np.flatnonzero(nz)[-1])+1 if nz.any() else 0
            direction=np.zeros(L,np.int8); direction[:last]=np.sign(row[:last]).astype(np.int8)
            digest=stable_hash(direction)
            if digest in seen:
                if int(y[seen[digest]])!=int(y[i]): conflicts.append((seen[digest],int(i)))
                continue
            seen[digest]=int(i); canonical.append(int(i))
        rows=[]; pstore=[]; bstores={k:[] for k in (5,10,20)}; rt_fail=[]; time_acc=Counter(); time_values=defaultdict(list)
        exclude_rows=[]; equal_rows=[]
        for i in canonical:
            raw=x[i]; prefix=raw[:L]; finite=np.isfinite(prefix); nz=finite & (prefix!=0)
            where=np.flatnonzero(nz); valid=int(where[-1])+1 if len(where) else 0
            leading=int(valid>0 and prefix[0]==0); interior=int(np.any(prefix[:valid]==0)) if valid else 0
            nonfinite=int(np.sum(~finite)); zero_stats.update(all_zero=int(valid==0),leading_zero=leading,interior_zero=interior,nonfinite=nonfinite)
            if interior or nonfinite or valid==0:
                # Do not bridge unknown positions; record and conservatively skip structural interpretation.
                continue
            direction=np.sign(prefix[:valid]).astype(np.int8); runs=signed_runs(direction); decoded=decode_runs(runs)
            ok=np.array_equal(decoded,direction) and int(np.abs(runs).sum())==valid and np.all(np.sign(runs[1:])!=np.sign(runs[:-1]))
            if not ok: rt_fail.append(i); continue
            censored=bool(valid==L and raw.shape[0]>L and np.isfinite(raw[L]) and raw[L]!=0 and np.sign(raw[L])==direction[-1])
            boundary_transition=bool(valid==L and raw.shape[0]>L and raw[L]!=0 and np.sign(raw[L])!=direction[-1])
            base=run_attributes(runs,valid); base.update(site=int(y[i]),row_index=i,date=date,terminal_censored=int(censored),boundary_transition=int(boundary_transition))
            rows.append(base); label_counts[int(y[i])]+=1; pstore.append(packet_position(direction,runs,10))
            for k in (5,10,20): bstores[k].append(burst_position(runs,k))
            eruns=runs[:-1] if censored else runs
            e=run_attributes(eruns,int(np.abs(eruns).sum())); e.update(site=int(y[i]),row_index=i,date=date,terminal_censored=int(censored)); exclude_rows.append(e)
            if valid>=500:
                rr=signed_runs(direction[:500]); z=run_attributes(rr,500); z.update(site=int(y[i]),row_index=i,date=date,terminal_censored=int(valid>500 and direction[499]==direction[500])); equal_rows.append(z)
            # Timing audit is separate from all direction-only outputs.
            t=np.abs(raw[:valid]); time_acc["packets"]+=valid; time_acc["monotonic_violations"]+=int(np.sum(np.diff(t)<0)); time_acc["nonfinite"]+=int(np.sum(~np.isfinite(t)))
            starts=np.r_[0,np.cumsum(np.abs(runs))[:-1]]; ends=starts+np.abs(runs)-1
            durations=t[ends]-t[starts]; gaps=t[starts[1:]]-t[ends[:-1]] if len(runs)>1 else np.empty(0)
            time_acc["bursts"]+=len(runs); time_acc["zero_duration_bursts"]+=int(np.sum(durations<=0)); time_acc["single_packet_bursts"]+=int(np.sum(np.abs(runs)==1))
            time_values["duration_median"].append(q(durations,.5)); time_values["gap_median"].append(q(gaps,.5))
            positive=durations>0
            if positive.any(): time_values["finite_rate_median"].append(q(np.abs(runs)[positive]/durations[positive],.5))
        all_rows[date]=rows; packet_pos[date]=pstore
        for k in (5,10,20): burst_pos[k][date]=bstores[k]
        sensitivity["exclude_terminal"][date]=exclude_rows; sensitivity["equal500"][date]=equal_rows
        audits[date]={"file":str(path),"X_shape":list(x.shape),"X_dtype":str(x.dtype),"y_shape":list(y.shape),
            "raw_rows":len(y),"candidate_rows":len(list(candidates)) if date!="day0" else len(day0_indices),"canonical_rows":len(canonical),"analyzed_rows":len(rows),
            "duplicate_rows_removed":len(list(candidates))-len(canonical) if date!="day0" else len(day0_indices)-len(canonical),
            "cross_label_duplicate_groups":len(conflicts),"zero_anomalies":dict(zero_stats),"per_site_raw":dict(sorted(raw_label_counts.items())),"per_site_canonical":dict(sorted(label_counts.items())),
            "valid_length_quantiles":[q([r["valid_length"] for r in rows],p) for p in (0,.1,.25,.5,.75,.9,1)],
            "full_L_count":sum(r["valid_length"]==L for r in rows),"terminal_censored_count":sum(r["terminal_censored"] for r in rows),"boundary_transition_count":sum(r["boundary_transition"] for r in rows)}
        roundtrip[date]={"checked":len(rows),"failed":len(rt_fail),"failure_rows":rt_fail[:20],"all_passed":not rt_fail}
        timing[date]={**dict(time_acc),"zero_duration_burst_fraction":time_acc["zero_duration_bursts"]/max(time_acc["bursts"],1),
            "single_packet_burst_fraction":time_acc["single_packet_bursts"]/max(time_acc["bursts"],1),
            **{k:q(v,.5) for k,v in time_values.items()}}
        del x,y
        print(date,audits[date]["analyzed_rows"],"rows",flush=True)
    if not all(x["all_passed"] for x in roundtrip.values()): raise RuntimeError("RLE round-trip failed")

    detail,joint,centers,grouped=analyze_core(all_rows)
    write_csv(RUN/"day0_internal_vs_temporal_shift.csv",detail); write_csv(RUN/"stability_discriminability_joint.csv",joint)
    # Trace features make every aggregate auditable without exposing a learned object.
    if not args.resume or not (RUN/"artifacts/trace_direction_features.csv").exists():
        write_csv(RUN/"artifacts/trace_direction_features.csv",[r for d in DATES for r in all_rows[d]])

    # Position summaries. Day0 split noise uses 200 deterministic partitions per site/metric/bin.
    position_rows=[]
    def position_family(store,coord,k):
        metrics=sorted({m for d in DATES for tr in store[d] for m in tr})
        for metric in metrics:
            for b in range(k):
                dvals={}
                for d in DATES:
                    by=defaultdict(list)
                    for r,tr in zip(all_rows[d],store[d]):
                        v=tr[metric][b]
                        if np.isfinite(v): by[int(r["site"])].append(float(v))
                    dvals[d]={s:q(v,.5) for s,v in by.items()}
                for d in DATES[1:]:
                    for site in range(102):
                        vals=np.asarray([tr[metric][b] for r,tr in zip(all_rows["day0"],store["day0"]) if int(r["site"])==site],float); vals=vals[np.isfinite(vals)]
                        diffs=[]; rng=np.random.default_rng(SEED+site+b*1009+sum(map(ord,metric)))
                        if len(vals)>=4:
                            for _ in range(RESAMPLES):
                                p=rng.permutation(len(vals)); h=len(vals)//2; diffs.append(abs(np.median(vals[p[:h]])-np.median(vals[p[h:2*h]])))
                        shift=abs(dvals[d].get(site,math.nan)-dvals["day0"].get(site,math.nan))
                        position_rows.append({"coordinate":coord,"bins":k,"bin":b,"metric":metric,"site":site,"date":d,
                            "day0_center":dvals["day0"].get(site,math.nan),"future_center":dvals[d].get(site,math.nan),"temporal_abs_shift":shift,
                            "day0_split_median_abs_diff":q(diffs,.5),"day0_split_q95_abs_diff":q(diffs,.95),
                            "shift_over_day0_split_median":shift/max(q(diffs,.5),EPS),"exceeds_day0_q95":int(shift>q(diffs,.95)) if np.isfinite(shift) and len(diffs) else ""})
    if not args.resume or not (RUN/"position_direction_diagnostic.csv").exists():
        position_family(packet_pos,"raw_packet_index",10); position_family(burst_pos[10],"relative_burst_order",10)
        write_csv(RUN/"position_direction_diagnostic.csv",position_rows)

    # Sensitivity repeats core attributes and records fixed-bin aggregate shifts.
    sens_summary={}
    for name,sets in sensitivity.items():
        sd,sj,_,_=analyze_core(sets); write_csv(RUN/f"artifacts/{name}_joint.csv",sj)
        sens_summary[name]={"rows":{d:len(sets[d]) for d in DATES},"joint_all":{r["attribute"]:r for r in sj if r["date"]=="ALL"}}
    # Full-L and fixed Day0 length strata are filtering sensitivities.
    full={d:[r for r in all_rows[d] if r["valid_length"]==L] for d in DATES}
    if min(len({r['site'] for r in full[d]}) for d in DATES) == 102:
        _,fj,_,_=analyze_core(full); write_csv(RUN/"artifacts/full_L_joint.csv",fj); sens_summary["full_L"]={"rows":{d:len(full[d]) for d in DATES},"joint_all":{r["attribute"]:r for r in fj if r["date"]=="ALL"}}
    else:
        sens_summary["full_L"]={"rows":{d:len(full[d]) for d in DATES},"not_comparable":"not all 102 sites represented in every date"}
    length_edges=np.quantile([r["valid_length"] for r in all_rows["day0"]],[.25,.5,.75]).tolist()
    sens_summary["day0_length_quartile_edges"]=length_edges
    sens_summary["length_strata_counts"]={d:[sum(([-math.inf]+length_edges)[j] < r["valid_length"] <= (length_edges+[math.inf])[j] for r in all_rows[d]) for j in range(4)] for d in DATES}
    bin_sens={}
    for k in (5,10,20):
        shifts=[]
        for d in DATES[1:]:
            for metric in sorted({m for tr in burst_pos[k][d] for m in tr}):
                for b in range(k):
                    for site in range(102):
                        a=[tr[metric][b] for r,tr in zip(all_rows["day0"],burst_pos[k]["day0"]) if int(r["site"])==site]
                        z=[tr[metric][b] for r,tr in zip(all_rows[d],burst_pos[k][d]) if int(r["site"])==site]
                        if np.isfinite(q(a,.5)) and np.isfinite(q(z,.5)): shifts.append(abs(q(a,.5)-q(z,.5)))
        bin_sens[str(k)]={"median_site_date_bin_shift":q(shifts,.5),"comparisons":len(shifts)}
    sens_summary["burst_bin_sensitivity"]=bin_sens

    candidates=[]
    for a in ATTRS:
        rr=[r for r in joint if r["attribute"]==a and r["date"]!="ALL"]
        stable=float(np.mean([r["fraction_not_exceeding_day0_q95"] for r in rr]))>=.8
        discr=sum(r["separability_ratio"]>=2 for r in rr)>=4
        if stable and discr: candidates.append(a)
    summary={"schema_version":1,"experiment":"exp_bdca1fca83844035","completed_at_unix":time.time(),"elapsed_seconds":time.time()-started,
        "training_runs":0,"adaptation_runs":0,"gpu_used":False,"observation_length":L,"analysis_seed":SEED,"day0_resamples":RESAMPLES,
        "input_hashes":split["source_files_sha256"],"audits":audits,"roundtrip":roundtrip,"timing":timing,"candidate_attributes_primary":candidates,
        "future_label_dependency":"same-site/different-site descriptive statistics and candidate ranking use future truth labels; no deployable rule was fitted",
        "sensitivity":sens_summary}
    (RUN/"artifacts").mkdir(exist_ok=True); (RUN/"artifacts/summary_raw.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"candidate_attributes":candidates,"elapsed_seconds":summary["elapsed_seconds"]},indent=2))


if __name__ == "__main__": main()
