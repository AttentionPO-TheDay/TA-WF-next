#!/usr/bin/env python3
"""Prepare, smoke, train and evaluate a frozen TemporalDrift screening run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ta_wf_next.models import DF, VarCNNDirection
from ta_wf_next.screening import (LOCAL_SPECS, LEGACY_LOCAL_SPECS, TraceDataset, atomic_json,
    classification_metrics, classify_features, indices_sha256, load_npz,
    masked_descriptors, classify_masked, valid_local_positions, seed_everything, sha256_file)

PROTOCOL_RUN = ROOT / "runs" / "exp_6238dacf9aa142cc"
RUN = PROTOCOL_RUN
DATA = Path("/mnt/data2/ren/datasets/TemporalDrift")
SPLIT = PROTOCOL_RUN / "artifacts" / "splits_v3.json"


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")


def prepare() -> None:
    output = SPLIT
    layer_output = RUN / "artifacts" / "local_layer_spec_v3.json"
    refuse_existing(output); refuse_existing(layer_output)
    source_x, labels = load_npz(DATA / "train.npz")
    valid_x,_valid_labels = load_npz(DATA / "valid.npz")
    valid_hashes={hashlib.sha256(np.sign(row[:5000]).astype(np.int8).tobytes()).hexdigest() for row in valid_x}
    canonical=[]; duplicate_indices=[]; seen={}
    validation_overlap_indices=[]
    for index,row in enumerate(source_x):
        digest=hashlib.sha256(np.sign(row[:5000]).astype(np.int8).tobytes()).hexdigest()
        if digest in valid_hashes: validation_overlap_indices.append(index)
        elif digest in seen: duplicate_indices.append(index)
        else: seen[digest]=index; canonical.append(index)
    rng = np.random.default_rng(6238)
    roles: dict[str, list[int]] = {"supervised_train": [], "reference": [], "source_holdout": []}
    per_class = {}
    for label in range(102):
        candidates = np.asarray([i for i in canonical if labels[i] == label],dtype=np.int64)
        shuffled = rng.permutation(candidates)
        if len(shuffled) < 23:
            raise ValueError(f"Class {label} has only {len(shuffled)} source rows")
        roles["reference"].extend(map(int, shuffled[:2]))
        roles["source_holdout"].extend(map(int, shuffled[2:22]))
        roles["supervised_train"].extend(map(int, shuffled[22:]))
        per_class[str(label)] = {key: int(sum(labels[i] == label for i in values)) for key, values in roles.items()}
    for values in roles.values(): values.sort()
    sets = {key: set(values) for key, values in roles.items()}
    union = set().union(*sets.values())
    pairwise = {f"{a}__{b}": len(sets[a] & sets[b]) for i,a in enumerate(sets) for b in list(sets)[i+1:]}
    if any(pairwise.values()) or union != set(canonical):
        raise AssertionError("Split is not a disjoint exhaustive partition of canonical rows")
    files = ["train.npz", "valid.npz", "day14.npz", "day30.npz", "day90.npz", "day150.npz", "day270.npz"]
    manifest = {
        "schema_version": 3, "created_before_future_scoring": True,
        "sample_id": "<source NPZ filename>:<zero-based row index>",
        "rule": "exclude train rows whose SHA-256 of int8 sign(X[:5000]) occurs in official valid; group remaining rows by the same hash and retain minimum row index; within each integer class 0..101, numpy default_rng(6238).permutation; first 2 reference, next 20 source_holdout, remainder supervised_train; numeric-sort each role",
        "excluded_source_validation_overlap_indices": validation_overlap_indices,
        "excluded_source_validation_overlap_count": len(validation_overlap_indices),
        "excluded_duplicate_indices": duplicate_indices,
        "excluded_duplicate_count": len(duplicate_indices),
        "official_valid_role": "source-only checkpoint selection; no rows moved into other roles",
        "future_role": "prediction followed by development scoring and post-hoc diagnostics only",
        "source_files_sha256": {name: sha256_file(DATA / name) for name in files},
        "roles": {key: {"source": "train.npz", "indices": values, "indices_sha256": indices_sha256(values), "count": len(values)} for key,values in roles.items()},
        "pairwise_intersection_counts": pairwise, "union_count": len(union),
        "all_102_classes_in_each_role": all(len(set(labels[values])) == 102 for values in roles.values()),
        "per_class_counts": per_class,
        "provenance_limit": "NPZ contains only X/y; row IDs prove index separation but not session-level independence. Exact duplicates are audited separately.",
    }
    atomic_json(output, manifest)
    layer = {
        "input_length": 5000, "padding_policy": "input zero padding is excluded by retaining only positions whose theoretical RF is fully within indices 0..4999",
        "interpretation_warning": "Regions aggregate highly overlapping convolutional receptive fields; they are region evidence, not independent web resources.",
        "region_rule": "ordered valid positions split by torch.tensor_split into 4 groups; mean channels then L2 normalize",
        "models": LEGACY_LOCAL_SPECS,
        "derivation": {
            "df": "standard RF recurrence across 8 conv(k=8,s=1,p=4) and 4 maxpool(k=8,s=4,p=0) operations; (RF,jump,start)=(1786,256,213)",
            "varcnn_direction": "standard RF recurrence from pad3+conv7/s2, maxpool3/s2/p1 and 8 residual blocks; max main-path RF gives (1755,32,0.5)",
        },
    }
    atomic_json(layer_output, layer)
    print(json.dumps({"splits": str(output), "counts": {k:len(v) for k,v in roles.items()}, "intersections": pairwise}, indent=2))


def model_for(name: str) -> torch.nn.Module:
    if name == "df": return DF(102)
    if name == "varcnn_direction": return VarCNNDirection(102)
    raise ValueError(name)


def frozen_outputs(model: torch.nn.Module, name: str, x: torch.Tensor):
    """Compute local map, embedding and logits with exactly one encoder pass."""
    captured = []
    layer = model.feature_extraction[0] if name == "df" else model.dir_encoder.convs[0]
    handle = layer.register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        logits, features = model(x)
    finally:
        handle.remove()
    return logits, features, captured[0]



def loaders(indices: np.ndarray, batch_size: int, *, x=None, y=None, path=None, shuffle=False):
    if x is None: x,y = load_npz(path)
    generator = torch.Generator().manual_seed(6238)
    return DataLoader(TraceDataset(x,y,indices), batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator), x, y


def smoke() -> None:
    output = RUN / "artifacts" / "smoke_test_v4.json"; refuse_existing(output)
    split = json.loads(SPLIT.read_text())
    x,y = load_npz(DATA / "train.npz")
    ref_idx=np.asarray(split["roles"]["reference"]["indices"][:8]); query_idx=np.asarray(split["roles"]["source_holdout"]["indices"][:5])
    device=torch.device("cpu"); torch.set_num_threads(2)
    evidence={"device":str(device),"models":{}}
    for name in ["df","varcnn_direction"]:
        seed_everything(6238); model=model_for(name).to(device).eval()
        ref_x=torch.stack([TraceDataset(x,y,ref_idx)[i][0] for i in range(len(ref_idx))]).to(device)
        q_x=torch.stack([TraceDataset(x,y,query_idx)[i][0] for i in range(len(query_idx))]).to(device)
        with torch.inference_mode():
            ref_logits,ref_global,ref_map=frozen_outputs(model,name,ref_x); q_logits,q_global,q_map=frozen_outputs(model,name,q_x)
            expected_logits, expected_features = model(q_x)
            torch.testing.assert_close(q_logits, expected_logits, rtol=0, atol=0)
            torch.testing.assert_close(q_global, expected_features, rtol=0, atol=0)
            ref_global=torch.nn.functional.normalize(ref_global,dim=1); q_global=torch.nn.functional.normalize(q_global,dim=1)
            ref_local, ref_same, ref_ok, _ = masked_descriptors(ref_map,ref_x,name)
            q_local, q_same, q_ok, _ = masked_descriptors(q_map,q_x,name)
            rb,rc,rd,re,pairs,used=classify_masked(q_global,q_local,q_same,q_ok,ref_global,ref_local,ref_same,ref_ok,torch.as_tensor(y[ref_idx],device=device))
        evidence["models"][name]={"logits":list(q_logits.shape),"global":list(q_global.shape),"local_map":list(q_map.shape),"regions":list(q_local.shape),"predictions": {"A":q_logits.argmax(1).cpu().tolist(),"B":rb.cpu().tolist(),"C":rc.cpu().tolist(),"D":rd.cpu().tolist(),"E":re.cpu().tolist()},"d_pair_shape":list(pairs.shape),"local_used":used.cpu().tolist()}
    atomic_json(output,evidence); print(json.dumps(evidence,indent=2))


def audit_split_content() -> None:
    """Check exact raw and admitted-input duplicate groups do not cross source roles."""
    output=RUN/"artifacts"/"split_content_audit_v4.json"; refuse_existing(output)
    split=json.loads(SPLIT.read_text()); x,_=load_npz(DATA/"train.npz"); vx,_=load_npz(DATA/"valid.npz")
    index_role={int(index):role for role in ["supervised_train","reference","source_holdout"] for index in split["roles"][role]["indices"]}
    summaries={}
    for representation in ["raw_full_row","admitted_sign_first5000"]:
        groups=defaultdict(list)
        for index,row in enumerate(x):
            if index not in index_role:
                continue
            value=np.ascontiguousarray(row) if representation=="raw_full_row" else np.sign(row[:5000]).astype(np.int8)
            groups[hashlib.sha256(value.view(np.uint8)).hexdigest()].append(index)
        crossing=[]
        for digest,indices in groups.items():
            roles=sorted({index_role[i] for i in indices})
            if len(roles)>1: crossing.append({"sha256":digest,"indices":indices,"roles":roles})
        considered=len(index_role)
        summaries[representation]={"considered_canonical_rows":considered,"unique_hashes":len(groups),"duplicate_rows_beyond_first":considered-len(groups),"cross_role_group_count":len(crossing),"cross_role_groups":crossing}
    valid_hashes={hashlib.sha256(np.sign(row[:5000]).astype(np.int8).tobytes()).hexdigest() for row in vx}
    role_valid_intersections={role:sum(hashlib.sha256(np.sign(x[i,:5000]).astype(np.int8).tobytes()).hexdigest() in valid_hashes for i in indices) for role,indices in ((r,split["roles"][r]["indices"]) for r in ["supervised_train","reference","source_holdout"])}
    passed=all(v["cross_role_group_count"]==0 for v in summaries.values()) and not any(role_valid_intersections.values())
    atomic_json(output,{"definition":"SHA-256 of exact C-contiguous bytes; admitted representation is int8 sign of first 5000 values","representations":summaries,"source_validation_intersection_counts":role_valid_intersections,"pass":passed})
    print(json.dumps({"roles":{k:{a:b for a,b in v.items() if a!="cross_role_groups"} for k,v in summaries.items()},"source_validation_intersections":role_valid_intersections,"pass":passed},indent=2))


def train(name: str) -> None:
    config=json.loads((RUN/"config.json").read_text())
    if config.get("status") != "frozen" or config["evaluation"].get("revision") != 4:
        raise ValueError("Requires frozen evaluation revision 4")
    audit=json.loads((PROTOCOL_RUN/"artifacts"/"source_length_audit_v4.json").read_text())
    if audit["split_sha256"] != sha256_file(SPLIT) or not audit["reference_coverage_pass"]:
        raise ValueError("Source length audit/split mismatch or incomplete reference coverage")
    checkpoint=RUN/"checkpoints"/f"{name}_best.pt"; history_path=RUN/"artifacts"/f"{name}_history.json"
    refuse_existing(checkpoint); refuse_existing(history_path)
    split=json.loads(SPLIT.read_text())
    train_idx=np.asarray(split["roles"]["supervised_train"]["indices"],dtype=np.int64)
    x,y=load_npz(DATA/"train.npz"); vx,vy=load_npz(DATA/"valid.npz")
    train_loader=DataLoader(TraceDataset(x,y,train_idx),batch_size=64,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(6238))
    val_loader=DataLoader(TraceDataset(vx,vy),batch_size=128,shuffle=False,num_workers=0)
    if not torch.cuda.is_available(): raise RuntimeError("Full training requires CUDA")
    device=torch.device("cuda"); seed_everything(6238); model=model_for(name).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); criterion=torch.nn.CrossEntropyLoss()
    history=[]; best=-1.0; started=time.perf_counter()
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
        metrics=classification_metrics(np.asarray(truth),np.asarray(pred)); record={"epoch":epoch,"train_loss":loss_sum/total,"train_accuracy":correct/total,**{f"val_{k}":v for k,v in metrics.items()},"elapsed_seconds":time.perf_counter()-started}; history.append(record)
        print(json.dumps(record),flush=True)
        if metrics["macro_f1"]>best:
            best=metrics["macro_f1"]
            torch.save({"model":model.state_dict(),"model_name":name,"seed":6238,"epoch":epoch,"selection_metric":"source_validation_macro_f1","selection_value":best},checkpoint)
    atomic_json(history_path,history)


def extract_reference(model,name,device,x,y,indices):
    loader=DataLoader(TraceDataset(x,y,indices),batch_size=128,shuffle=False,num_workers=0)
    gs=[];ls=[];es=[];oks=[];ys=[]
    with torch.inference_mode():
        for xb,yb,_ in loader:
            xb=xb.to(device); _,g,l=frozen_outputs(model,name,xb)
            regions, same, ok, _ = masked_descriptors(l,xb,name)
            gs.append(torch.nn.functional.normalize(g,dim=1)); ls.append(regions); es.append(same); oks.append(ok); ys.append(yb.to(device))
    return torch.cat(gs),torch.cat(ls),torch.cat(es),torch.cat(oks),torch.cat(ys)


def evaluate(name: str) -> None:
    config=json.loads((RUN/"config.json").read_text())
    if config.get("status") != "frozen" or config["evaluation"].get("revision") != 4:
        raise ValueError("Requires frozen evaluation revision 4")
    output=RUN/"artifacts"/f"{name}_evaluation.json"; refuse_existing(output)
    checkpoint=torch.load(RUN/"checkpoints"/f"{name}_best.pt",map_location="cpu",weights_only=True)
    if checkpoint["model_name"]!=name or checkpoint["seed"]!=6238: raise ValueError("Checkpoint metadata mismatch")
    if not torch.cuda.is_available(): raise RuntimeError("Full evaluation requires CUDA")
    device=torch.device("cuda"); model=model_for(name); model.load_state_dict(checkpoint["model"]); model.to(device).eval()
    split=json.loads(SPLIT.read_text()); ref_idx=np.asarray(split["roles"]["reference"]["indices"]); hold_idx=np.asarray(split["roles"]["source_holdout"]["indices"])
    sx,sy=load_npz(DATA/"train.npz"); started=time.perf_counter(); rg,rl,re,rok,ry=extract_reference(model,name,device,sx,sy,ref_idx); torch.cuda.synchronize(); reference_seconds=time.perf_counter()-started
    reference={"count":len(ref_idx),"raw_input_bytes":int(len(ref_idx)*5000*4),"global_feature_bytes":rg.numel()*rg.element_size(),"local_feature_bytes":rl.numel()*rl.element_size(),"global_shape":list(rg.shape),"local_shape":list(rl.shape),"extraction_seconds":reference_seconds,"same_layer_feature_bytes":re.numel()*re.element_size(),"eligible_reference_count":int(rok.sum()),"eligible_reference_classes":int(torch.unique(ry[rok]).numel())}
    domains=[("source_holdout",DATA/"train.npz",hold_idx)]+[(f"day{d}",DATA/f"day{d}.npz",None) for d in [14,30,90,150,270]]
    results={}; predictions={}
    for domain,path,indices in domains:
        x,y=(sx,sy) if domain=="source_holdout" else load_npz(path)
        loader=DataLoader(TraceDataset(x,y,indices),batch_size=128,shuffle=False,num_workers=0)
        truth=[];row_ids=[];preds={k:[] for k in "ABCDE"}; chosen=[]; used_rows=[]; valid_counts=[]; valid_masks=[]; extraction_time=classification_time=0.0
        with torch.inference_mode():
            for xb,yb,ib in loader:
                xb=xb.to(device); torch.cuda.synchronize(); t=time.perf_counter(); logits,g,local=frozen_outputs(model,name,xb); g=torch.nn.functional.normalize(g,dim=1); local,same,ok,counts=masked_descriptors(local,xb,name); torch.cuda.synchronize(); extraction_time+=time.perf_counter()-t
                t=time.perf_counter(); pb,pc,pd,pe,pairs,used=classify_masked(g,local,same,ok,rg,rl,re,rok,ry); pa=logits.argmax(1); torch.cuda.synchronize(); classification_time+=time.perf_counter()-t
                truth.extend(yb.tolist()); row_ids.extend(ib.tolist());
                for key,val in zip("ABCDE",[pa,pb,pc,pd,pe]): preds[key].extend(val.cpu().tolist())
                chosen.extend(pairs.cpu().tolist()); used_rows.extend(used.cpu().tolist()); valid_counts.extend(counts.cpu().tolist())
                # Persist the exact per-position eligibility used by D/E in a
                # compact, lossless representation for protocol diagnostics.
                mask = valid_local_positions(xb, name).cpu().numpy()
                valid_masks.extend(row.tobytes().hex() for row in np.packbits(mask, axis=1, bitorder="little"))
        truth_np=np.asarray(truth); results[domain]={key:classification_metrics(truth_np,np.asarray(value)) for key,value in preds.items()}
        results[domain]["per_website_accuracy"]={key:{str(label):float((np.asarray(value)[truth_np==label]==label).mean()) for label in range(102)} for key,value in preds.items()}
        results[domain]["cost"]={"samples":len(truth),"shared_single_encoder_pass_seconds":extraction_time,"all_five_classifiers_seconds":classification_time,"equivalent_scores_per_query":{"A_logits":102,"B_prototype_cosines":102,"C_reference_cosines":len(ref_idx),"D_region_cosines":int(rok.sum())*16,"E_reference_cosines":int(rok.sum())}}
        # Labels enter only here, after all predictions and match choices are fixed.
        used_np=np.asarray(used_rows,dtype=bool)
        chosen_np=np.asarray(chosen)[used_np]; diagnostic_truth=truth_np[used_np]
        selected_ref_labels=sy[ref_idx[chosen_np[:,0]]]; correct=selected_ref_labels==diagnostic_truth
        results[domain]["regional_coverage"]={"used":int(used_np.sum()),"fallback_to_C":int((~used_np).sum()),
            "eligible_subset_metrics":{key:classification_metrics(truth_np[used_np],np.asarray(value)[used_np]) for key,value in preds.items()} if used_np.any() else None}
        ref_sites=defaultdict(set); ref_hits=defaultdict(int); region_sites=defaultdict(set); region_hits=defaultdict(int)
        for true_label,row in zip(diagnostic_truth,chosen_np):
            ri=int(row[0]); ref_sites[ri].add(int(true_label)); ref_hits[ri]+=1
            for qr,rr in enumerate(row[1:]): region_sites[(ri,int(rr))].add(int(true_label)); region_hits[(ri,int(rr))]+=1
        results[domain]["local_diagnostic"]={"correct_same_site_matches":int(correct.sum()),"incorrect_cross_site_matches":int((~correct).sum()),"selected_reference_unique_true_site_breadth_histogram":dict(sorted((str(k),sum(len(v)==k for v in ref_sites.values())) for k in set(map(len,ref_sites.values())))),"max_true_sites_hitting_one_reference":max(map(len,ref_sites.values()),default=0),"max_true_sites_hitting_one_reference_region":max(map(len,region_sites.values()),default=0),"top_broad_reference_regions":[{"reference_position":ri,"reference_row_index":int(ref_idx[ri]),"region":rr,"distinct_true_sites":len(sites),"hits":region_hits[(ri,rr)]} for (ri,rr),sites in sorted(region_sites.items(),key=lambda z:(-len(z[1]),-region_hits[z[0]]))[:20]]}
        predictions[domain]={"source_file":path.name,"row_indices":row_ids,"truth_scoring_only":truth,"predictions":preds,"d_selected_reference_and_four_regions":chosen,"regional_inference_used":used_rows,"valid_local_position_counts":valid_counts,"valid_local_position_mask":{"encoding":"numpy.packbits rows as hex","bitorder":"little","positions":LOCAL_SPECS[name]["positions"],"rows":valid_masks}}
        atomic_json(RUN/"artifacts"/f"{name}_{domain}_predictions.json",predictions[domain])
        if domain!="source_holdout": del x,y
    source_gap=results["source_holdout"]["D"]["accuracy"]-results["source_holdout"]["C"]["accuracy"]
    for domain in results:
        results[domain]["D_minus_C_accuracy"]=results[domain]["D"]["accuracy"]-results[domain]["C"]["accuracy"]
        results[domain]["future_D_minus_C_minus_source"]=(None if domain=="source_holdout" else results[domain]["D_minus_C_accuracy"]-source_gap)
    for metric in ["accuracy", "macro_f1"]:
        source_delta=results["source_holdout"]["D"][metric]-results["source_holdout"]["E"][metric]
        for domain in results:
            delta=results[domain]["D"][metric]-results[domain]["E"][metric]
            results[domain]["D_minus_E_"+metric]=delta
            results[domain]["future_D_minus_E_minus_source_"+metric]=None if domain=="source_holdout" else delta-source_delta
    atomic_json(output,{"evaluation_revision":4,"model":name,"checkpoint":checkpoint|{"model":"omitted; see checkpoint file"},"reference":reference,"domains":results}); print(output)


def audit_lengths():
    """Source-only structure audit; no model, future arrays, or performance scores."""
    output=RUN/"artifacts"/"source_length_audit_v4.json"; refuse_existing(output)
    layer_output=RUN/"artifacts"/"local_layer_spec_v4.json"; refuse_existing(layer_output)
    torch.set_num_threads(2)
    split=json.loads(SPLIT.read_text()); x,y=load_npz(DATA/"train.npz")
    summaries={}; coverage=True
    sources=[(role,x,y,np.asarray(info["indices"])) for role,info in split["roles"].items()]
    vx,vy=load_npz(DATA/"valid.npz"); sources.append(("source_validation",vx,vy,np.arange(len(vy))))
    for role,values,labels,indices in sources:
        a=values[indices,:5000]; nz=a!=0
        lengths=np.where(nz,np.arange(1,5001),0).max(1)
        interior=(~nz)&(np.arange(5000)[None,:]<lengths[:,None])
        summary={"count":len(a),"observed_span_quantiles":np.quantile(lengths,[0,.1,.5,.9,1]).tolist(),
                 "shorter_than_5000":int((lengths<5000).sum()),"all_zero":int((lengths==0).sum()),
                 "rows_with_interior_zero":int(interior.any(1).sum()),"nonfinite":int((~np.isfinite(a)).sum()),"models":{}}
        if summary["nonfinite"]: raise ValueError("Nonfinite source inputs")
        for name in LOCAL_SPECS:
            counts=[]
            for start in range(0,len(a),128):
                xb=torch.from_numpy(np.sign(a[start:start+128]).astype(np.float32))[:,None,:]
                counts.extend(valid_local_positions(xb,name).sum(1).tolist())
            eligible=np.asarray(counts)>=4
            classes=len(np.unique(labels[indices][eligible]))
            summary["models"][name]={"eligible":int(eligible.sum()),"ineligible":int((~eligible).sum()),
                                     "eligible_classes":classes,"minimum_valid_positions":min(counts)}
            if role=="reference": coverage=coverage and classes==102
        summaries[role]=summary
    atomic_json(output,{"split_sha256":sha256_file(SPLIT),"source_only":True,"roles":summaries,"reference_coverage_pass":coverage})
    atomic_json(layer_output,{"revision":4,"models":LOCAL_SPECS,"regions":4,
        "validity":"All inclusive RF input indices inside 0..4999 and nonzero/finite per sample; zero means unknown, not proven padding.",
        "rf_inclusive_bounds":{"df":"[4*i-8,4*i+13]","varcnn_direction":"[4*i-17,4*i+17]"},
        "E":"L2-normalized mean of raw features over exactly the D-valid positions",
        "fallback":"D/E fall back to C for fewer than 4 valid positions; exclude ineligible references for both. If any class loses all references, all D/E queries fall back; report coverage and eligible subset metrics.",
        "interpretation":"Shallow convolutional regional evidence with overlapping RF, not independent resources. BN training statistics can include padding; eval uses frozen running statistics."})
    print(json.dumps({"roles":summaries,"reference_coverage_pass":coverage},indent=2))


def main():
    global RUN
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["prepare","audit-split","audit-length","smoke","train","evaluate"]); parser.add_argument("--model",choices=["df","varcnn_direction"]); parser.add_argument("--run-dir",type=Path,default=PROTOCOL_RUN); args=parser.parse_args()
    run_dir = args.run_dir.resolve()
    runs_root = (ROOT / "runs").resolve()
    if run_dir.parent != runs_root or not run_dir.is_dir():
        parser.error("--run-dir must name an existing direct child of this workspace's runs directory")
    RUN = run_dir
    if args.stage in {"train","evaluate"} and not args.model: parser.error("--model required")
    {"prepare":prepare,"audit-split":audit_split_content,"audit-length":audit_lengths,"smoke":smoke}.get(args.stage,lambda: (train if args.stage=="train" else evaluate)(args.model))()

if __name__=="__main__": main()
