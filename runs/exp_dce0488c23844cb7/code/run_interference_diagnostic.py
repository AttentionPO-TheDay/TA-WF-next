"""Frozen shared-update interference diagnostic for exp_dce0488c23844cb7.

This is a minimal, self-contained import of the archived DF + source-statistics
Tent behavior. It never imports the archived project and never writes a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/exp_dce0488c23844cb7"
DATA = Path("/mnt/data2/ren/datasets/TemporalDrift/day90.npz")
CHECKPOINT = Path("/home/rbf/TA-WF/outputs/df_source_tent_seed3407_pilot-20260905-194436/best.pt")
EXPECTED_CHECKPOINT_SHA256 = "27491ed4f488f517cea31a5b50f661e9d90f843ae5a43d0509688de19e6913f4"
SPLIT_SEED = 20260919
CONTROL_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924)
CLASSES = 102
BUDGET = 64
LR = 0.005
STEPS = 5
THRESHOLD = 0.5
BATCH = 256


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, activation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inc, outc, 8, 1, padding=4, bias=False),
            nn.BatchNorm1d(outc), activation(inplace=True),
            nn.Conv1d(outc, outc, 8, 1, padding=4, bias=False),
            nn.BatchNorm1d(outc), activation(inplace=True),
            nn.MaxPool1d(8, 4), nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.block(x)


class DF(nn.Module):
    def __init__(self, classes=CLASSES):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            ConvBlock(1, 32, nn.ELU), ConvBlock(32, 64, nn.ReLU),
            ConvBlock(64, 128, nn.ReLU), ConvBlock(128, 256, nn.ReLU),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 18, 512, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.7),
            nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
        )
        self.mlp = nn.Linear(512, classes)

    def forward(self, x):
        z = self.classifier(self.feature_extraction(x))
        return self.mlp(z), z


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def entropy(logits):
    lp = logits.log_softmax(1)
    return -(lp.exp() * lp).sum(1).mean()


def logits_of(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def make_model(device):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 30 or payload.get("num_classes") != CLASSES:
        raise ValueError("checkpoint metadata mismatch")
    model = DF().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    selected = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
            selected += [f"{module_name}.weight", f"{module_name}.bias"]
    selected = tuple(selected)
    for name, p in model.named_parameters():
        p.requires_grad_(name in selected)
    return model, selected


def forward(model, parameters, x, want_embedding=False):
    out = functional_call(model, parameters, (x,))
    return out if want_embedding else logits_of(out)


def adapt(model, selected, source, support):
    current = dict(source)
    step_effective, step_grad = [], []
    with torch.no_grad():
        before_logits = forward(model, source, support)
        before_entropy = float(entropy(before_logits).item())
    for _ in range(STEPS):
        logits = forward(model, current, support)
        mask = logits.softmax(1).amax(1) >= THRESHOLD
        loss = entropy(logits[mask]) if mask.any() else entropy(logits) * 0.0
        params = [current[n] for n in selected]
        grads = torch.autograd.grad(loss, params)
        grad_norm = math.sqrt(sum(float(g.detach().square().sum().item()) for g in grads))
        updated = dict(current)
        for name, parameter, gradient in zip(selected, params, grads, strict=True):
            updated[name] = parameter - LR * gradient
        current = {n: p.detach().requires_grad_(p.requires_grad) for n, p in updated.items()}
        step_effective.append(int(mask.sum().item()))
        step_grad.append(grad_norm)
    with torch.no_grad():
        after_entropy = float(entropy(forward(model, current, support)).item())
    accepted = after_entropy <= before_entropy
    final = current if accepted else source
    delta_sq = source_sq = 0.0
    for name in selected:
        delta_sq += float((final[name] - source[name]).square().sum().item())
        source_sq += float(source[name].square().sum().item())
    return final, {
        "support_count": len(support), "effective_counts": step_effective,
        "effective_fraction_mean": float(np.mean(step_effective) / len(support)),
        "gradient_norms": step_grad, "gradient_norm_mean": float(np.mean(step_grad)),
        "parameter_delta_l2": math.sqrt(delta_sq),
        "relative_parameter_delta_l2": math.sqrt(delta_sq) / max(math.sqrt(source_sq), 1e-30),
        "support_entropy_before": before_entropy, "support_entropy_after": after_entropy,
        "accepted": accepted,
    }


def predict(model, parameters, x, device, want_embedding=False):
    logits_parts, emb_parts = [], []
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            xb = torch.from_numpy(x[start:start+BATCH]).to(device)
            out = forward(model, parameters, xb, want_embedding=True)
            logits_parts.append(out[0].cpu())
            if want_embedding:
                emb_parts.append(out[1].cpu())
    logits = torch.cat(logits_parts).numpy()
    emb = torch.cat(emb_parts).numpy() if want_embedding else None
    return logits, emb


def softmax(a):
    z = a - a.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics(y, baseline_logits, updated_logits):
    bp, up = baseline_logits.argmax(1), updated_logits.argmax(1)
    bprob, uprob = softmax(baseline_logits), softmax(updated_logits)
    rows = np.arange(len(y))
    bc, uc = bp == y, up == y
    return {
        "sample_count": len(y),
        "baseline_accuracy": float(bc.mean()), "updated_accuracy": float(uc.mean()),
        "delta_accuracy": float(uc.mean() - bc.mean()),
        "baseline_true_probability": float(bprob[rows, y].mean()),
        "updated_true_probability": float(uprob[rows, y].mean()),
        "delta_true_probability": float((uprob[rows, y] - bprob[rows, y]).mean()),
        "wrong_to_correct": int((~bc & uc).sum()),
        "correct_to_wrong": int((bc & ~uc).sum()),
        "wrong_to_different_wrong": int((~bc & ~uc & (bp != up)).sum()),
        "unchanged_correct": int((bc & uc & (bp == up)).sum()),
    }


def macro_f1(y, pred):
    cm = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float); actual = cm.sum(1); guessed = cm.sum(0)
    precision = np.divide(tp, guessed, out=np.zeros_like(tp), where=guessed != 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros_like(tp), where=precision+recall != 0)
    return float(f1.mean())


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]: j += 1
        out[order[i:j]] = (i + j - 1) / 2
        i = j
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return float("nan")
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def group_run(model, selected, source, x, y, adapt_a, adapt_b, eval_a, eval_b,
              kind, repeat, pair_id, group_labels, matrix, audits, prediction_store):
    supports = {
        "A": adapt_a,
        "B": adapt_b,
        "mixed": np.random.default_rng(880000 + repeat*1000 + pair_id).permutation(
            np.concatenate((adapt_a[:BUDGET//2], adapt_b[:BUDGET//2]))),
    }
    params = {}
    source_audits = {}
    for update_source, indices in supports.items():
        support = torch.from_numpy(x[indices]).to(next(model.parameters()).device)
        params[update_source], audit = adapt(model, selected, source, support)
        audit.update({"kind": kind, "repeat": repeat, "pair_id": pair_id,
                      "update_source": update_source})
        audits.append(audit)
        source_audits[update_source] = audit
    for recipient, indices in (("A", eval_a), ("B", eval_b)):
        base_logits, _ = predict(model, source, x[indices], next(model.parameters()).device)
        prediction_store[(kind, repeat, pair_id, recipient, "baseline")] = (y[indices], base_logits)
        baseline_row = {"date":"day90", "checkpoint_sha256":EXPECTED_CHECKPOINT_SHA256,
                   "tta_config":"source_stats_bn_affine_lr0.005_steps5_conf0.5_fallback",
                   "kind":kind, "repeat":repeat, "pair_id":pair_id,
                   "group":recipient, "group_definition":group_labels[recipient],
                   "update_source":"no_update", "eval_group":recipient,
                   "adaptation_budget":0, "effective_fraction_mean":0.0,
                   "gradient_norm_mean":0.0, "parameter_delta_l2":0.0,
                   "relative_parameter_delta_l2":0.0, "update_accepted":False}
        baseline_row.update(metrics(y[indices], base_logits, base_logits)); matrix.append(baseline_row)
        for update_source in ("A", "B", "mixed"):
            ul, _ = predict(model, params[update_source], x[indices], next(model.parameters()).device)
            prediction_store[(kind, repeat, pair_id, recipient, update_source)] = (y[indices], ul)
            row = {"date":"day90", "checkpoint_sha256":EXPECTED_CHECKPOINT_SHA256,
                   "tta_config":"source_stats_bn_affine_lr0.005_steps5_conf0.5_fallback",
                   "kind":kind, "repeat":repeat, "pair_id":pair_id,
                   "group":recipient, "group_definition":group_labels[recipient],
                   "update_source":update_source, "eval_group":recipient,
                   "adaptation_budget":len(supports[update_source]),
                   "effective_fraction_mean":source_audits[update_source]["effective_fraction_mean"],
                   "gradient_norm_mean":source_audits[update_source]["gradient_norm_mean"],
                   "parameter_delta_l2":source_audits[update_source]["parameter_delta_l2"],
                   "relative_parameter_delta_l2":source_audits[update_source]["relative_parameter_delta_l2"],
                   "update_accepted":source_audits[update_source]["accepted"]}
            row.update(metrics(y[indices], base_logits, ul)); matrix.append(row)


def summarize_kind(kind, repeat, matrix, prediction_store):
    rows = [r for r in matrix if r["kind"] == kind and r["repeat"] == repeat]
    directed = []
    for pair_id in sorted({r["pair_id"] for r in rows}):
        pr = [r for r in rows if r["pair_id"] == pair_id]
        for recipient, own, other in (("A","A","B"),("B","B","A")):
            self_r = next(r for r in pr if r["group"]==recipient and r["update_source"]==own)
            cross_r = next(r for r in pr if r["group"]==recipient and r["update_source"]==other)
            mixed_r = next(r for r in pr if r["group"]==recipient and r["update_source"]=="mixed")
            directed.append({"kind":kind,"repeat":repeat,"pair_id":pair_id,"recipient":recipient,
                "group_definition":self_r["group_definition"],
                "self_delta_accuracy":self_r["delta_accuracy"],"cross_delta_accuracy":cross_r["delta_accuracy"],
                "mixed_delta_accuracy":mixed_r["delta_accuracy"],
                "self_delta_true_probability":self_r["delta_true_probability"],
                "cross_delta_true_probability":cross_r["delta_true_probability"],
                "mixed_delta_true_probability":mixed_r["delta_true_probability"],
                "interference_contrast_accuracy":self_r["delta_accuracy"]-cross_r["delta_accuracy"],
                "self_positive_cross_negative":int(self_r["delta_accuracy"]>0 and cross_r["delta_accuracy"]<0),
                "mixed_cancels_positive_self":int(self_r["delta_accuracy"]>0 and mixed_r["delta_accuracy"]<=.5*self_r["delta_accuracy"])})
    agg=[]
    for condition in ("baseline","A","B","mixed","self","cross"):
        ys=[]; logits=[]
        for pair_id in sorted({r["pair_id"] for r in rows}):
            for recipient in ("A","B"):
                src = condition
                if condition == "self": src = recipient
                elif condition == "cross": src = "B" if recipient=="A" else "A"
                key=(kind,repeat,pair_id,recipient,src)
                if key not in prediction_store: continue
                yy,ll=prediction_store[key]; ys.append(yy); logits.append(ll)
        if ys:
            yy=np.concatenate(ys); ll=np.concatenate(logits); pred=ll.argmax(1); prob=softmax(ll)
            agg.append({"kind":kind,"repeat":repeat,"condition":condition,"sample_count":len(yy),
                        "accuracy":float((pred==yy).mean()),"macro_f1":macro_f1(yy,pred),
                        "mean_true_probability":float(prob[np.arange(len(yy)),yy].mean())})
    return directed, agg


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="cuda"); ap.add_argument("--oracle-only",action="store_true")
    args=ap.parse_args(); started=time.time()
    if sha256_file(CHECKPOINT)!=EXPECTED_CHECKPOINT_SHA256: raise ValueError("checkpoint hash mismatch")
    np.random.seed(SPLIT_SEED); random.seed(SPLIT_SEED); torch.manual_seed(SPLIT_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SPLIT_SEED)
    device=torch.device(args.device if args.device!="cuda" or torch.cuda.is_available() else "cpu")
    with np.load(DATA,allow_pickle=False) as z:
        raw=z["X"]; y=z["y"].astype(np.int64)
        x=np.asarray(raw[:,:5000],dtype=np.float32)[:,None,:]
    if set(np.unique(y)) != set(range(CLASSES)): raise ValueError("label set mismatch")
    model,selected=make_model(device); source=dict(model.named_parameters())
    # Static fingerprint guards both the archived model and raw-timestamp pipeline.
    static_logits,_=predict(model,source,x[:64],device)
    static_fingerprint=hashlib.sha256(static_logits.astype("<f4").tobytes()).hexdigest()
    rng=np.random.default_rng(SPLIT_SEED); adapt_by_class={}; eval_by_class={}
    for c in range(CLASSES):
        idx=np.flatnonzero(y==c); idx=rng.permutation(idx)
        adapt_by_class[c]=idx[:BUDGET]; eval_by_class[c]=idx[BUDGET:]
    matrix=[]; audits=[]; store={}
    for pair_id,a in enumerate(range(0,CLASSES,2)):
        b=a+1
        group_run(model,selected,source,x,y,adapt_by_class[a],adapt_by_class[b],eval_by_class[a],eval_by_class[b],
                  "oracle",0,pair_id,{"A":f"website:{a}","B":f"website:{b}"},matrix,audits,store)
        print(f"oracle {pair_id+1}/51",flush=True)
    if not args.oracle_only:
        for repeat,seed in enumerate(CONTROL_SEEDS,1):
            crng=np.random.default_rng(seed)
            for pair_id,a in enumerate(range(0,CLASSES,2)):
                b=a+1; ad=crng.permutation(np.concatenate((adapt_by_class[a],adapt_by_class[b])))
                ev=crng.permutation(np.concatenate((eval_by_class[a],eval_by_class[b])))
                na=len(eval_by_class[a])
                group_run(model,selected,source,x,y,ad[:BUDGET],ad[BUDGET:],ev[:na],ev[na:],
                          "random",repeat,pair_id,{"A":f"label_blind_random_seed:{seed}:half:A","B":f"label_blind_random_seed:{seed}:half:B"},matrix,audits,store)
            print(f"random repeat {repeat}/5",flush=True)
    directed=[]; aggregates=[]
    for kind,repeat in sorted({(r["kind"],r["repeat"]) for r in matrix}):
        d,a=summarize_kind(kind,repeat,matrix,store); directed+=d; aggregates+=a
    write_csv(RUN/"UPDATE_SOURCE_RECIPIENT_MATRIX.csv",matrix)
    write_csv(RUN/"pair_level_interference.csv",[r for r in directed if r["kind"]=="oracle"])
    write_csv(RUN/"random_group_controls.csv",[r for r in directed if r["kind"]=="random"])
    audit_rows=[]
    for r in audits:
        q=dict(r); q["effective_counts"]=json.dumps(q["effective_counts"]); q["gradient_norms"]=json.dumps(q["gradient_norms"]); audit_rows.append(q)
    write_csv(RUN/"update_magnitude_audit.csv",audit_rows); write_csv(RUN/"aggregate_metrics.csv",aggregates)
    oracle=[r for r in directed if r["kind"]=="oracle"]
    random_summaries=[]
    for repeat in range(1,6):
        rr=[r for r in directed if r["kind"]=="random" and r["repeat"]==repeat]
        if rr: random_summaries.append({"repeat":repeat,"contrast_mean":float(np.mean([x["interference_contrast_accuracy"] for x in rr])),
                                        "direction_rate":float(np.mean([x["self_positive_cross_negative"] for x in rr]))})
    self_acc=np.array([r["self_delta_accuracy"] for r in oracle]); cross_acc=np.array([r["cross_delta_accuracy"] for r in oracle]); mixed_acc=np.array([r["mixed_delta_accuracy"] for r in oracle])
    self_tp=np.array([r["self_delta_true_probability"] for r in oracle]); cross_tp=np.array([r["cross_delta_true_probability"] for r in oracle])
    oa=[r for r in audits if r["kind"]=="oracle" and r["update_source"] in ("A","B")]
    contrast=np.array([r["interference_contrast_accuracy"] for r in oracle])
    magnitude_ok=False; magnitude_detail={}
    if oa and len(oa)==len(oracle):
        eff=np.array([r["effective_fraction_mean"] for r in oa]); mag=np.array([r["relative_parameter_delta_l2"] for r in oa])
        bysource={s:[r for r in audits if r["kind"]=="oracle" and r["update_source"]==s] for s in ("A","B","mixed")}
        effmed=[np.median([r["effective_fraction_mean"] for r in bysource[s]]) for s in bysource]
        magmed=[np.median([r["relative_parameter_delta_l2"] for r in bysource[s]]) for s in bysource]
        ratio=max(magmed)/max(min(magmed),1e-30)
        magnitude_detail={"effective_medians":effmed,"relative_delta_medians":magmed,"relative_delta_median_ratio":ratio,
                          "rho_contrast_effective":spearman(contrast,eff),"rho_contrast_relative_delta":spearman(contrast,mag)}
        magnitude_ok=(max(effmed)-min(effmed)<=.10 and ratio<=2 and abs(magnitude_detail["rho_contrast_effective"])<.5 and abs(magnitude_detail["rho_contrast_relative_delta"])<.5)
    gates={
        "self_usable":bool(self_acc.mean()>=.005 and self_tp.mean()>=.002 and (self_acc>0).mean()>=.60),
        "cross_damage":bool(cross_acc.mean()<=-.005 and cross_tp.mean()<=-.002 and (cross_acc<0).mean()>=.60),
        "mixed_cancellation":bool(((mixed_acc[self_acc>0] <= .5*self_acc[self_acc>0]).mean() if (self_acc>0).any() else 0)>=.60 and mixed_acc.mean()<=.5*self_acc.mean()),
        "exceeds_random":bool(random_summaries and contrast.mean()>max(r["contrast_mean"] for r in random_summaries) and np.mean([r["self_positive_cross_negative"] for r in oracle])>max(r["direction_rate"] for r in random_summaries)),
        "magnitude_not_simple":bool(magnitude_ok),
    }
    passed=all(gates.values())
    summary={"experiment_id":"exp_dce0488c23844cb7","evidence_role":"TemporalDrift development evidence",
             "checkpoint_sha256":EXPECTED_CHECKPOINT_SHA256,"dataset_sha256":sha256_file(DATA),"device":str(device),
             "static_logits_first64_sha256":static_fingerprint,"oracle_directed_groups":len(oracle),
             "random_repeats_completed":len(random_summaries),"first_gate":gates,
             "first_gate_passed":passed,"decision":"PENDING_STAGE2" if passed else "STOP_NO_USABLE_INTERFERENCE",
             "oracle_summary":{"self_accuracy_mean":float(self_acc.mean()),"self_true_probability_mean":float(self_tp.mean()),"self_positive_rate":float((self_acc>0).mean()),
                 "cross_accuracy_mean":float(cross_acc.mean()),"cross_true_probability_mean":float(cross_tp.mean()),"cross_negative_rate":float((cross_acc<0).mean()),
                 "mixed_accuracy_mean":float(mixed_acc.mean()),"interference_contrast_mean":float(contrast.mean()),
                 "self_positive_cross_negative_rate":float(np.mean([r["self_positive_cross_negative"] for r in oracle]))},
             "random_summaries":random_summaries,"magnitude_audit":magnitude_detail,
             "runtime_seconds":time.time()-started,"training_runs":0,"checkpoint_outputs":0,"external_test":False}
    (RUN/"first_gate_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__": main()
