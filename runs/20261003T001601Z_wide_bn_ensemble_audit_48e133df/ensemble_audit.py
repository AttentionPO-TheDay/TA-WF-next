import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
BASE = ROOT / "runs/20261002T053706Z_progressive_capacity_bn_mixup_1ad233da"
HEAD = ROOT / "runs/20261002T045631Z_wf_generator_head_transfer_4ed54f62"
SEEDS = [21729, 23407, 22026]
CLASSES = np.arange(102)


def load(kind, seed, base=BASE):
    with np.load(base / "artifacts" / f"{kind}_{seed}" / "logits_best.npz") as z:
        return {role: z[role].astype(np.float64) for role in ("source", "valid")}


def zscore(x):
    center = x.mean(axis=1, keepdims=True)
    scale = x.std(axis=1, keepdims=True)
    return (x - center) / np.where(scale < 1e-12, 1.0, scale)


def metric(y, logits):
    pred = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "correct": int((pred == y).sum()),
        "count": int(len(y)),
    }, pred


def main():
    raw = torch.load(BASE / "artifacts" / "prepared.pt", weights_only=False, map_location="cpu")
    labels = {role: raw[role]["labels"].numpy() for role in ("source", "valid")}
    wide = {seed: load("wide_bn", seed) for seed in SEEDS}
    mlp = {seed: load("progressive_mlp", seed, HEAD) for seed in SEEDS}
    rf = {seed: load("rf_matched", seed, HEAD) for seed in SEEDS}

    methods = {}
    for role in ("source", "valid"):
        wraw = np.stack([wide[s][role] for s in SEEDS])
        wz = np.stack([zscore(wide[s][role]) for s in SEEDS])
        mraw = np.stack([mlp[s][role] for s in SEEDS])
        mz = np.stack([zscore(mlp[s][role]) for s in SEEDS])
        rraw = np.stack([rf[s][role] for s in SEEDS])
        rz = np.stack([zscore(rf[s][role]) for s in SEEDS])
        methods.setdefault("wide_bn_seed_mean", {})[role] = wraw.mean(axis=0)
        methods.setdefault("wide_bn_z_mean", {})[role] = wz.mean(axis=0)
        vote = np.zeros_like(wraw[0])
        for row in np.argmax(wraw, axis=2):
            vote[np.arange(len(row)), row] += 1
        methods.setdefault("wide_bn_vote", {})[role] = vote + 1e-6 * wz.sum(axis=0)
        for alpha in (0.25, 0.5, 0.75):
            methods.setdefault(f"mlp_fusion_{alpha:.2f}", {})[role] = alpha * wz.mean(axis=0) + (1 - alpha) * mz.mean(axis=0)
            methods.setdefault(f"rf_fusion_{alpha:.2f}", {})[role] = alpha * wz.mean(axis=0) + (1 - alpha) * rz.mean(axis=0)

    rows = []
    predictions = {}
    for name, by_role in methods.items():
        predictions[name] = {}
        for role in ("source", "valid"):
            stats, pred = metric(labels[role], by_role[role])
            predictions[name][role] = pred
            rows.append({"method": name, "role": role, **stats})

    # Disagreement and oracle union are diagnostics only, never a deployed selector.
    disagreement = {}
    for left_name, left_pool, right_name, right_pool in [("wide_bn", wide, "mlp", mlp), ("wide_bn", wide, "rf", rf)]:
        for role in ("source", "valid"):
            left = np.stack([left_pool[s][role].argmax(1) for s in SEEDS])
            right = np.stack([right_pool[s][role].argmax(1) for s in SEEDS])
            left_correct = left == labels[role][None, :]
            right_correct = right == labels[role][None, :]
            disagreement[f"{left_name}_vs_{right_name}_{role}"] = {
                "any_disagreement": int(np.any(left != right, axis=0).sum()),
                "left_mean_accuracy": float(left_correct.mean()),
                "right_mean_accuracy": float(right_correct.mean()),
                "oracle_union_accuracy": float(np.any(left_correct | right_correct, axis=0).mean()),
            }
    for role in ("source", "valid"):
        p = np.stack([wide[s][role].argmax(1) for s in SEEDS])
        c = p == labels[role][None, :]
        disagreement[f"wide_bn_seed_disagreement_{role}"] = {
            "any_disagreement": int(np.any(p != p[0], axis=0).sum()),
            "oracle_union_accuracy": float(np.any(c, axis=0).mean()),
        }
    manifest = {}
    for path in [BASE / "artifacts" / "prepared.pt"] + [BASE / "artifacts" / f"wide_bn_{s}" / "logits_best.npz" for s in SEEDS] + [HEAD / "artifacts" / f"progressive_mlp_{s}" / "logits_best.npz" for s in SEEDS] + [HEAD / "artifacts" / f"rf_matched_{s}" / "logits_best.npz" for s in SEEDS]:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[str(path.relative_to(ROOT))] = h
    out = {"rows": rows, "disagreement": disagreement, "input_hashes": manifest, "future_access": False, "training": False}
    (RUN / "artifacts").mkdir(exist_ok=True)
    (RUN / "artifacts" / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(RUN / "artifacts" / "predictions.npz", **{
        f"{name}__{role}": pred for name, by_role in methods.items() for role, pred in predictions[name].items()
    })
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
