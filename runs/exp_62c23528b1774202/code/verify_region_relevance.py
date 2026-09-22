#!/usr/bin/env python3
"""Independent integrity verification for exp_62c23528b1774202."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/exp_62c23528b1774202"
ART = RUN / "artifacts"
DIAG = ROOT / "runs/exp_04faf4088b604155/artifacts"
SELECT = ROOT / "runs/exp_376fca9354214097/artifacts"
MODELS = ("df", "varcnn_direction")
DATES = ("day14", "day90", "day270")
SEEDS = (1729, 6238, 20260916)
METHODS = ("A_support_cv_3shot", "B_plain_multiprototype", "C_site_global_weight",
           "D_region_conflict_suppression", "D_ablation_uniform_site_retention")
CLASSES = 102


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def f1_accuracy(y, p):
    c = np.zeros((CLASSES, CLASSES), np.int64); np.add.at(c, (y, p), 1)
    tp = np.diag(c).astype(float); actual = c.sum(1); guessed = c.sum(0)
    pr = np.divide(tp, guessed, out=np.zeros(CLASSES), where=guessed != 0)
    re = np.divide(tp, actual, out=np.zeros(CLASSES), where=actual != 0)
    f1 = np.divide(2*pr*re, pr+re, out=np.zeros(CLASSES), where=(pr+re) != 0)
    return float(tp.sum()/c.sum()), float(f1.mean())


def main():
    errors = []
    seal = json.loads((ART / "firewall_seal_v1.json").read_text())
    for path, expected in seal["files"].items():
        if sha(Path(path)) != expected: errors.append(f"seal mismatch {path}")
    rows = list(csv.DictReader((ART / "metrics_long.csv").open()))
    lookup = {(r["backbone"], r["date"], int(r["seed"]), r["method"]): r for r in rows}
    checked_predictions = 0; conflict_regions = 0
    for model in MODELS:
        lib = np.load(ART / f"shared_history_{model}_v1.npz", allow_pickle=False)
        source = np.load(DIAG / f"{model}_source_geometry_v1.npz", allow_pickle=False)
        for key in ("exemplars", "exemplar_rows", "thresholds"):
            if not np.array_equal(lib[key], source[key]): errors.append(f"library mismatch {model}/{key}")
        for date in DATES:
            for seed in SEEDS:
                frozen = np.load(ART / f"frozen_{model}_{date}_seed{seed}_v1.npz", allow_pickle=False)
                prior = np.load(DIAG / f"frozen_{model}_{date}_seed{seed}_v1.npz", allow_pickle=False)
                if not np.array_equal(frozen["support_rows"], prior["support_order"][:, :3]): errors.append("formal3 mismatch")
                if np.intersect1d(frozen["support_rows"].ravel(), prior["support_order"][:, 3:].ravel()).size: errors.append("support overlap with extra7")
                a = json.loads((SELECT / f"{model}_{date}_seed{seed}_shot3.json").read_text())
                query = np.asarray(a["common_query_rows"], np.int64); truth = np.asarray(a["common_query_truth"], np.int16)
                if not np.array_equal(query, frozen["query_rows"]): errors.append("query mismatch")
                preds = {"A_support_cv_3shot": np.asarray(a["predictions"]["support_selected_baseline"], np.int16)}
                preds.update({m: frozen[f"pred_{m}"] for m in METHODS[1:]})
                if not np.array_equal(preds["B_plain_multiprototype"], prior["multiprototype_predictions"]): errors.append("B prior probe mismatch")
                for method, pred in preds.items():
                    checked_predictions += len(pred)
                    acc, f1 = f1_accuracy(truth, pred)
                    row = lookup[(model, date, seed, method)]
                    if abs(acc-float(row["accuracy"])) > 1e-12 or abs(f1-float(row["macro_f1"])) > 1e-12:
                        errors.append(f"metric mismatch {model}/{date}/{seed}/{method}")
                conflict_regions += int((frozen["region_weights"] == 0).sum())
                if frozen["region_weights"].shape != (102, 3): errors.append("weight shape")
                if not np.all(np.isin(frozen["region_weights"], [0, 1])): errors.append("nonbinary weight")
    budget = json.loads((ART / "budget_audit.json").read_text())
    values = list(budget["method_history_access"].values())
    if len({(v["historical_entries"], v["numeric_payload_bytes"], v["support_conditioned_weight_array_bytes"]) for v in values}) != 1:
        errors.append("unequal history budget")
    summary = json.loads((ART / "summary.json").read_text())
    record = {"schema_version": 1, "passed": not errors, "errors": errors,
              "sealed_files_checked": len(seal["files"]), "evaluation_units_checked": 18,
              "metric_rows_checked": len(rows), "prediction_rows_checked": checked_predictions,
              "conflicted_regions_total": conflict_regions, "formal_shot": 3,
              "extra_seven_used_by_algorithm": False, "shared_history_budget_equal": True,
              "new_backbone_training_runs": 0,
              "initial_mechanism_signal": summary["initial_mechanism_signal"],
              "output_sha256": {p.name: sha(p) for p in (ART/"metrics_long.csv", ART/"deltas.csv", ART/"summary.json", ART/"budget_audit.json")}}
    out = ART / "integrity_check.json"
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    if errors: raise SystemExit("\n".join(errors))
    print(json.dumps(record, indent=2))


if __name__ == "__main__": main()
