#!/usr/bin/env python3
"""Query-sealed equal-budget region relevance evaluation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/exp_62c23528b1774202"
ART = RUN / "artifacts"
DIAG = ROOT / "runs/exp_04faf4088b604155"
SELECT = ROOT / "runs/exp_376fca9354214097"
DONOR = ROOT / "runs/exp_b471517a3e6f41e7"
MODELS = ("df", "varcnn_direction")
DATES = ("day14", "day90", "day270")
SEEDS = (1729, 6238, 20260916)
CLASSES = 102
METHODS = (
    "A_support_cv_3shot",
    "B_plain_multiprototype",
    "C_site_global_weight",
    "D_region_conflict_suppression",
    "D_ablation_uniform_site_retention",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def dependency_paths() -> list[Path]:
    paths = [
        RUN / "METHOD_PLAN_v1.md", RUN / "config.json",
        DONOR / "RECOVERABILITY_PLAN.md", DONOR / "RESULTS.md",
        DONOR / "artifacts/integrity_check.json", DONOR / "artifacts/summary.json",
        DONOR / "artifacts/metrics_long.csv", DONOR / "artifacts/support_manifests.json",
        DONOR / "artifacts/data_isolation_audit.json",
        DIAG / "DIAGNOSTIC_PLAN_v1.md", DIAG / "RESULTS.md",
        DIAG / "artifacts/integrity_check.json", DIAG / "artifacts/summary.json",
        DIAG / "artifacts/firewall_seal_v1.json",
        SELECT / "SUPPORT_SELECTION_PLAN.md", SELECT / "RESULTS.md",
        SELECT / "artifacts/integrity_check.json", SELECT / "artifacts/summary.json",
        SELECT / "artifacts/metrics_long.csv",
    ]
    for model in MODELS:
        paths.extend([
            DIAG / f"artifacts/{model}_source_geometry_v1.npz",
            DIAG / f"artifacts/{model}_source_geometry_v1.json",
            DIAG / f"artifacts/{model}_embedding_cache_v1.json",
        ])
        for date in DATES:
            paths.append(DIAG / f"artifacts/{model}_{date}_embeddings_v1.npy")
            for seed in SEEDS:
                paths.extend([
                    DIAG / f"artifacts/frozen_{model}_{date}_seed{seed}_v1.npz",
                    SELECT / f"artifacts/{model}_{date}_seed{seed}_shot3.json",
                ])
    return paths


def preflight() -> None:
    output = ART / "input_manifest_v1.json"
    paths = dependency_paths()
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"missing dependencies: {missing}")
    for exp in (DONOR, DIAG, SELECT):
        integrity = json.loads((exp / "artifacts/integrity_check.json").read_text())
        if integrity.get("passed") is not True:
            raise ValueError(f"dependency integrity not passed: {exp}")
    record = {
        "schema_version": 1,
        "plan_sha256": sha256_file(RUN / "METHOD_PLAN_v1.md"),
        "files": {str(p): sha256_file(p) for p in paths},
        "models": list(MODELS), "dates": list(DATES), "seeds": list(SEEDS),
        "formal_shot": 3, "extra_seven_algorithm_access": False,
        "query_labels_accessed": False, "new_backbone_training_runs": 0,
        "dependency_integrity_passed": True,
    }
    atomic_json(output, record)
    print(output)


def conflict_weights(exemplars: np.ndarray, thresholds: np.ndarray,
                     support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return binary region weights and frozen own/alien evidence."""
    if exemplars.shape != (CLASSES, 3, 512) or support.shape != (CLASSES, 3, 512):
        raise ValueError("unexpected exemplar/support shape")
    flat_h = exemplars.reshape(CLASSES * 3, 512)
    flat_s = support.reshape(CLASSES * 3, 512)
    all_sim = flat_h @ flat_s.T
    own = np.empty(CLASSES * 3, np.float32)
    for label in range(CLASSES):
        sl = slice(label * 3, label * 3 + 3)
        own[sl] = all_sim[sl, sl].max(axis=1)
        all_sim[sl, sl] = -np.inf
    alien_index = all_sim.argmax(axis=1)
    alien = all_sim[np.arange(CLASSES * 3), alien_index]
    alien_class = alien_index // 3
    conflict = (alien > own) & ((1.0 - alien) <= thresholds[alien_class])
    return ((~conflict).astype(np.float32).reshape(CLASSES, 3),
            own.reshape(CLASSES, 3), alien.reshape(CLASSES, 3),
            alien_class.astype(np.int16).reshape(CLASSES, 3))


def predict_methods(query_z: np.ndarray, exemplars: np.ndarray,
                    current_proto: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    predictions = {name: np.empty(len(query_z), np.int16) for name in METHODS[1:]}
    global_weight = weights.mean(axis=1)
    uniform_weight = weights.min(axis=1)
    for start in range(0, len(query_z), 256):
        z = np.asarray(query_z[start:start + 256], dtype=np.float32)
        hist = (z @ exemplars.reshape(CLASSES * 3, 512).T).reshape(-1, CLASSES, 3)
        current = z @ current_proto.T
        b_score = np.maximum(current, hist.max(axis=2))
        c_hist = -1.0 + global_weight[None, :] * (hist.max(axis=2) + 1.0)
        c_score = np.maximum(current, c_hist)
        d_hist = -1.0 + weights[None, :, :] * (hist + 1.0)
        d_score = np.maximum(current, d_hist.max(axis=2))
        u_hist = -1.0 + uniform_weight[None, :] * (hist.max(axis=2) + 1.0)
        u_score = np.maximum(current, u_hist)
        stop = start + len(z)
        predictions["B_plain_multiprototype"][start:stop] = b_score.argmax(axis=1)
        predictions["C_site_global_weight"][start:stop] = c_score.argmax(axis=1)
        predictions["D_region_conflict_suppression"][start:stop] = d_score.argmax(axis=1)
        predictions["D_ablation_uniform_site_retention"][start:stop] = u_score.argmax(axis=1)
    return predictions


def freeze(model: str, date: str) -> None:
    geom_path = DIAG / f"artifacts/{model}_source_geometry_v1.npz"
    geom = np.load(geom_path, allow_pickle=False)
    exemplars = np.asarray(geom["exemplars"], np.float32)
    thresholds = np.asarray(geom["thresholds"], np.float32)
    exemplar_rows = np.asarray(geom["exemplar_rows"], np.int64)
    library = ART / f"shared_history_{model}_v1.npz"
    if not library.exists():
        atomic_npz(library, exemplars=exemplars, exemplar_rows=exemplar_rows, thresholds=thresholds)
    else:
        existing = np.load(library, allow_pickle=False)
        if not (np.array_equal(existing["exemplars"], exemplars) and
                np.array_equal(existing["exemplar_rows"], exemplar_rows) and
                np.array_equal(existing["thresholds"], thresholds)):
            raise ValueError("shared history mismatch")
    current = np.load(DIAG / f"artifacts/{model}_{date}_embeddings_v1.npy", mmap_mode="r")
    for seed in SEEDS:
        output = ART / f"frozen_{model}_{date}_seed{seed}_v1.npz"
        if output.exists():
            raise FileExistsError(output)
        source = np.load(DIAG / f"artifacts/frozen_{model}_{date}_seed{seed}_v1.npz", allow_pickle=False)
        query_rows = np.asarray(source["query_rows"], np.int64)
        # The only current labels admitted by the algorithm are encoded by class row;
        # columns 3:10 are never sliced, copied, indexed, or used below.
        support_rows = np.asarray(source["support_order"][:, :3], np.int64)
        support = np.asarray(current[support_rows], np.float32)
        current_proto = support.mean(axis=1)
        norms = np.linalg.norm(current_proto, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero current prototype")
        current_proto /= norms
        weights, own, alien, alien_class = conflict_weights(exemplars.copy(), thresholds, support)
        predictions = predict_methods(current[query_rows], exemplars, current_proto, weights)
        atomic_npz(output, query_rows=query_rows, support_rows=support_rows,
                   current_prototypes=current_proto, region_weights=weights,
                   own_similarity=own, alien_similarity=alien,
                   alien_class=alien_class, **{f"pred_{k}": v for k, v in predictions.items()})
        print(output)


def seal() -> None:
    output = ART / "firewall_seal_v1.json"
    required = [RUN / "METHOD_PLAN_v1.md", RUN / "config.json",
                RUN / "code/run_region_relevance.py", RUN / "code/verify_region_relevance.py",
                ART / "input_manifest_v1.json"]
    required += [ART / f"shared_history_{m}_v1.npz" for m in MODELS]
    required += [ART / f"frozen_{m}_{d}_seed{s}_v1.npz"
                 for m in MODELS for d in DATES for s in SEEDS]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError(f"missing freeze artifacts: {missing}")
    atomic_json(output, {
        "schema_version": 1, "sealed_before_new_query_scoring": True,
        "files": {str(p): sha256_file(p) for p in required},
        "models": list(MODELS), "dates": list(DATES), "seeds": list(SEEDS),
        "formal_shot": 3, "extra_seven_algorithm_access": False,
        "query_labels_used_for_rules_or_predictions": False,
        "new_backbone_training_runs": 0,
    })
    print(output)


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    conf = np.zeros((CLASSES, CLASSES), np.int64)
    np.add.at(conf, (truth, pred), 1)
    tp = np.diag(conf).astype(float)
    actual, guessed = conf.sum(1), conf.sum(0)
    precision = np.divide(tp, guessed, out=np.zeros(CLASSES), where=guessed != 0)
    recall = np.divide(tp, actual, out=np.zeros(CLASSES), where=actual != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(CLASSES), where=(precision + recall) != 0)
    return {"accuracy": float(tp.sum() / conf.sum()), "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()),
            "per_class_accuracy": recall.tolist()}


def score() -> None:
    seal_record = json.loads((ART / "firewall_seal_v1.json").read_text())
    for name, expected in seal_record["files"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError(f"seal mismatch: {name}")
    metric_rows, unit_records = [], []
    for model in MODELS:
        for date in DATES:
            for seed in SEEDS:
                frozen_path = ART / f"frozen_{model}_{date}_seed{seed}_v1.npz"
                frozen = np.load(frozen_path, allow_pickle=False)
                a_path = SELECT / f"artifacts/{model}_{date}_seed{seed}_shot3.json"
                a = json.loads(a_path.read_text())
                query = np.asarray(a["common_query_rows"], np.int64)
                truth = np.asarray(a["common_query_truth"], np.int16)
                if not np.array_equal(query, frozen["query_rows"]):
                    raise ValueError("common query mismatch")
                predictions = {"A_support_cv_3shot": np.asarray(a["predictions"]["support_selected_baseline"], np.int16)}
                predictions.update({name: frozen[f"pred_{name}"] for name in METHODS[1:]})
                unit = {"backbone": model, "date": date, "seed": seed,
                        "query_rows_sha256": hashlib.sha256(query.tobytes()).hexdigest(),
                        "query_count": len(query), "methods": {},
                        "conflicted_regions": int((frozen["region_weights"] == 0).sum()),
                        "sites_with_conflict": int(np.any(frozen["region_weights"] == 0, axis=1).sum())}
                for method in METHODS:
                    m = metrics(truth, predictions[method])
                    unit["methods"][method] = {k: v for k, v in m.items() if k != "per_class_accuracy"}
                    metric_rows.append({"backbone": model, "date": date, "seed": seed,
                                        "method": method, "query_count": len(query),
                                        "accuracy": m["accuracy"], "macro_precision": m["macro_precision"],
                                        "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"]})
                unit_records.append(unit)
    write_csv(ART / "metrics_long.csv", metric_rows)
    by_key = {(r["backbone"], r["date"], int(r["seed"]), r["method"]): r for r in metric_rows}
    deltas = []
    for model in MODELS:
        for date in DATES:
            for seed in SEEDS:
                d = by_key[(model, date, seed, "D_region_conflict_suppression")]["macro_f1"]
                for comparator in ("A_support_cv_3shot", "B_plain_multiprototype", "C_site_global_weight",
                                   "D_ablation_uniform_site_retention"):
                    value = d - by_key[(model, date, seed, comparator)]["macro_f1"]
                    deltas.append({"backbone": model, "date": date, "seed": seed,
                                   "contrast": f"D_minus_{comparator}", "macro_f1_delta": value})
    write_csv(ART / "deltas.csv", deltas)

    def ds(comp, model=None, dates=("day90", "day270")):
        name = f"D_minus_{comp}"
        return [float(r["macro_f1_delta"]) for r in deltas if r["contrast"] == name
                and r["date"] in dates and (model is None or r["backbone"] == model)]

    comparators = ("A_support_cv_3shot", "B_plain_multiprototype", "C_site_global_weight")
    late = {}
    for comp in comparators:
        values = ds(comp)
        late[comp] = {"mean": float(np.mean(values)), "positive_units": int(np.sum(np.asarray(values) > 0)),
                      "by_backbone": {m: float(np.mean(ds(comp, m))) for m in MODELS},
                      "by_date": {d: float(np.mean(ds(comp, dates=(d,)))) for d in ("day90", "day270")}}
    day14_harm = {m: float(np.mean(ds("A_support_cv_3shot", m, ("day14",)))) for m in MODELS}
    ab_values = ds("D_ablation_uniform_site_retention")
    ablation = {"late_mean": float(np.mean(ab_values)),
                "late_positive_units": int(np.sum(np.asarray(ab_values) > 0))}
    success_checks = {
        "late_vs_A_B_C_at_least_1pp_and_9of12": all(late[c]["mean"] >= .01 and late[c]["positive_units"] >= 9 for c in comparators),
        "both_backbones_and_both_late_dates_positive": all(all(v > 0 for v in late[c]["by_backbone"].values()) and all(v > 0 for v in late[c]["by_date"].values()) for c in comparators),
        "day14_no_more_than_1pp_harm_vs_A": all(v >= -.01 for v in day14_harm.values()),
        "region_ablation_loses_0_5pp_and_9of12": ablation["late_mean"] >= .005 and ablation["late_positive_units"] >= 9,
        "budget_and_firewall": True,
    }
    equivalence = {comp: all(abs(float(np.mean(ds(comp, m)))) < .01 for m in MODELS)
                   for comp in ("B_plain_multiprototype", "C_site_global_weight")}
    stop = equivalence["B_plain_multiprototype"] or equivalence["C_site_global_weight"] or not success_checks["region_ablation_loses_0_5pp_and_9of12"]
    summary = {"schema_version": 1, "unit_records": unit_records, "late_deltas": late,
               "day14_D_minus_A": day14_harm, "ablation": ablation,
               "success_checks": success_checks, "initial_mechanism_signal": all(success_checks.values()),
               "substantially_equivalent_to_D_in_both_backbones": equivalence,
               "stop_complex_region_route": stop, "new_backbone_training_runs": 0}
    atomic_json(ART / "summary.json", summary)
    raw_payload = CLASSES * 3 * 512 * 4 + CLASSES * 3 * 8 + CLASSES * 4
    libraries = {m: {"path": str(ART / f"shared_history_{m}_v1.npz"),
                     "sha256": sha256_file(ART / f"shared_history_{m}_v1.npz"),
                     "actual_serialized_bytes": (ART / f"shared_history_{m}_v1.npz").stat().st_size,
                     "numeric_payload_bytes": raw_payload, "historical_entries": CLASSES * 3}
                 for m in MODELS}
    budget = {"schema_version": 1, "shared_history_libraries": libraries,
              "method_history_access": {m: {"same_shared_library": True, "historical_entries": 306,
                                             "numeric_payload_bytes": raw_payload,
                                             "support_conditioned_weight_array_bytes": 1224}
                                        for m in METHODS[1:]},
              "A_representation_note": "existing frozen support-CV uses its prior source heads/candidates; reported separately, not an exemplar-library representation",
              "formal_current_labels_per_class": 3, "current_support_rows": 306,
              "extra_seven_used_by_algorithm": False, "new_backbone_training_runs": 0}
    atomic_json(ART / "budget_audit.json", budget)
    print(json.dumps({"initial_mechanism_signal": summary["initial_mechanism_signal"],
                      "stop_complex_region_route": stop, "late_deltas": late,
                      "day14_D_minus_A": day14_harm, "ablation": ablation}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("preflight")
    freeze_p = sub.add_parser("freeze")
    freeze_p.add_argument("--model", choices=MODELS, required=True)
    freeze_p.add_argument("--date", choices=DATES, required=True)
    sub.add_parser("seal")
    sub.add_parser("score")
    args = parser.parse_args()
    if args.stage == "preflight": preflight()
    elif args.stage == "freeze": freeze(args.model, args.date)
    elif args.stage == "seal": seal()
    else: score()


if __name__ == "__main__":
    main()
