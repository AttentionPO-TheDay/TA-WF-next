#!/usr/bin/env python3
"""Post-hoc, prediction-preserving summary for the frozen screening run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


MODELS = ("df", "varcnn_direction")
DOMAINS = ("source_holdout", "day14", "day30", "day90", "day150", "day270")
METHODS = tuple("ABCDE")
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def load(path: Path):
    return json.loads(path.read_text())


def pattern_rows(predictions):
    counts = defaultdict(Counter)
    hits = Counter()
    for truth, chosen, used in zip(
        predictions["truth_scoring_only"],
        predictions["d_selected_reference_and_four_regions"],
        predictions["regional_inference_used"],
    ):
        if not used:
            continue
        ref_position = int(chosen[0])
        for ref_region in chosen[1:]:
            key = (ref_position, int(ref_region))
            counts[key][int(truth)] += 1
            hits[key] += 1
    rows = []
    for key, n in hits.items():
        distribution = counts[key]
        total = sum(distribution.values())
        purity = max(distribution.values()) / total
        entropy = -sum((v / total) * math.log2(v / total) for v in distribution.values())
        rows.append({
            "reference_position": key[0],
            "reference_region": key[1],
            "hits": n,
            "distinct_true_sites": len(distribution),
            "dominant_true_site_fraction": purity,
            "true_site_entropy_bits": entropy,
            "top_true_sites": [{"site": k, "hits": v} for k, v in distribution.most_common(10)],
        })
    return sorted(rows, key=lambda x: (-x["hits"], x["dominant_true_site_fraction"], -x["distinct_true_sites"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    artifacts = run / "artifacts"
    evaluations = {model: load(artifacts / f"{model}_evaluation.json") for model in MODELS}
    predictions = {
        model: {domain: load(artifacts / f"{model}_{domain}_predictions.json") for domain in DOMAINS}
        for model in MODELS
    }

    metric_rows = []
    deltas = []
    costs = {}
    for model in MODELS:
        evaluation = evaluations[model]
        source = evaluation["domains"]["source_holdout"]
        costs[model] = {
            "checkpoint_epoch": evaluation["checkpoint"]["epoch"],
            "source_validation_macro_f1": evaluation["checkpoint"]["selection_value"],
            "reference": evaluation["reference"],
            "domains": {},
        }
        for domain in DOMAINS:
            result = evaluation["domains"][domain]
            for method in METHODS:
                metric_rows.append({"backbone": model, "method": method, "date": domain, **result[method]})
            dc_acc = result["D"]["accuracy"] - result["C"]["accuracy"]
            dc_f1 = result["D"]["macro_f1"] - result["C"]["macro_f1"]
            de_acc = result["D"]["accuracy"] - result["E"]["accuracy"]
            de_f1 = result["D"]["macro_f1"] - result["E"]["macro_f1"]
            deltas.append({
                "backbone": model,
                "date": domain,
                "D_minus_C_accuracy": dc_acc,
                "D_minus_C_macro_f1": dc_f1,
                "future_D_minus_C_minus_source_accuracy": None if domain == "source_holdout" else dc_acc - (source["D"]["accuracy"] - source["C"]["accuracy"]),
                "future_D_minus_C_minus_source_macro_f1": None if domain == "source_holdout" else dc_f1 - (source["D"]["macro_f1"] - source["C"]["macro_f1"]),
                "D_minus_E_accuracy": de_acc,
                "D_minus_E_macro_f1": de_f1,
                "future_D_minus_E_minus_source_accuracy": None if domain == "source_holdout" else de_acc - (source["D"]["accuracy"] - source["E"]["accuracy"]),
                "future_D_minus_E_minus_source_macro_f1": None if domain == "source_holdout" else de_f1 - (source["D"]["macro_f1"] - source["E"]["macro_f1"]),
            })
            costs[model]["domains"][domain] = result["cost"] | result["regional_coverage"]

    with (artifacts / "metrics_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("backbone", "method", "date", *METRICS))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (artifacts / "deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=deltas[0].keys())
        writer.writeheader()
        writer.writerows(deltas)

    local_summary = {}
    model_pattern_sets = {}
    for model in MODELS:
        all_future = {key: [] for key in predictions[model]["day14"]}
        for domain in DOMAINS[1:]:
            for key, value in predictions[model][domain].items():
                if isinstance(value, list):
                    all_future.setdefault(key, []).extend(value)
        patterns = pattern_rows(all_future)
        model_pattern_sets[model] = {(x["reference_position"], x["reference_region"]) for x in patterns[:50]}
        local_summary[model] = {
            "by_domain": {
                domain: evaluations[model]["domains"][domain]["local_diagnostic"]
                | evaluations[model]["domains"][domain]["regional_coverage"]
                for domain in DOMAINS
            },
            "future_top_50_reference_region_patterns": patterns[:50],
            "future_top_50_lowest_purity_patterns": sorted(patterns[:50], key=lambda x: (x["dominant_true_site_fraction"], -x["hits"]))[:20],
        }

    shared_keys = model_pattern_sets["df"] & model_pattern_sets["varcnn_direction"]
    shared_patterns = []
    for ref_position, ref_region in sorted(shared_keys):
        row = {"reference_position": ref_position, "reference_region": ref_region}
        for model in MODELS:
            item = next(x for x in local_summary[model]["future_top_50_reference_region_patterns"] if (x["reference_position"], x["reference_region"]) == (ref_position, ref_region))
            row[model] = item
        shared_patterns.append(row)
    shared_patterns.sort(key=lambda x: -(x["df"]["hits"] + x["varcnn_direction"]["hits"]))

    common_errors = {"by_domain": {}, "repeated_true_to_D_prediction_pairs_with_C_also_wrong": []}
    pair_by_model = {model: Counter() for model in MODELS}
    for domain in DOMAINS:
        left = predictions["df"][domain]
        right = predictions["varcnn_direction"][domain]
        if left["source_file"] != right["source_file"] or left["row_indices"] != right["row_indices"] or left["truth_scoring_only"] != right["truth_scoring_only"]:
            raise ValueError(f"Prediction alignment mismatch for {domain}")
        n_both_c_wrong = n_same_d_wrong = n_same_d_wrong_both_c_wrong = 0
        pairs = Counter()
        for i, truth in enumerate(left["truth_scoring_only"]):
            dc = left["predictions"]["C"][i]
            vc = right["predictions"]["C"][i]
            dd = left["predictions"]["D"][i]
            vd = right["predictions"]["D"][i]
            both_c_wrong = dc != truth and vc != truth
            same_d_wrong = dd == vd and dd != truth
            n_both_c_wrong += both_c_wrong
            n_same_d_wrong += same_d_wrong
            n_same_d_wrong_both_c_wrong += same_d_wrong and both_c_wrong
            if same_d_wrong and both_c_wrong:
                pairs[(int(truth), int(dd))] += 1
            for model, c_pred, d_pred in (("df", dc, dd), ("varcnn_direction", vc, vd)):
                if d_pred != truth and c_pred != truth:
                    pair_by_model[model][(int(truth), int(d_pred))] += 1
        common_errors["by_domain"][domain] = {
            "samples": len(left["row_indices"]),
            "both_C_wrong_same_queries": n_both_c_wrong,
            "both_D_same_wrong_prediction": n_same_d_wrong,
            "both_D_same_wrong_and_both_C_wrong": n_same_d_wrong_both_c_wrong,
            "top_true_to_shared_D_error_pairs": [{"true_site": k[0], "predicted_site": k[1], "count": v} for k, v in pairs.most_common(20)],
        }
    repeated = set(pair_by_model["df"]) & set(pair_by_model["varcnn_direction"])
    common_errors["repeated_true_to_D_prediction_pairs_with_C_also_wrong"] = sorted(({
        "true_site": key[0], "predicted_site": key[1], "df_count": pair_by_model["df"][key],
        "varcnn_direction_count": pair_by_model["varcnn_direction"][key],
    } for key in repeated), key=lambda x: -(x["df_count"] + x["varcnn_direction_count"]))[:100]

    output = {
        "post_hoc_only": True,
        "predictions_modified": False,
        "metrics_rows": len(metric_rows),
        "deltas": deltas,
        "costs": costs,
        "local_matching": local_summary,
        "shared_top_50_reference_region_patterns": shared_patterns,
        "common_errors": common_errors,
    }
    (artifacts / "posthoc_diagnostics.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "metric_rows": len(metric_rows),
        "shared_top_50_patterns": len(shared_patterns),
        "repeated_C_unresolved_D_error_pairs": len(common_errors["repeated_true_to_D_prediction_pairs_with_C_also_wrong"]),
    }, indent=2))


if __name__ == "__main__":
    main()
