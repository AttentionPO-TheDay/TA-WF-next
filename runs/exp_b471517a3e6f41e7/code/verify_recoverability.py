#!/usr/bin/env python3
"""Independent integrity verification for completed recoverability artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs" / "exp_b471517a3e6f41e7"
ART = RUN / "artifacts"
OUTPUT = ART / "integrity_check.json"
EXPECTED = {
    RUN / "RECOVERABILITY_PLAN.md": "0db36bb1196f29a993b4b28c7fae5478d1d9d323efe81e58bd4967ed7c9bae7b",
    ART / "data_isolation_audit.json": "6f1da408573f70cdf5ed64b80c9ce7b83ed56691a88208cb86a607403b3a17d6",
    ART / "support_manifests.json": "cbf37c99af375c576e882e049ffb4a37946ea631099d6b15f1ba9458782725ca",
    ART / "df_g_source.joblib": "895df6445a7ef7a068704a00f23f5fb84efbde4df7b6acf48aa5da4e85a27949",
    ART / "df_source_linear.json": "94e863b59257f1544cb2832c441ff9709aea2b9c06f68dd961990985adce8e94",
    ROOT / "runs/exp_9121b664a1854097/checkpoints/df_best.pt": "1bf851278365d0cb716d0ab2d59d6ef43f5b715d50b2a282418151836c9133ae",
    ROOT / "runs/exp_9121b664a1854097/checkpoints/varcnn_direction_best.pt": "fc3ade7932cabde922ed8c4e25a97826e067c2fac612b4d689fe1704cea1ff83",
    ROOT / "runs/exp_6238dacf9aa142cc/artifacts/splits_v3.json": "0f322e9a418f0e96eff3e3d7ceca7237219f2ea6cb910c061fdd7f8025f3f142",
    ROOT / "runs/exp_4b72ed0c9a354d54/artifacts/frozen_selection.json": "4280429d4f5397af8cc2fa422ab20b3345fe6eee384a4c7ab487761fd79e720f",
}
MODELS = ("df", "varcnn_direction")
DATES = ("day14", "day90", "day270")
SEEDS = (1729, 6238, 20260916)
SHOTS = (3, 10)
METHODS = ("A", "G_source", "G_current", "simple_maintenance")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def int_hash(values: list[int]) -> str:
    return hashlib.sha256(b"".join(struct.pack("<q", value) for value in values)).hexdigest()


def metrics(truth: list[int], prediction: list[int]) -> tuple[float, float]:
    confusion = [[0] * 102 for _ in range(102)]
    for actual, predicted in zip(truth, prediction):
        confusion[actual][predicted] += 1
    correct = sum(confusion[label][label] for label in range(102))
    f1 = []
    for label in range(102):
        tp = confusion[label][label]
        actual = sum(confusion[label])
        predicted = sum(confusion[row][label] for row in range(102))
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return correct / len(truth), sum(f1) / 102


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    errors = []
    hashes = {}
    for path, expected in EXPECTED.items():
        actual = sha(path) if path.is_file() else None
        hashes[str(path)] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            errors.append(f"hash mismatch: {path}")
    audit = json.loads((ART / "data_isolation_audit.json").read_text())
    manifest = json.loads((ART / "support_manifests.json").read_text())
    if not audit.get("passed") or audit.get("infeasible"):
        errors.append("audit is not passed and feasible")
    if audit.get("support_manifest_sha256") != sha(ART / "support_manifests.json"):
        errors.append("audit/manifest hash mismatch")
    source = {}
    for model in MODELS:
        record_path = ART / f"{model}_source_linear.json"
        model_path = ART / f"{model}_g_source.joblib"
        record = json.loads(record_path.read_text())
        actual_model_hash = sha(model_path)
        good = (record["model_artifact"]["sha256"] == actual_model_hash and
                record["selected_C"] in (0.1, 1.0, 10.0) and
                record["new_backbone_training_runs"] == 0)
        source[model] = {"record_sha256": sha(record_path), "model_sha256": actual_model_hash,
                         "selected_C": record["selected_C"], "valid": good}
        if not good:
            errors.append(f"invalid source artifact: {model}")
    pattern = re.compile(r"^(df|varcnn_direction)_(day14|day90|day270)_seed(1729|6238|20260916)_shot(3|10)\.json$")
    actual_eval = sorted(path for path in ART.glob("*.json") if pattern.match(path.name))
    if len(actual_eval) != 36:
        errors.append(f"expected 36 evaluation JSON files, got {len(actual_eval)}")
    query_reference = {}
    checked = 0
    prediction_rows = 0
    warnings = []
    artifact_hashes = {}
    for model in MODELS:
        for date in DATES:
            pool = {row["row_index"] for row in manifest["dates"][date]["eligible_rows"]}
            for seed in SEEDS:
                three = manifest["dates"][date]["configurations"][f"seed{seed}_shot3"]
                ten = manifest["dates"][date]["configurations"][f"seed{seed}_shot10"]
                if not set(three["support_rows"]) < set(ten["support_rows"]):
                    errors.append(f"non-nested support: {date}/{seed}")
                for shot in SHOTS:
                    item = manifest["dates"][date]["configurations"][f"seed{seed}_shot{shot}"]
                    path = ART / f"{model}_{date}_seed{seed}_shot{shot}.json"
                    record = json.loads(path.read_text())
                    artifact_hashes[path.name] = sha(path)
                    query = record["query_rows"]
                    truth = record["query_truth"]
                    expected_query = item["query_rows"]
                    if query != expected_query or int_hash(query) != item["query_rows_sha256"]:
                        errors.append(f"query/manifest mismatch: {path.name}")
                    if record["support_rows"] != item["support_rows"] or set(query) != pool - set(item["support_rows"]):
                        errors.append(f"support/complement mismatch: {path.name}")
                    shared_key = (date, seed, shot)
                    if shared_key in query_reference and query_reference[shared_key] != (query, truth):
                        errors.append(f"backbone query/truth mismatch: {date}/{seed}/{shot}")
                    query_reference[shared_key] = (query, truth)
                    if record["new_backbone_training_runs"] != 0:
                        errors.append(f"nonzero backbone training: {path.name}")
                    fit = record["G_current_fit"]
                    if fit["query_labels_used_for_fit_or_selection"] is not False or fit["fit_rows"] != shot * 102 or fit["C_inherited_from_G_source"] != source[model]["selected_C"]:
                        errors.append(f"G-current permission/config mismatch: {path.name}")
                    if fit["convergence_warning"]:
                        warnings.append(path.name)
                    if len(truth) != len(query):
                        errors.append(f"truth length mismatch: {path.name}")
                    for method in METHODS:
                        prediction = record["predictions"].get(method, [])
                        if len(prediction) != len(query):
                            errors.append(f"prediction length mismatch: {path.name}/{method}")
                            continue
                        accuracy, macro_f1 = metrics(truth, prediction)
                        saved = record["metrics"][method]
                        if abs(accuracy - saved["accuracy"]) > 1e-12 or abs(macro_f1 - saved["macro_f1"]) > 1e-12:
                            errors.append(f"metric mismatch: {path.name}/{method}")
                        if len(saved["per_website"]) != 102:
                            errors.append(f"per-website coverage mismatch: {path.name}/{method}")
                    checked += 1
                    prediction_rows += len(query) * len(METHODS)
    with (ART / "metrics_long.csv").open(newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    if len(metric_rows) != 144:
        errors.append(f"metrics_long expected 144 rows, got {len(metric_rows)}")
    if not (ART / "summary.json").is_file():
        errors.append("summary.json missing")
    temporary = sorted(str(path) for path in RUN.rglob("*.tmp"))
    if temporary:
        errors.append("temporary files remain")
    result = {
        "passed": not errors,
        "errors": errors,
        "warnings": {"G_current_convergence_warning_files": warnings},
        "frozen_hashes": hashes,
        "source_artifacts": source,
        "evaluation_artifacts_checked": checked,
        "fixed_split_configurations_shared_across_backbones": len(query_reference),
        "prediction_rows_checked_across_methods": prediction_rows,
        "metrics_long_rows": len(metric_rows),
        "all_per_website_tables_have_102_classes": not any("per-website" in error for error in errors),
        "all_query_predictions_recomputed_metrics_match": not any("metric mismatch" in error for error in errors),
        "all_new_backbone_training_counts_zero": not any("backbone training" in error for error in errors),
        "all_G_current_records_deny_query_label_selection": not any("permission/config" in error for error in errors),
        "temporary_files": temporary,
        "evaluation_artifact_sha256": artifact_hashes,
    }
    temporary_output = OUTPUT.with_suffix(".json.tmp")
    temporary_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary_output.replace(OUTPUT)
    print(json.dumps({key: result[key] for key in ("passed", "errors", "warnings", "evaluation_artifacts_checked", "prediction_rows_checked_across_methods", "metrics_long_rows")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
