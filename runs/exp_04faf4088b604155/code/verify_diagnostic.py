#!/usr/bin/env python3
"""Read-only integrity verification for the completed frozen diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/exp_04faf4088b604155"
ART = RUN / "artifacts"
MODELS = ("df", "varcnn_direction")
DATES = ("day14", "day90", "day270")
SEEDS = (1729, 6238, 20260916)
CLASSES = 102


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def main() -> None:
    errors: list[str] = []
    manifest = json.loads((ART / "input_manifest_v1.json").read_text())
    seal = json.loads((ART / "firewall_seal_v1.json").read_text())
    summary = json.loads((ART / "summary.json").read_text())

    if not manifest.get("passed") or manifest.get("new_backbone_training_runs") != 0:
        errors.append("input manifest did not pass or training count is nonzero")
    if not seal.get("sealed_before_query_label_access") or seal.get("query_labels_used_for_rules_predictions_or_metrics"):
        errors.append("query-label firewall flags are invalid")
    for raw_path, expected in seal["files"].items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"sealed hash mismatch: {path}")

    frozen = sorted(ART.glob("frozen_*_v1.npz"))
    if len(frozen) != 18:
        errors.append(f"expected 18 frozen files, found {len(frozen)}")
    expected_fields = {
        "query_rows", "support_order", "subset_predictions", "ten_predictions",
        "multiprototype_predictions", "support_metrics", "added_category",
        "added_d3", "added_dsrc", "center_shift", "scale3", "scale10",
    }
    for path in frozen:
        arrays = np.load(path)
        if set(arrays.files) != expected_fields:
            errors.append(f"unexpected frozen fields: {path.name}")
            continue
        query_count = len(arrays["query_rows"])
        expected_shapes = {
            "support_order": (CLASSES, 10),
            "subset_predictions": (120, query_count),
            "ten_predictions": (query_count,),
            "multiprototype_predictions": (query_count,),
            "support_metrics": (120, CLASSES, 4),
            "added_category": (CLASSES, 7),
        }
        for key, shape in expected_shapes.items():
            if arrays[key].shape != shape:
                errors.append(f"bad shape {path.name}:{key}={arrays[key].shape}")
        if not set(np.unique(arrays["added_category"])).issubset({0, 1, 2}):
            errors.append(f"bad support category in {path.name}")
        if set(arrays["query_rows"]).intersection(arrays["support_order"].ravel()):
            errors.append(f"support/query overlap in {path.name}")

    expected_rows = {
        "resampling_results.csv": 18 * 120,
        "per_site_stats.csv": 18 * 120 * CLASSES,
        "support_only_metrics.csv": 18 * 120 * CLASSES,
        "probe_results.csv": 18,
        "error_attribution.csv": sum(unit["repaired"] + unit["harmed"] for unit in summary["unit_summaries"]),
    }
    for filename, expected in expected_rows.items():
        actual = csv_rows(ART / filename)
        if actual != expected:
            errors.append(f"row count {filename}: expected {expected}, found {actual}")

    if len(summary.get("unit_summaries", [])) != 18:
        errors.append("summary does not contain 18 units")
    adjudication = summary.get("adjudication", {})
    if adjudication.get("A_oracle_units") != sum((unit["oracle_explain"] or 0) >= 0.5 for unit in summary["unit_summaries"]):
        errors.append("A oracle count mismatch")
    if adjudication.get("A_near10_units") != sum(unit["p_within_2pp_ten"] >= 0.10 for unit in summary["unit_summaries"]):
        errors.append("A near-10 count mismatch")
    if summary.get("new_backbone_training_runs") != 0:
        errors.append("summary training count is nonzero")
    if list(ART.glob("*.tmp")) or list(ART.glob("*.tmp.*")):
        errors.append("temporary artifact remains")

    output_files = [
        ART / "summary.json", ART / "error_attribution_summary.json", ART / "probe_results.json",
        *(ART / filename for filename in expected_rows),
    ]
    record = {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "sealed_file_count": len(seal["files"]),
        "frozen_unit_count": len(frozen),
        "output_row_counts": {filename: csv_rows(ART / filename) for filename in expected_rows},
        "output_sha256": {path.name: sha256_file(path) for path in output_files},
        "adjudication": adjudication,
        "new_backbone_training_runs": 0,
    }
    output = ART / "integrity_check.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
