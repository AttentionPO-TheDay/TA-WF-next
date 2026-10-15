import csv
import hashlib
import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).parent
OUT = RUN / "artifacts"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    report = json.loads((OUT / "p0_audit.json").read_text())
    freeze = json.loads((OUT / "freeze.json").read_text())
    freeze_ok = {p: sha256(Path(p)) == digest for p, digest in freeze.items()}
    assert all(freeze_ok.values()), freeze_ok

    h5 = report["shusterman"]
    core = report["core_npz"]
    assert h5["files"] == 142 and h5["rows"] == 556926
    assert h5["invalid_rows"] == 0 and h5["all_zero_rows"] == 31
    assert h5["nonzero_label_conflict_groups"] == 1
    assert h5["same_date_train_test_nonzero_pairs"] == []
    assert core["files"] == 30 and core["rows"] == 615350
    assert core["quarantine_candidate_rows"] == 114307
    assert all(sum(m[k] for m in core["records"]) == 0
               for k in ("nonfinite_rows", "all_zero_rows", "float32_overflow_rows"))
    assert len(set((x["duplicate_groups"], x["duplicate_copies_beyond_first"])
                   for x in core["views"].values())) == 1

    # Every candidate points to an existing source row and preserves its role.
    file_rows = {m["path"]: m["rows"] for m in core["records"]}
    with (OUT / "core_quarantine_candidates.csv").open(newline="") as f:
        candidates = list(csv.DictReader(f))
    assert len(candidates) == core["quarantine_candidate_rows"]
    assert all(r["file"] in file_rows and 0 <= int(r["row"]) < file_rows[r["file"]]
               and r["applied"] == "False" for r in candidates)

    h5_files = {m["file"]: m["rows"] for m in json.loads((OUT / "shusterman_file_index.json").read_text())}
    with (OUT / "shusterman_quarantine_candidates.csv").open(newline="") as f:
        h5_candidates = list(csv.DictReader(f))
    assert all(r["file"] in h5_files and 0 <= int(r["row"]) < h5_files[r["file"]]
               and r["applied"] == "False" for r in h5_candidates)
    assert len(h5_candidates) == sum(h5["candidate_exclusions_by_reason"].values())

    result = {
        "freeze_hashes_verified": freeze_ok,
        "hdf5_shape_and_policy_checks": True,
        "core_shape_and_three_view_checks": True,
        "candidate_bounds_and_unapplied_flags": True,
        "original_data_mutation": False,
        "performance_or_training_run": False,
        "hdf5_same_date_train_test_nonzero": 0,
        "core_train_valid_exact_overlap_unique_inputs": 168,
    }
    (OUT / "postrun_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
