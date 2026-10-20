"""Independent file and metric audit for the nine frozen predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
C = json.loads((RUN / "config.json").read_text())
ART = RUN / "artifacts"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(y, pred):
    n = C["classes"]
    cm = np.bincount(y * n + pred, minlength=n * n).reshape(n, n)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), den, out=np.zeros(n), where=den > 0)
    return float((y == pred).mean()), float(f1.mean())


def main():
    frozen = json.loads((ART / "freeze.json").read_text())
    assert all(digest(ROOT / name) == expected for name, expected in frozen.items())
    scores = json.loads((ART / "scores.json").read_text())
    assert len(scores) == C["budget"]["cells"] == 9
    rows = []
    seen = set()
    prior = ROOT / C["interrupted_run"] / "artifacts"
    for score in scores:
        key = (score["seed"], score["role"], score["condition"])
        assert key not in seen and key[2] == "frozen"
        seen.add(key)
        path = ART / f"pred_{key[0]}_{key[1]}_frozen.npz"
        with np.load(path, allow_pickle=False) as prediction:
            y = prediction["labels"]
            pred = prediction["predictions"]
            logits = prediction["logits"]
            sample_rows = prediction["rows"]
        assert logits.shape == (score["rows"], C["classes"])
        assert y.shape == pred.shape == sample_rows.shape == (score["rows"],)
        assert np.isfinite(logits).all() and np.array_equal(pred, logits.argmax(1))
        with np.load(ROOT / C["split_source_run"] / "artifacts" / f"split_{key[1]}.npz", allow_pickle=False) as split:
            assert np.array_equal(sample_rows, split["query"])
        accuracy, macro_f1 = metrics(y, pred)
        assert abs(accuracy - score["accuracy"]) < 1e-12
        assert abs(macro_f1 - score["macro_f1"]) < 1e-12
        old_path = prior / path.name
        old_comparison = None
        if old_path.exists():
            with np.load(old_path, allow_pickle=False) as old:
                assert np.array_equal(sample_rows, old["rows"])
                assert np.array_equal(y, old["labels"])
                old_comparison = {"prediction_mismatches_vs_interrupted_cpu": int((pred != old["predictions"]).sum()), "max_abs_logit_difference_vs_cpu": float(np.max(np.abs(logits - old["logits"])))}
        rows.append({"seed": key[0], "role": key[1], "rows": len(y), "accuracy": accuracy, "macro_f1": macro_f1, "prediction_sha256": digest(path), "prior_cpu_comparison": old_comparison})
    assert seen == {(seed, role, "frozen") for seed in C["seeds"] for role in C["targets"]}
    result = {"passed": True, "cells": len(rows), "frozen_files_verified": len(frozen), "rows": rows}
    (ART / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"audit_passed": True, "cells": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
