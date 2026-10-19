"""Build the run-local source/valid tensor from the audited row manifest."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
CFG = json.loads((RUN / "config.json").read_text())
DATASETS = json.loads((ROOT / CFG["dataset_config"]).read_text())
DATA_ROOT = Path(DATASETS["data_root"])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tam_from_timestamps(x):
    out = np.empty((len(x), 2, 1800), dtype=np.float32)
    for i, row in enumerate(x):
        nz = row[row != 0]
        bins = np.minimum(np.floor(np.abs(nz).astype(np.float64) * 1799 / 80), 1799).astype(np.int64)
        channels = (nz < 0).astype(np.int64)
        out[i] = np.bincount(bins + 1800 * channels, minlength=3600).reshape(2, 1800)
    return out


def load_x_y(path, rows=None):
    with np.load(path, allow_pickle=False) as data:
        raw_x = data["X"]
        raw_y = data["y"]
        if rows is None:
            rows = np.arange(len(raw_y), dtype=np.int64)
        x = np.ascontiguousarray(raw_x[rows, :5000], dtype=np.float32)
        y = np.asarray(raw_y[rows], dtype=np.int64)
        assert np.array_equal(raw_y[rows], y), "nonintegral label"
    assert x.shape == (len(rows), 5000)
    assert np.isfinite(x).all() and (x != 0).any(axis=1).all()
    assert not (((x == 0).cumsum(axis=1) > 0) & (x != 0)).any(), "internal zero"
    assert set(np.unique(y)) == set(range(102))
    return x, y, rows


def row_hashes(x):
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in x]


def main():
    assert CFG["status"] == "draft"
    manifest = ROOT / CFG["candidate_manifest"]
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18553
    assert set(r["file"] for r in rows) == {CFG["source_file"]}
    assert set(r["original_role"] for r in rows) == {"train"}
    assert set(r["applied"] for r in rows) == {"False"}
    assert len({r["raw_group_id"] for r in rows}) == len(rows)
    source_rows = np.array([int(r["row"]) for r in rows], dtype=np.int64)
    assert len(set(source_rows)) == len(source_rows)
    source_path = DATA_ROOT / CFG["source_file"]
    valid_path = DATA_ROOT / CFG["valid_file"]
    sx, sy, sr = load_x_y(source_path, source_rows)
    assert np.array_equal(sy, np.array([int(r["label"]) for r in rows]))
    vx, vy, vr = load_x_y(valid_path)
    sh, vh = row_hashes(sx), row_hashes(vx)
    assert len(set(sh)) == len(sh), "selected source duplicate"
    assert not (set(sh) & set(vh)), "source/valid exact overlap"
    prepared = {
        "source": {"timestamps": torch.from_numpy(sx), "tam": torch.from_numpy(tam_from_timestamps(sx)), "labels": torch.from_numpy(sy), "rows": torch.from_numpy(sr)},
        "valid": {"timestamps": torch.from_numpy(vx), "tam": torch.from_numpy(tam_from_timestamps(vx)), "labels": torch.from_numpy(vy), "rows": torch.from_numpy(vr)},
    }
    out = RUN / "artifacts/prepared.pt"
    torch.save(prepared, out)
    summary = {
        "passed": True,
        "source_rows": len(sx), "valid_rows": len(vx),
        "source_class_min": int(np.bincount(sy, minlength=102).min()),
        "source_class_max": int(np.bincount(sy, minlength=102).max()),
        "source_valid_exact_overlap": 0,
        "valid_unique_inputs": len(set(vh)),
        "sha256": {"candidate_manifest": sha(manifest), "source_npz": sha(source_path), "valid_npz": sha(valid_path), "prepared": sha(out)},
    }
    (RUN / "artifacts/data_preflight.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
