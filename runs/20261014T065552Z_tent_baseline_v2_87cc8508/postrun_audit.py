import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
import port_base as B
import runner as Q
from tent_bn1d import configure_model

R, ROOT, C = B.R, B.ROOT, B.C
Q.setup()
scores = json.loads((R / "artifacts/scores.json").read_text())
assert len(scores) == 36
hist = [x for x in scores if x["origin"] == "historical_reference"]
new = [x for x in scores if x["origin"] == "new"]
assert len(hist) == 18 and len(new) == 18
assert {x["condition"] for x in hist} == {"frozen", "no_pseudo_fixed"}
assert {x["condition"] for x in new} == set(C["conditions"])
assert len(list((R / "checkpoints").glob("*.pt"))) == 18
assert len(list((R / "artifacts").glob("update_*.json"))) == 18
assert not list((R / "artifacts").glob("gmm_*.npz"))
assert not list((R / "artifacts").glob("pseudo_*.npz"))
for path, digest in json.loads((R / "artifacts/freeze.json").read_text()).items():
    assert B.sha(ROOT / path) == digest, path

for row in scores:
    path = R / "artifacts" / f"pred_{row['seed']}_{row['role']}_{row['condition']}.npz"
    with np.load(path) as prediction:
        y = prediction["labels"]; pred = prediction["predictions"]
        logits = prediction["logits"]; rows = prediction["rows"]
        assert len(y) == row["rows"] and logits.shape == (len(y), 102)
        assert np.array_equal(pred, logits.argmax(1)) and np.isfinite(logits).all()
        assert abs(accuracy_score(y, pred) - row["accuracy"]) < 1e-12
        assert abs(f1_score(y, pred, labels=np.arange(102), average="macro", zero_division=0) - row["macro_f1"]) < 1e-12
        with np.load(R / "artifacts" / f"pred_{row['seed']}_{row['role']}_frozen.npz") as frozen:
            assert np.array_equal(rows, frozen["rows"]) and np.array_equal(y, frozen["labels"])
            base = frozen["predictions"]
        correct = base == y; wrong = ~correct
        expected = {
            "negative_adaptation_rate": float((correct & (pred != y)).sum() / max(1, correct.sum())),
            "recovery_rate": float((wrong & (pred == y)).sum() / max(1, wrong.sum())),
            "changed_rate": float((base != pred).mean()),
        }
        for key, value in expected.items(): assert abs(value - row[key]) < 1e-12

for row in new:
    assert row["candidate_updates"] == 800
    assert row["gmm_calls"] == row["pseudo_label_calls"] == 0
    if row["condition"] == "tent_bn":
        assert row["executed_updates"] == 800 and row["changed_parameter_tensors"] > 0
    else:
        assert row["executed_updates"] == 0 and row["changed_parameter_tensors"] == 0

# Exact support stream pairing between TENT and its no-update control.
for seed in C["seeds"]:
    for role in C["roles"]:
        cells = [x for x in new if x["seed"] == seed and x["role"] == role]
        assert len(cells) == 2 and len({x["streams"]["target"] for x in cells}) == 1

# Checkpoint reload audit: configure BN batch-stat mode before loading because
# official TENT deliberately removes running-stat buffers.
for row in new:
    target = B.load_prepared(C["prepared"][row["role"]])
    with np.load(R / "artifacts" / f"split_{row['role']}.npz") as split:
        query = split["query_indices"][:128]
    model, _ = B.load_model(row["seed"])
    adapter = configure_model(Q.InputAdapter(model))
    checkpoint = torch.load(R / "checkpoints" / f"{row['seed']}_{row['role']}_{row['condition']}.pt",
                            map_location="cpu", weights_only=True)
    missing, unexpected = adapter.model.load_state_dict(checkpoint["state_dict"], strict=False)
    assert all("running_" in x or "num_batches_tracked" in x for x in missing)
    assert not unexpected
    adapter.model.eval()
    reloaded = B.predict(adapter.model, target["timestamps"][query], target["tam"][query], 128)
    with np.load(R / "artifacts" / f"pred_{row['seed']}_{row['role']}_{row['condition']}.npz") as prediction:
        assert np.allclose(reloaded, prediction["logits"][:128], atol=1e-5, rtol=1e-5)
    del model, adapter
    torch.cuda.empty_cache()

B.save_json("audit.json", {"passed": True, "total_rows": 36, "historical_rows": 18,
                           "new_tasks": 18, "metrics_recomputed": 36,
                           "checkpoints": 18, "checkpoint_reloads": 18,
                           "gmm_calls": 0, "pseudo_label_calls": 0,
                           "batchstats_control": 9, "cost_fields_verified": True,
                           "paired_stream_cells": 9})
print("audit passed")
