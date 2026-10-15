import json
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parent
scores = json.loads((R / "artifacts/scores.json").read_text())
conditions = {row["condition"] for row in scores}
assert len(scores) == 36
assert conditions == {"frozen", "global_entropy_adapter", "stepwise_rollback", "proteus_port"}
by_cell = {(row["seed"], row["role"], row["condition"]): row for row in scores}
assert len(by_cell) == 36


def recompute(labels, predictions):
    confusion = np.bincount(labels * 102 + predictions, minlength=102 * 102).reshape(102, 102)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(102), where=denominator > 0)
    return float((labels == predictions).mean()), float(f1.mean())


for row in scores:
    prediction_path = R / "artifacts" / f"pred_{row['seed']}_{row['role']}_{row['condition']}.npz"
    with np.load(prediction_path) as values:
        assert len(values["rows"]) == row["rows"]
        assert len(values["labels"]) == row["rows"]
        assert len(values["predictions"]) == row["rows"]
        assert values["logits"].shape == (row["rows"], 102)
        assert np.isfinite(values["logits"]).all()
        accuracy, macro_f1 = recompute(values["labels"], values["predictions"])
        assert abs(accuracy - row["accuracy"]) < 1e-12
        assert abs(macro_f1 - row["macro_f1"]) < 1e-12
    if row["condition"] == "proteus_port":
        assert row["steps"] == 100
        assert 0 < row["final_pseudo_coverage"] <= 1
        assert len(row["loss_history"]) == 100
        assert len(row["gmm_history"]) == 20
        assert np.isfinite(row["parameter_norm"])
        checkpoint = R / "checkpoints" / f"proteus_{row['seed']}_{row['role']}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert "state_dict" in state and len(state["state_dict"]) > 0
        for key in ("source_ce", "mmd", "entropy_diversity", "pseudo_ce"):
            assert all(np.isfinite(item[key]) for item in row["loss_history"])
        assert all(np.isfinite(item["coverage"]) for item in row["gmm_history"])
    if row["condition"] != "frozen":
        frozen = by_cell[(row["seed"], row["role"], "frozen")]
        with np.load(R / "artifacts" / f"pred_{row['seed']}_{row['role']}_frozen.npz") as base, np.load(prediction_path) as adapted:
            frozen_correct = base["predictions"] == base["labels"]
            frozen_wrong = ~frozen_correct
            negative = float((frozen_correct & (adapted["predictions"] != base["labels"])).sum() / max(1, frozen_correct.sum()))
            recovery = float((frozen_wrong & (adapted["predictions"] == base["labels"])).sum() / max(1, frozen_wrong.sum()))
            changed = float((adapted["predictions"] != base["predictions"]).mean())
            assert abs(negative - row["negative_adaptation_rate"]) < 1e-12
            assert abs(recovery - row["recovery_rate"]) < 1e-12
            assert abs(changed - row["changed_rate"]) < 1e-12

for role in ("day14", "day90", "day270"):
    with np.load(R / "artifacts" / f"split_{role}.npz") as split:
        assert len(np.unique(split["support"])) == len(split["support"])
        assert len(np.unique(split["query"])) == len(split["query"])
        assert set(split["support"]).isdisjoint(set(split["query"]))

assert len(list((R / "artifacts").glob("pred_*.npz"))) == 36
assert len(list((R / "checkpoints").glob("proteus_*.pt"))) == 9
(R / "artifacts/audit.json").write_text(json.dumps({"passed": True, "rows": len(scores), "prediction_files": 36, "proteus_checkpoints": 9}, indent=2) + "\n")
