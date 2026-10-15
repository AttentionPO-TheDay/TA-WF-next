import json, os
import numpy as np
import torch
import port_base as B
import runner as Q

R, ROOT, C = B.R, B.ROOT, B.C
Q.setup()
assert C["status"] == "frozen"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(C["physical_gpu"])
assert torch.cuda.is_available()
assert C["permissions"]["support_labels"] is False
assert C["permissions"]["query_adaptation"] is False

source = B.load_prepared(C["prepared"]["source_valid"], "source")
valid = B.load_prepared(C["prepared"]["source_valid"], "valid")
source_all = {k: np.concatenate([source[k], valid[k]]) for k in source}
checked = {}
for role in C["roles"]:
    target = B.load_prepared(C["prepared"][role])
    support, query, _ = B.split_targets(source_all, role, target)
    with np.load(ROOT / C["comparison_run"] / "artifacts" / f"split_{role}.npz") as old:
        assert np.array_equal(support, old["support_indices"])
        assert np.array_equal(query, old["query_indices"])
    checked[role] = {"support_rows": int(len(support)), "query_rows": int(len(query)),
                     "same_historical_split": True, "exact_input_disjoint": True}

role = C["roles"][0]
target = B.load_prepared(C["prepared"][role])
support, _, _ = B.split_targets(source_all, role, target)
support_data = {k: target[k][support] for k in ("timestamps", "tam", "rows")}
for condition in C["conditions"]:
    model, _ = B.load_model(C["seeds"][0])
    cost = Q.train_condition(model, source, support_data, condition, C["seeds"][0], role, max_steps=2)
    assert cost["candidate_updates"] == 2
    assert cost["gmm_calls"] == cost["pseudo_label_calls"] == 0
    if condition == "tent_bn":
        assert cost["executed_updates"] == 2 and cost["changed_parameter_tensors"] > 0
    else:
        assert cost["executed_updates"] == 0 and cost["changed_parameter_tensors"] == 0
    del model
    torch.cuda.empty_cache()

B.save_json("preflight.json", {"passed": True, "gpu": torch.cuda.get_device_name(0),
                               "splits": checked, "candidate_smoke_updates": 2,
                               "official_tent_source": True, "bn1d_extension": True,
                               "gmm_calls": 0, "pseudo_label_calls": 0,
                               "config_sha": B.sha(R / "config.json"),
                               "runner_sha": B.sha(R / "runner.py")})
print("preflight passed", R.name, flush=True)
