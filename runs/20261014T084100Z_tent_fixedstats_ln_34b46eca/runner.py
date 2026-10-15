"""Official Tent entropy adaptation, ported to the audited WF model."""
import hashlib, json, math, os, random, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import port_base as B
from enhanced_tent import collect_params, configure_model, forward_and_adapt

R, ROOT, C, DEVICE = B.R, B.ROOT, B.C, B.DEVICE
CONDS = C["conditions"]
A = C["adaptation"]


def setup():
    assert C["status"] == "frozen" and C["id"] == R.name
    torch.set_num_threads(C["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def snapshot_gpu():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader"], capture_output=True, text=True, check=True
    ).stdout.strip()


def status_update(summary):
    import fcntl
    with (ROOT / ".status.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = ROOT / "STATUS.md"
        lines = p.read_text().splitlines()
        prefix = f"当前图片实验 {CONDS[0]}：`{R.name}`。"
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                lines[i] = prefix + summary
                break
        else:
            lines.insert(1, prefix + summary)
        p.write_text("\n".join(lines) + "\n")


class InputAdapter(nn.Module):
    """Official Tent's single-input interface over timestamp/TAM tensors."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        timestamps, tam = batch
        return B.forward_with_features(self.model, timestamps, tam)[0]


def loader(data, target=False):
    tensors = [torch.from_numpy(data[k]) for k in ("timestamps", "tam", "rows")]
    return DataLoader(
        TensorDataset(*tensors), batch_size=128, shuffle=True, drop_last=True,
        num_workers=0, generator=torch.Generator().manual_seed(2026 if target else 2025)
    )


def take(iterator, data_loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(data_loader)
        return next(iterator), iterator


def clock_start():
    torch.cuda.synchronize()
    return time.perf_counter()


def clock_end(start):
    torch.cuda.synchronize()
    return time.perf_counter() - start


@torch.no_grad()
def query_predict(model, timestamps, tam):
    model.eval()
    return B.predict(model, timestamps, tam, 128)


def norm_diagnostics(model, logits):
    pred = logits.argmax(1)
    counts = np.bincount(pred, minlength=102)
    return {
        "finite_logits": bool(np.isfinite(logits).all()),
        "unique_predicted_classes": int((counts > 0).sum()),
        "max_predicted_class_fraction": float(counts.max() / max(1, len(pred))),
        "mean_query_entropy": float(B.entropy(torch.from_numpy(logits)).mean()),
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def set_tent_model(model):
    adapter = InputAdapter(model).to(DEVICE)
    adapter = configure_model(adapter)
    params, names = collect_params(adapter)
    assert len(params) == 22 and sum(p.numel() for p in params) == 3072
    optimizer = torch.optim.Adam(params, lr=A["base_learning_rate"],
                                 betas=tuple(A["betas"]), weight_decay=A["weight_decay"])
    return adapter, optimizer, params, names


def train_condition(model, source, support, condition, seed, role, max_steps=None):
    assert condition in CONDS
    random.seed(2024); np.random.seed(2024); torch.manual_seed(2024)
    source_loader = loader(source, target=False)
    target_loader = loader(support, target=True)
    source_it, target_it = iter(source_loader), iter(target_loader)
    adapter, optimizer, params, names = set_tent_model(model)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rows = []
    streams = {k: hashlib.sha256() for k in ("source", "target")}
    begin = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    peak = 0
    steps = A["candidate_updates"] if max_steps is None else max_steps
    for step in range(steps):
        if time.monotonic() - begin > C["budget"]["max_task_seconds"]:
            raise TimeoutError("task budget exceeded")
        sb, source_it = take(source_it, source_loader)
        tb, target_it = take(target_it, target_loader)
        sx, st, sy = [v.to(DEVICE) for v in sb]
        tx, tt, _ = [v.to(DEVICE) for v in tb]
        streams["source"].update(sb[-1].numpy().astype("<i8").tobytes())
        streams["target"].update(tb[-1].numpy().astype("<i8").tobytes())
        start = clock_start()
        torch.manual_seed(2024 + 100000 + step)
        with torch.enable_grad():
            outputs = forward_and_adapt((tx, tt), adapter, optimizer)
        assert torch.isfinite(outputs).all()
        seconds = clock_end(start)
        peak = max(peak, torch.cuda.max_memory_allocated())
        ent = float(B.entropy(outputs).mean().detach().cpu())
        rows.append({"step": step, "entropy_nats": ent, "update_seconds": seconds,
                     "executed": condition == "tent_fixedstats_ln", "batch": len(tx)})
        if step % 80 == 79 or step == steps - 1:
            B.event("adaptation", seed=seed, role=role, condition=condition,
                    candidates=step + 1, executed=(step + 1 if condition == "tent_fixedstats_ln" else 0))
    model.eval()
    for key, value in model.state_dict().items():
        if key not in {n.removeprefix("model.") for n in names}:
            assert torch.equal(value.cpu(), initial[key]), key
    delta = original = 0.0
    changed = 0
    for name, parameter in model.named_parameters():
        before = initial[name].to(parameter.device)
        delta += float((parameter.detach().double() - before.double()).square().sum())
        original += float(before.double().square().sum())
        changed += int(not torch.equal(parameter.detach(), before))
    if condition == "tent_fixedstats_ln":
        assert changed > 0
    key = f"{seed}_{role}_{condition}"
    B.save_json(f"update_{key}.json", rows)
    totals = sum(x["update_seconds"] for x in rows)
    return {
        "candidate_updates": len(rows), "executed_updates": len(rows) if condition == "tent_fixedstats_ln" else 0,
        "skipped_updates": 0 if condition == "tent_fixedstats_ln" else len(rows),
        "task_seconds": time.monotonic() - begin, "adaptation_update_seconds": totals,
        "peak_allocated_bytes": peak, "streams": {k: v.hexdigest() for k, v in streams.items()},
        "relative_parameter_change": math.sqrt(delta / max(original, 1e-12)),
        "changed_parameter_tensors": changed, "trainable_parameter_count": sum(p.numel() for p in params),
        "trainable_parameter_names": names, "source_labels_used": False,
        "target_entropy_final_batch": rows[-1]["entropy_nats"], "gmm_calls": 0,
        "pseudo_label_calls": 0, "tent_loss": "softmax_entropy(logits).mean()",
        "tent_steps_per_batch": 1, "batch_statistics": False, "source_bn_buffers_unchanged": True,
    }


def freeze_inputs():
    hist_files = []
    for alias, relative in C["historical_references"].items():
        p = ROOT / relative
        hist_files.append(p)
        hist_files.extend(p.parent / f"pred_{s}_{r}_{alias}.npz" for s in C["seeds"] for r in C["roles"])
    files = list(R.glob("*.py")) + [R / "PLAN.md", R / "SOURCES.md", R / "config.json",
                                    R / "official_tent.py", R / "tent_bn1d.py", R / "TENT_README.md"]
    files += [ROOT / C["dataset_config"]] + [ROOT / p for p in C["prepared"].values()]
    files += [ROOT / C["base_checkpoint_run"] / "checkpoints" / f"wide_bn_{s}_best.pt" for s in C["seeds"]]
    files += [ROOT / C["comparison_run"] / "artifacts" / f"split_{r}.npz" for r in C["roles"]]
    files += hist_files
    B.save_json("freeze.json", {str(p.relative_to(ROOT)): B.sha(p) for p in files})


def historical_rows():
    out = []
    for alias, relative in C["historical_references"].items():
        old = json.loads((ROOT / relative).read_text())
        picked = [r for r in old if r["condition"] == alias]
        assert len(picked) == 9
        for row in picked:
            x = {k: row[k] for k in ("seed", "role", "rows", "accuracy", "macro_f1",
                                     "negative_adaptation_rate", "recovery_rate", "changed_rate")}
            x.update(condition=alias, origin="historical_reference", source_scores=relative)
            out.append(x)
            shutil.copy2((ROOT / relative).parent / f"pred_{row['seed']}_{row['role']}_{alias}.npz",
                         R / "artifacts" / f"pred_{row['seed']}_{row['role']}_{alias}.npz")
    assert len(out) == C["budget"]["historical_rows"]
    return out


def report(scores):
    from reporting import write_report
    return write_report(scores)


def main():
    setup(); started = time.time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("batch budget exceeded")))
    signal.alarm(C["budget"]["max_wall_seconds"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(C["physical_gpu"]) and torch.cuda.is_available()
    # Preflight intentionally writes events.jsonl; use the launch marker and a one-shot runner marker for duplicate protection.
    assert (R / "artifacts/launch.json").exists()
    with (R / "artifacts/runner_started.json").open("x") as marker:
        json.dump({"utc": time.time(), "run_id": R.name}, marker)
    preflight = json.loads((R / "artifacts/preflight.json").read_text())
    assert preflight["passed"] and preflight["config_sha"] == B.sha(R / "config.json")
    for name, digest in preflight["code_sha"].items():
        assert B.sha(R / name) == digest, name
    freeze_inputs()
    B.save_json("hardware_start.json", {"snapshot": snapshot_gpu(), "gpu": torch.cuda.get_device_name(0)})
    B.registry("running", "9项固定源BN统计量+LN纯熵增强已启动；36条历史参考。")
    status_update("9项固定源BN统计量+LN纯熵增强已启动；36条历史参考。")
    source = B.load_prepared(C["prepared"]["source_valid"], "source")
    valid = B.load_prepared(C["prepared"]["source_valid"], "valid")
    source_all = {k: np.concatenate([source[k], valid[k]]) for k in source}
    splits = {}
    for role in C["roles"]:
        target = B.load_prepared(C["prepared"][role]); s, q, _ = B.split_targets(source_all, role, target)
        with np.load(ROOT / C["comparison_run"] / "artifacts" / f"split_{role}.npz") as old:
            assert np.array_equal(s, old["support_indices"]) and np.array_equal(q, old["query_indices"])
        np.savez_compressed(R / "artifacts" / f"split_{role}.npz", support=target["rows"][s], query=target["rows"][q], support_indices=s, query_indices=q)
        splits[role] = (s, q)
    scores = historical_rows(); B.save_json("scores.json", scores); B.event("preflight_verified", new_tasks=9, historical_rows=36, splits_match=True)
    for seed in C["seeds"]:
        for role in C["roles"]:
            target = B.load_prepared(C["prepared"][role]); s, q = splits[role]
            support = {k: target[k][s] for k in ("timestamps", "tam", "rows")}
            paired = None
            for condition in CONDS:
                model, _ = B.load_model(seed)
                cost = train_condition(model, source, support, condition, seed, role)
                reference = next(x for x in json.loads((ROOT / C["historical_references"]["no_pseudo_fixed"]).read_text()) if x["seed"] == seed and x["role"] == role and x["condition"] == "no_pseudo_fixed")
                assert reference["streams"] == cost["streams"] if "streams" in reference else True
                paired = cost["streams"] if paired is None else paired
                assert paired == cost["streams"]
                start = clock_start(); logits = query_predict(model, target["timestamps"][q], target["tam"][q]); query_seconds = clock_end(start)
                pred = logits.argmax(1); frozen = np.load(R / "artifacts" / f"pred_{seed}_{role}_frozen.npz")["predictions"]
                row = {"seed": seed, "role": role, "condition": condition, "origin": "new", "rows": len(q),
                       **B.cell_metrics(target["labels"][q], frozen, pred), **cost,
                       "query_inference_seconds": query_seconds,
                       "query_latency_ms_per_trace": 1000 * query_seconds / len(q), **norm_diagnostics(model, logits)}
                np.savez_compressed(R / "artifacts" / f"pred_{seed}_{role}_{condition}.npz", rows=target["rows"][q], labels=target["labels"][q], predictions=pred, logits=logits)
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "seed": seed, "role": role, "condition": condition, "config_sha": B.sha(R / "config.json")}, R / "checkpoints" / f"{seed}_{role}_{condition}.pt")
                scores.append(row); B.save_json("scores.json", scores)
                B.save_json("progress.json", {"completed_tasks": sum(x["origin"] == "new" for x in scores), "total_tasks": 9, "seed": seed, "role": role, "condition": condition, "elapsed_seconds": time.time() - started})
                B.event("task_completed", seed=seed, role=role, condition=condition, completed=sum(x["origin"] == "new" for x in scores))
                del model; torch.cuda.empty_cache()
    subprocess.run([sys.executable, str(R / "postrun_audit.py")], check=True)
    summary = report(scores)
    B.save_json("hardware_end.json", {"snapshot": snapshot_gpu()})
    B.registry("completed", summary)
    status_update(summary)
    B.event("completed", elapsed_seconds=time.time() - started)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        B.save_json("failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        B.registry("failed", f"TENT实验失败：{type(exc).__name__}: {exc}；产物保留。")
        status_update(f"执行失败：{type(exc).__name__}，产物已保留。")
        raise
