"""Scale-Lock plus local Span-Invariance source-free TTA."""
import hashlib, json, math, os, random, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import port_base as B
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
    return subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()


def status_update(summary):
    import fcntl
    with (ROOT / ".status.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = ROOT / "STATUS.md"
        lines = p.read_text().splitlines()
        prefix = f"当前实验 {CONDS[0]}：`{R.name}`。"
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                lines[i] = prefix + summary
                break
        else:
            lines.insert(1, prefix + summary)
        p.write_text("\n".join(lines) + "\n")


class InputAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        timestamps, tam = batch
        return B.forward_with_features(self.model, timestamps, tam)[0]


def loader(data):
    tensors = [torch.from_numpy(data[k]) for k in ("timestamps", "tam", "rows")]
    return DataLoader(TensorDataset(*tensors), batch_size=A["target_batch_size"], shuffle=True,
                      drop_last=True, num_workers=0,
                      generator=torch.Generator().manual_seed(A["target_loader_seed"]))


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
    return B.predict(model, timestamps, tam, C["eval_batch_size"])


def norm_diagnostics(model, logits):
    pred = logits.argmax(1)
    counts = np.bincount(pred, minlength=102)
    return {"finite_logits": bool(np.isfinite(logits).all()),
            "unique_predicted_classes": int((counts > 0).sum()),
            "max_predicted_class_fraction": float(counts.max() / max(1, len(pred))),
            "mean_query_entropy": float(B.entropy(torch.from_numpy(logits)).mean()),
            "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}


def configure_shot_span_model(model: nn.Module):
    """Freeze output scale/prototypes; open BN affine, time projection/positions and head LayerNorm."""
    model.train()
    model.requires_grad_(False)
    for module in model.generator.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
            module.track_running_stats = True
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
    for module in model.head.modules():
        if isinstance(module, nn.LayerNorm):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    model.time_projection.requires_grad_(True)
    model.time_positions.requires_grad_(True)
    model.final_norm.requires_grad_(False)
    model.classifier.requires_grad_(False)
    params, names = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params.append(parameter)
            names.append(name)
    return params, names


def apply_span_mask(tam: torch.Tensor, step_seed: int, mask_len: int = 90) -> torch.Tensor:
    gen = torch.Generator(device=tam.device).manual_seed(step_seed)
    b = len(tam)
    starts = torch.randint(0, 1800 - mask_len + 1, (b,), generator=gen, device=tam.device)
    bins = torch.arange(1800, device=tam.device)[None, :]
    mask = (bins >= starts[:, None]) & (bins < starts[:, None] + mask_len)
    return tam.masked_fill(mask[:, None, :], 0.0)


@torch.enable_grad()
def forward_and_adapt_invariance(batch, adapter, optimizer, step, total_steps=800,
                                  tau_low=0.20, tau_high=0.60):
    tx, tt = batch
    logits_clean = adapter((tx, tt))
    ent_clean = B.entropy(logits_clean)
    probs_clean = logits_clean.softmax(dim=1).detach()
    tt_masked = apply_span_mask(tt, step_seed=20260000 + step, mask_len=A["mask_len_bins"])
    logits_masked = adapter((tx, tt_masked))
    with torch.no_grad():
        agree = logits_clean.argmax(dim=1).eq(logits_masked.argmax(dim=1))
        keep = (ent_clean >= tau_low) & (ent_clean <= tau_high) & agree
    if not bool(keep.any()):
        return logits_clean, ent_clean.detach(), keep.detach(), agree.detach(), False
    loss_ent = ent_clean[keep].mean()
    loss_consist = -(probs_clean[keep] * logits_masked[keep].log_softmax(dim=1)).sum(dim=1).mean()
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
    loss = cosine_decay * (0.5 * loss_ent + 0.5 * loss_consist)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for parameter in adapter.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), "nonfinite gradient"
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return logits_clean, ent_clean.detach(), keep.detach(), agree.detach(), True


def set_tent_model(model):
    params, base_names = configure_shot_span_model(model)
    adapter = InputAdapter(model).to(DEVICE)
    names = ["model." + name for name in base_names]
    expected = 32384
    assert sum(p.numel() for p in params) == expected
    optimizer = torch.optim.Adam(params, lr=A["base_learning_rate"], betas=tuple(A["betas"]),
                                 weight_decay=A["weight_decay"])
    return adapter, optimizer, params, names


def train_condition(model, support, condition, seed, role, max_steps=None):
    assert condition in CONDS and set(support) == {"timestamps", "tam", "rows"}
    random.seed(2024); np.random.seed(2024); torch.manual_seed(2024)
    target_loader = loader(support)
    target_it = iter(target_loader)
    adapter, optimizer, params, names = set_tent_model(model)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rows, entropies, masks, agrees = [], [], [], []
    streams = {"target": hashlib.sha256()}
    begin = time.monotonic(); torch.cuda.reset_peak_memory_stats(); peak = 0
    steps = A["candidate_updates"] if max_steps is None else max_steps
    for step in range(steps):
        if time.monotonic() - begin > C["budget"]["max_task_seconds"]:
            raise TimeoutError("task budget exceeded")
        tb, target_it = take(target_it, target_loader)
        tx, tt, _ = [v.to(DEVICE) for v in tb]
        streams["target"].update(tb[-1].numpy().astype("<i8").tobytes())
        start = clock_start()
        torch.manual_seed(2024 + 100000 + step)
        outputs, entropy, keep, agree, executed = forward_and_adapt_invariance(
            (tx, tt), adapter, optimizer, step, A["candidate_updates"], A["tau_low_nats"], A["tau_high_nats"])
        assert torch.isfinite(outputs).all() and torch.isfinite(entropy).all()
        seconds = clock_end(start); peak = max(peak, torch.cuda.max_memory_allocated())
        retained = int(keep.sum().item()); agreed = int(agree.sum().item())
        rows.append({"step": step, "entropy_nats": float(entropy.mean().cpu()), "update_seconds": seconds,
                     "executed": bool(executed), "batch": len(tx), "retained_samples": retained,
                     "skipped_samples": len(keep) - retained, "retained_fraction": retained / len(keep),
                     "agreement_samples": agreed, "agreement_fraction": agreed / len(agree),
                     "cosine_weight": 0.5 * (1.0 + math.cos(math.pi * step / A["candidate_updates"]))})
        entropies.append(entropy.cpu().numpy()); masks.append(keep.cpu().numpy()); agrees.append(agree.cpu().numpy())
        if step % 80 == 79 or step == steps - 1:
            B.event("adaptation", seed=seed, role=role, condition=condition,
                    candidates=step + 1, executed=sum(x["executed"] for x in rows))
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for suffix in ("running_mean", "running_var", "num_batches_tracked"):
                assert torch.equal(getattr(module, suffix).cpu(), initial[name + "." + suffix]), name + "." + suffix
    allowed = {name.removeprefix("model.") for name in names}
    for name, parameter in model.named_parameters():
        if name not in allowed:
            assert torch.equal(parameter.detach().cpu(), initial[name]), name
    delta = original = 0.0; changed = 0
    for name, parameter in model.named_parameters():
        before = initial[name].to(parameter.device)
        delta += float((parameter.detach().double() - before.double()).square().sum())
        original += float(before.double().square().sum())
        changed += int(not torch.equal(parameter.detach(), before))
    key = f"{seed}_{role}_{condition}"
    B.save_json(f"update_{key}.json", rows)
    np.savez_compressed(R / "artifacts" / f"gate_{key}.npz", entropy_nats=np.stack(entropies),
                        selected=np.stack(masks), agreement=np.stack(agrees))
    return {"candidate_updates": len(rows), "executed_updates": sum(x["executed"] for x in rows),
            "skipped_updates": sum(not x["executed"] for x in rows),
            "task_seconds": time.monotonic() - begin,
            "adaptation_update_seconds": sum(x["update_seconds"] for x in rows),
            "peak_allocated_bytes": peak, "streams": {k: v.hexdigest() for k, v in streams.items()},
            "relative_parameter_change": math.sqrt(delta / max(original, 1e-12)),
            "changed_parameter_tensors": changed, "trainable_parameter_count": sum(p.numel() for p in params),
            "trainable_parameter_names": names, "source_labels_used": False,
            "target_entropy_final_batch": rows[-1]["entropy_nats"],
            "retained_sample_fraction": float(sum(x["retained_samples"] for x in rows) / sum(x["batch"] for x in rows)),
            "agreement_fraction": float(sum(x["agreement_samples"] for x in rows) / sum(x["batch"] for x in rows)),
            "skipped_update_count": sum(not x["executed"] for x in rows),
            "target_backward_trace_exposures": sum(x["retained_samples"] for x in rows),
            "gmm_calls": 0, "pseudo_label_calls": 0,
            "tent_loss": "cosine_decay*(0.5*clean_entropy+0.5*masked_consistency)",
            "tent_steps_per_batch": 1, "batch_statistics": False,
            "bn_forward_batches": 2 * steps, "bn_num_batches_delta": {},
            "bn_momentum": None, "source_forward_trace_exposures": 0,
            "source_backward_trace_exposures": 0, "source_samples_used_for_adaptation": False,
            "target_forward_trace_exposures": 2 * sum(x["batch"] for x in rows)}


def freeze_inputs():
    hist_files = []
    for alias, relative in C["historical_references"].items():
        p = ROOT / relative; hist_files.append(p)
        hist_files.extend(p.parent / f"pred_{s}_{r}_{alias}.npz" for s in C["seeds"] for r in C["roles"])
    files = list(R.glob("*.py")) + [R / "PLAN.md", R / "SOURCES.md", R / "config.json", R / "TENT_README.md"]
    files += [ROOT / C["dataset_config"]] + [ROOT / p for p in C["prepared"].values()]
    files += [ROOT / C["base_checkpoint_run"] / "checkpoints" / f"wide_bn_{s}_best.pt" for s in C["seeds"]]
    files += [ROOT / C["comparison_run"] / "artifacts" / f"split_{r}.npz" for r in C["roles"]]
    files += hist_files + [ROOT / "runs/20261014T070454Z_tent_baseline_v3_83d48bf3/runner.py"]
    B.save_json("freeze.json", {str(p.relative_to(ROOT)): B.sha(p) for p in files})


def historical_rows():
    out = []
    for alias, relative in C["historical_references"].items():
        old = json.loads((ROOT / relative).read_text())
        picked = [r for r in old if r["condition"] == alias]
        assert len(picked) == 9
        for row in picked:
            x = {k: row[k] for k in ("seed", "role", "rows", "accuracy", "macro_f1", "negative_adaptation_rate", "recovery_rate", "changed_rate")}
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
    assert (R / "artifacts/launch.json").exists()
    with (R / "artifacts/runner_started.json").open("x") as marker:
        json.dump({"utc": time.time(), "run_id": R.name}, marker)
    preflight = json.loads((R / "artifacts/preflight.json").read_text())
    assert preflight["passed"] and preflight["config_sha"] == B.sha(R / "config.json")
    for name, digest in preflight["code_sha"].items():
        assert B.sha(R / name) == digest, name
    freeze_inputs(); B.save_json("hardware_start.json", {"snapshot": snapshot_gpu(), "gpu": torch.cuda.get_device_name(0)})
    B.registry("running", "Scale-Lock冻结输出头、固定BN统计量并启用5% Span-Invariance，9项目标任务已启动。")
    status_update("Scale-Lock冻结输出头、固定BN统计量并启用5% Span-Invariance，9项目标任务已启动。")
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
    scores = historical_rows(); B.save_json("scores.json", scores)
    B.event("preflight_verified", new_tasks=9, historical_rows=C["budget"]["historical_rows"], splits_match=True)
    for seed in C["seeds"]:
        for role in C["roles"]:
            target = B.load_prepared(C["prepared"][role]); s, q = splits[role]
            support = {k: target[k][s] for k in ("timestamps", "tam", "rows")}
            paired = None
            for condition in CONDS:
                model, _ = B.load_model(seed)
                cost = train_condition(model, support, condition, seed, role)
                reference = next(x for x in json.loads((ROOT / C["historical_references"]["no_pseudo_fixed"]).read_text()) if x["seed"] == seed and x["role"] == role and x["condition"] == "no_pseudo_fixed")
                assert reference["streams"]["target"] == cost["streams"]["target"] if "streams" in reference else True
                paired = cost["streams"] if paired is None else paired
                assert paired == cost["streams"]
                start = clock_start(); logits = query_predict(model, target["timestamps"][q], target["tam"][q]); query_seconds = clock_end(start)
                pred = logits.argmax(1)
                frozen = np.load(R / "artifacts" / f"pred_{seed}_{role}_frozen.npz")["predictions"]
                row = {"seed": seed, "role": role, "condition": condition, "origin": "new", "rows": len(q),
                       **B.cell_metrics(target["labels"][q], frozen, pred), **cost,
                       "query_inference_seconds": query_seconds, "query_latency_ms_per_trace": 1000 * query_seconds / len(q), **norm_diagnostics(model, logits)}
                np.savez_compressed(R / "artifacts" / f"pred_{seed}_{role}_{condition}.npz", rows=target["rows"][q], labels=target["labels"][q], predictions=pred, logits=logits)
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "seed": seed, "role": role, "condition": condition, "config_sha": B.sha(R / "config.json")}, R / "checkpoints" / f"{seed}_{role}_{condition}.pt")
                scores.append(row); B.save_json("scores.json", scores)
                B.save_json("progress.json", {"completed_tasks": sum(x["origin"] == "new" for x in scores), "total_tasks": 9, "seed": seed, "role": role, "condition": condition, "elapsed_seconds": time.time() - started})
                B.event("task_completed", seed=seed, role=role, condition=condition, completed=sum(x["origin"] == "new" for x in scores))
                del model; torch.cuda.empty_cache()
    assert sum(p.stat().st_size for p in R.rglob("*") if p.is_file()) <= C["budget"]["max_output_gib"] * 1024**3
    subprocess.run([sys.executable, str(R / "postrun_audit.py")], check=True)
    summary = report(scores); B.save_json("hardware_end.json", {"snapshot": snapshot_gpu()})
    B.registry("completed", summary); status_update(summary); B.event("completed", elapsed_seconds=time.time() - started)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        B.save_json("failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        B.registry("failed", f"Scale-Lock/Span-Invariance实验失败：{type(exc).__name__}: {exc}；产物保留。")
        status_update(f"执行失败：{type(exc).__name__}，产物已保留。")
        raise
