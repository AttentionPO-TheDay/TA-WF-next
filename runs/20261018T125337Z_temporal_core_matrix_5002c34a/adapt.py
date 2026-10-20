"""Support-only 800-step adapters with explicit parameter and BN scopes."""
import hashlib
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from methods import entropy, forward_features, gmm_pseudo, mmd


def _loader(data, batch_size, seed, labeled):
    fields = [data["timestamps"], data["tam"]]
    if labeled:
        fields.append(data["labels"])
    fields.append(data["rows"])
    return DataLoader(TensorDataset(*fields), batch_size=batch_size, shuffle=True, drop_last=True,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def _next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def configure(model, condition):
    model.train()
    model.requires_grad_(False)
    if condition == "tent_bn":
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    elif condition == "tent_entropy_gate":
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()
            if isinstance(module, nn.LayerNorm) and (name.startswith("head.") or name == "final_norm"):
                module.requires_grad_(True)
    elif condition in ("proteus_default", "no_pseudo_fixed"):
        model.requires_grad_(True)
        if condition == "no_pseudo_fixed":
            model.classifier.requires_grad_(False)
    else:
        raise ValueError(condition)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    if condition == "tent_bn":
        assert all(".norm" in name and "generator" in name for name in trainable)
    if condition == "tent_entropy_gate":
        assert all(name.startswith("head.") or name.startswith("final_norm.") for name in trainable)
        assert not any(name.startswith("classifier.") for name in trainable)
    if condition == "no_pseudo_fixed":
        assert not any(name.startswith("classifier.") for name in trainable)
    return trainable


def adapt(model, source, support, condition, config, device, steps=800, require_change=True):
    assert set(support) == {"timestamps", "tam", "rows"}
    a = config["adaptation"]
    trainable = configure(model, condition)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    lr = a["proteus_lr"] if condition == "proteus_default" else a["no_pseudo_lr"] if condition == "no_pseudo_fixed" else a["tent_lr"]
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, betas=(0.9, 0.999))
    support_loader = _loader(support, a["batch_size"], a["support_loader_seed"], False)
    support_it = iter(support_loader)
    source_loader = _loader(source, a["source_batch_size"], a["source_loader_seed"], True) if condition in ("proteus_default", "no_pseudo_fixed") else None
    source_it = iter(source_loader) if source_loader is not None else None
    pseudo_loader = pseudo_it = None
    pseudo_history, dynamics = [], []
    stream = hashlib.sha256()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        if time.monotonic() - started > config["budget"]["max_seconds_per_cell"]:
            raise TimeoutError(f"{condition} exceeded cell budget")
        if condition == "proteus_default" and step % a["gmm_refresh_every_steps"] == 0:
            pseudo = gmm_pseudo(model, support, device, a["batch_size"], a["gmm_low_entropy_posterior_threshold"])
            pseudo_history.append({"step": step, "coverage": pseudo["coverage"], "means": pseudo["means"]})
            selected = torch.from_numpy(pseudo["selected"])
            pdata = {"timestamps": support["timestamps"][selected], "tam": support["tam"][selected],
                     "labels": torch.from_numpy(pseudo["labels"][pseudo["selected"]]), "rows": support["rows"][selected]}
            pseudo_loader = _loader(pdata, a["batch_size"], 2030 + step // a["gmm_refresh_every_steps"], True)
            pseudo_it = iter(pseudo_loader)
            model.train()
        target_batch, support_it = _next(support_it, support_loader)
        tx, tt = target_batch[0].to(device), target_batch[1].to(device)
        stream.update(target_batch[-1].numpy().astype("<i8").tobytes())
        torch.manual_seed(a["dropout_seed_base"] + step)
        optimizer.zero_grad(set_to_none=True)
        if source_loader is not None:
            source_batch, source_it = _next(source_it, source_loader)
            sx, st, sy = (value.to(device) for value in source_batch[:3])
        if condition == "tent_bn":
            logits = model(tx, tt)
            target_entropy = entropy(logits)
            loss = target_entropy.mean()
            kept = len(tx)
        elif condition == "tent_entropy_gate":
            logits = model(tx, tt)
            target_entropy = entropy(logits)
            keep = (target_entropy >= a["entropy_gate_low_nats"]) & (target_entropy <= a["entropy_gate_high_nats"])
            kept = int(keep.sum())
            loss = target_entropy[keep].mean() if kept else None
        else:
            source_logits, source_features = forward_features(model, sx, st)
            logits, target_features = forward_features(model, tx, tt)
            target_entropy = entropy(logits)
            marginal = logits.softmax(1).mean(0)
            loss = F.cross_entropy(source_logits, sy) + target_entropy.mean() + (marginal * torch.log(marginal + 1e-5)).sum() + mmd(source_features, target_features)
            kept = len(tx)
            if condition == "proteus_default":
                pseudo_batch, pseudo_it = _next(pseudo_it, pseudo_loader)
                px, pt, py = (value.to(device) for value in pseudo_batch[:3])
                pseudo_logits, _ = forward_features(model, px, pt)
                loss = loss + F.cross_entropy(pseudo_logits, py)
        if loss is not None:
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        dynamics.append({"step": step + 1, "mean_logit_l2": float(logits.detach().norm(dim=1).mean().cpu()),
                         "mean_entropy_nats": float(target_entropy.detach().mean().cpu()),
                         "entropy_histogram_20_bins_0_to_ln102": torch.histc(target_entropy.detach().cpu(), bins=20, min=0, max=float(np.log(102))).int().tolist(),
                         "retained": kept, "loss": None if loss is None else float(loss.detach().cpu())})
    after = model.state_dict()
    changed = 0
    for name, parameter in model.named_parameters():
        old = before[name]
        if name in trainable:
            changed += int(not torch.equal(parameter.detach().cpu(), old))
        else:
            assert torch.equal(parameter.detach().cpu(), old), f"locked parameter changed: {name}"
    if require_change:
        assert changed > 0
    if condition == "tent_entropy_gate":
        for name, value in after.items():
            if name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked"):
                assert torch.equal(value.cpu(), before[name]), f"source BN buffer changed: {name}"
    return {"candidate_updates": steps, "executed_updates": sum(row["retained"] > 0 for row in dynamics),
            "trainable_names": trainable, "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "changed_parameter_tensors": changed, "target_row_stream_sha256": stream.hexdigest(),
            "pseudo_refreshes": len(pseudo_history), "pseudo_history": pseudo_history,
            "dynamics": dynamics, "seconds": time.monotonic() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())}
