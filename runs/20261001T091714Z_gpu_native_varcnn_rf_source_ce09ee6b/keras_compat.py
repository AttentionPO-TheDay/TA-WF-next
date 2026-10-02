"""Minimal Keras 2.0.8 training semantics for the native Var-CNN port.

Source equations/order audited against keras-team/keras tag 2.0.8;
see runs/20261001T091714Z_gpu_native_varcnn_rf_source_ce09ee6b/KERAS_RECIPE.md.
This is a formula/behavior port, not a cross-framework numeric parity claim.
"""
import math

import torch


class KerasAdam(torch.optim.Optimizer):
    """Adam with epsilon added to the *uncorrected* second moment root."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0 or eps < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid Adam parameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, iterations=0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["iterations"] += 1
            t = group["iterations"]
            b1, b2 = group["betas"]
            rate = group["lr"] * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("KerasAdam does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-rate)
        return loss


class KerasCallbacks:
    """Var-CNN's ordered plateau, early-stop, strict-best callbacks."""

    def __init__(self):
        self.plateau_best = -math.inf
        self.early_best = -math.inf
        self.checkpoint_best = -math.inf
        self.plateau_wait = 0
        self.early_wait = 0
        self.stopped = False

    def state_dict(self):
        return vars(self).copy()

    def load_state_dict(self, state):
        if set(state) != set(vars(self)):
            raise ValueError("Callback state fields differ")
        vars(self).update(state)

    def step(self, acc, optimizer):
        acc = float(acc)
        if not math.isfinite(acc) or not 0 <= acc <= 1:
            raise ValueError("Validation accuracy must be finite and in [0, 1]")
        old_rates = [float(g["lr"]) for g in optimizer.param_groups]
        if acc > self.plateau_best + 1e-4:
            self.plateau_best = acc
            self.plateau_wait = 0
        else:
            if self.plateau_wait >= 5:
                reduced = False
                for group in optimizer.param_groups:
                    if group["lr"] > 1e-5 + 1e-9:
                        group["lr"] = max(group["lr"] * math.sqrt(0.1), 1e-5)
                        reduced = True
                if reduced:
                    self.plateau_wait = 0
            self.plateau_wait += 1
        if acc > self.early_best:
            self.early_best = acc
            self.early_wait = 0
        else:
            if self.early_wait >= 10:
                self.stopped = True
            self.early_wait += 1
        improved = acc > self.checkpoint_best
        if improved:
            self.checkpoint_best = acc
        new_rates = [float(g["lr"]) for g in optimizer.param_groups]
        return dict(stop=self.stopped, improved=improved,
                    lr_before=old_rates[0], lr_after=new_rates[0],
                    reduced_lr=old_rates != new_rates,
                    plateau_wait=self.plateau_wait, early_wait=self.early_wait)
