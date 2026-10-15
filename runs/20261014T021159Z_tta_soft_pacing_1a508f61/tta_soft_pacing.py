"""Soft pacing control for entropy-based test-time adaptation.

The controller is deliberately agnostic to the model and to the base TTA
loss.  It adds two terms to an already constructed loss:

* a source-parameter anchor, normalized per tensor so its scale does not grow
  with the number of parameters; and
* a marginal-prediction diversity term.

The caller still decides which parameters are trainable and whether the model
is in ``train`` mode.  In particular, ``train`` mode is needed if BN running
statistics are intended to adapt.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TTASoftPacingWrapper:
    """Add elastic source anchoring and marginal diversity to one TTA step.

    ``optimizer`` must contain every parameter with ``requires_grad=True``.
    This catches accidental updates outside the intended TTA subspace.  BN
    running statistics are buffers and are therefore intentionally not part of
    the anchor; their update is controlled by the caller's BN mode/momentum.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lambda_anchor_wide: float = 0.05,
        lambda_anchor_attn: float = 0.005,
        lambda_marginal: float = 0.5,
        *,
        classifier_dead_prefix: int | None = None,
        anchor_floor: float = 1e-3,
        bn_momentum: float | None = None,
        strict_optimizer: bool = True,
    ) -> None:
        if min(lambda_anchor_wide, lambda_anchor_attn, lambda_marginal) < 0:
            raise ValueError("soft-pacing weights must be non-negative")
        if anchor_floor <= 0:
            raise ValueError("anchor_floor must be positive")
        if bn_momentum is not None and not 0.0 <= bn_momentum <= 1.0:
            raise ValueError("bn_momentum must be in [0, 1]")

        self.model = model
        self.optimizer = optimizer
        self.lambda_anchor_wide = float(lambda_anchor_wide)
        self.lambda_anchor_attn = float(lambda_anchor_attn)
        self.lambda_marginal = float(lambda_marginal)
        self.classifier_dead_prefix = classifier_dead_prefix
        self.anchor_floor = float(anchor_floor)
        self.bn_momentum = bn_momentum

        if bn_momentum is not None:
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.momentum = float(bn_momentum)

        named = dict(model.named_parameters())
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        trainable = {
            name: parameter
            for name, parameter in named.items()
            if parameter.requires_grad
        }
        missing = sorted(name for name, parameter in trainable.items()
                        if id(parameter) not in optimizer_ids)
        if strict_optimizer and missing:
            raise ValueError(
                "requires_grad parameters absent from optimizer: "
                + ", ".join(missing)
            )
        self._parameters = {
            name: parameter
            for name, parameter in trainable.items()
            if id(parameter) in optimizer_ids
        }
        if not self._parameters:
            raise ValueError("optimizer has no trainable model parameters")

        # Keep the source copy detached and on the same device/dtype as the
        # parameter at construction time. It is never passed to the optimizer.
        self.source_params = {
            name: parameter.detach().clone()
            for name, parameter in self._parameters.items()
        }

    @staticmethod
    def _is_wide_parameter(name: str) -> bool:
        # The current wide+BN model names all generator convolutions/BNs under
        # generator.* and its input projection as time_projection.*.
        return name.startswith("generator.") or name.startswith("time_projection.")

    def compute_anchor_loss(self) -> Tensor:
        """Return a scale-stable, per-tensor elastic anchor penalty."""
        device = next(iter(self._parameters.values())).device
        dtype = next(iter(self._parameters.values())).dtype
        groups: dict[str, list[Tensor]] = {"wide": [], "attn": []}
        for name, parameter in self._parameters.items():
            reference = self.source_params[name].to(
                device=parameter.device, dtype=parameter.dtype
            )
            # Relative MSE makes lambda values comparable across convolutions,
            # LayerNorm, biases, and the classifier. The floor handles zero or
            # near-zero source tensors without producing infinities.
            scale = reference.detach().square().mean().sqrt().clamp_min(
                self.anchor_floor
            )
            normalized = (parameter - reference) / scale
            key = "wide" if self._is_wide_parameter(name) else "attn"
            groups[key].append(normalized.square().mean())

        loss = torch.zeros((), device=device, dtype=dtype)
        if groups["wide"]:
            loss = loss + self.lambda_anchor_wide * torch.stack(groups["wide"]).mean()
        if groups["attn"]:
            loss = loss + self.lambda_anchor_attn * torch.stack(groups["attn"]).mean()
        return loss

    @staticmethod
    def compute_marginal_diversity_loss(probs: Tensor) -> Tensor:
        """Return ``sum(p_bar log p_bar)`` for a batch of class probabilities."""
        if probs.ndim != 2 or probs.shape[0] == 0:
            raise ValueError("probs must have shape (batch, classes)")
        if not torch.isfinite(probs).all() or (probs < 0).any():
            raise ValueError("probs must be finite and non-negative")
        p_bar = probs.mean(dim=0)
        return (p_bar * p_bar.clamp_min(1e-8).log()).sum()

    def _mask_dead_classifier_columns(self) -> int:
        prefix = self.classifier_dead_prefix
        if prefix is None:
            return 0
        classifier = getattr(self.model, "classifier", None)
        weight = getattr(classifier, "weight", None)
        if weight is None or weight.grad is None:
            return 0
        if weight.ndim != 2 or not 0 <= prefix <= weight.shape[1]:
            raise ValueError(
                "classifier_dead_prefix must be in [0, classifier input width]"
            )
        with torch.no_grad():
            weight.grad[:, :prefix].zero_()
        return int(prefix)

    def step(
        self,
        base_tta_loss: Tensor,
        logits: Tensor,
        *,
        retain_graph: bool = False,
    ) -> dict[str, Any]:
        """Perform one update and return auditable scalar diagnostics.

        ``base_tta_loss`` should contain the source CE, target entropy, and
        feature MMD terms already used by the current baseline. If it already
        contains marginal diversity, omit that term before calling this method
        to avoid counting it twice.
        """
        if base_tta_loss.ndim != 0:
            raise ValueError("base_tta_loss must be scalar")
        if logits.ndim != 2 or logits.shape[0] == 0:
            raise ValueError("logits must have shape (batch, classes)")
        if not torch.isfinite(base_tta_loss):
            raise ValueError("base_tta_loss is not finite")

        probs = F.softmax(logits, dim=-1)
        loss_anchor = self.compute_anchor_loss()
        loss_marginal = self.compute_marginal_diversity_loss(probs)
        total_loss = (
            base_tta_loss
            + loss_anchor
            + self.lambda_marginal * loss_marginal
        )
        if not torch.isfinite(total_loss):
            raise ValueError("total TTA loss is not finite")

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward(retain_graph=retain_graph)
        masked = self._mask_dead_classifier_columns()
        max_gradient = 0.0
        for parameter in self._parameters.values():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError("non-finite TTA gradient")
                max_gradient = max(max_gradient, float(parameter.grad.detach().abs().max()))
        self.optimizer.step()
        return {
            "total_loss": float(total_loss.detach()),
            "base_tta_loss": float(base_tta_loss.detach()),
            "loss_anchor": float(loss_anchor.detach()),
            "loss_marginal": float(loss_marginal.detach()),
            "max_gradient": max_gradient,
            "masked_classifier_columns": masked,
        }

