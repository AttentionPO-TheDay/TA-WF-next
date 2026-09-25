"""Small generator-token Transformer prototype.

This module is intentionally a research prototype, not a claim of drift
invariance.  It consumes the explicit views produced by ``traffic_views``/
``batch_views`` and keeps packet, run and window tokens distinguishable by a
learned type embedding.  No labels are needed for ``masked_reconstruction``;
labels are used only by the optional classifier and TTA helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .traffic_views import TrafficViews


@dataclass(frozen=True)
class GeneratorTokenBatch:
    """Padded generator views and their valid-token masks.

    Values have shapes ``[B, L, C]`` and masks have shape ``[B, L]``.  A true
    mask entry denotes an observed token; padding must be false.  The packet,
    run and window feature widths are deliberately explicit to prevent a
    silent coercion of one view into another.
    """

    packet: Tensor
    packet_mask: Tensor
    runs: Tensor
    runs_mask: Tensor
    windows: Tensor
    windows_mask: Tensor
    spans: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        for name, value, mask in (
            ("packet", self.packet, self.packet_mask),
            ("runs", self.runs, self.runs_mask),
            ("windows", self.windows, self.windows_mask),
        ):
            if value.ndim != 3 or mask.ndim != 2 or value.shape[:2] != mask.shape:
                raise ValueError(f"{name} must be [B,L,C] with [B,L] mask")
            if mask.dtype != torch.bool:
                raise TypeError(f"{name}_mask must be bool")
        if not (self.packet.shape[0] == self.runs.shape[0] == self.windows.shape[0]):
            raise ValueError("all views must have the same batch size")
        for name in ("packet", "runs", "windows"):
            if name not in self.spans or self.spans[name].shape != (*getattr(self, name).shape[:2], 2):
                raise ValueError(f"{name} spans must be [B,L,2]")

    def to(self, device: torch.device | str) -> GeneratorTokenBatch:
        return GeneratorTokenBatch(
            *(getattr(self, name).to(device) for name in (
                "packet", "packet_mask", "runs", "runs_mask", "windows", "windows_mask"
            )),
            {name: span.to(device) for name, span in self.spans.items()},
        )


def batch_from_views(
    views: Sequence[TrafficViews], *, packet_patch: int = 50, max_runs: int = 128,
) -> GeneratorTokenBatch:
    """Convert the project's fixed generator output to typed, bounded tokens.

    Run count is log-scaled.  ``spans`` preserve the source-packet interval of
    each token, allowing all views overlapping a masked raw interval to be
    hidden together.  No labels, times, sizes, URLs or domain IDs are used.
    """
    if not views or packet_patch <= 0 or max_runs <= 0:
        raise ValueError("nonempty views and positive token budgets required")
    budget = views[0].runs.budget
    window_sizes = tuple(width for width, _ in views[0].direction_windows)
    if any(v.runs.budget != budget or tuple(w for w, _ in v.direction_windows) != window_sizes for v in views):
        raise ValueError("all views in a batch need identical observation/window budgets")
    fixed_lengths = {
        "packet": math.ceil(budget / packet_patch),
        "runs": max_runs,
        "windows": sum(math.ceil(budget / width) for width in window_sizes),
    }
    rows: dict[str, list[list[list[float]]]] = {n: [] for n in ("packet", "runs", "windows")}
    spans: dict[str, list[list[tuple[int, int]]]] = {n: [] for n in rows}
    for view in views:
        directions = view.packet_direction
        packet_values: list[list[float]] = []
        packet_spans: list[tuple[int, int]] = []
        for start in range(0, len(directions), packet_patch):
            part = directions[start:start + packet_patch]
            packet_values.append([
                sum(part) / len(part),
                sum(a != b for a, b in zip(part, part[1:])) / max(len(part) - 1, 1),
            ])
            packet_spans.append((start, start + len(part)))
        rows["packet"].append(packet_values)
        spans["packet"].append(packet_spans)

        run_values: list[list[float]] = []
        run_spans: list[tuple[int, int]] = []
        offset = 0
        for run in view.runs.runs[:max_runs]:
            run_values.append([
                float(run.direction), math.log1p(run.count),
                float(run.touches_left_boundary), float(run.touches_right_boundary),
            ])
            run_spans.append((offset, offset + run.count))
            offset += run.count
        rows["runs"].append(run_values)
        spans["runs"].append(run_spans)

        window_values: list[list[float]] = []
        window_spans: list[tuple[int, int]] = []
        for width, windows in view.direction_windows:
            for window in windows:
                window_values.append([
                    window.positive_fraction, window.transition_fraction,
                    window.observed_count / width, float(window.partial),
                ])
                window_spans.append((window.start, window.start + window.observed_count))
        rows["windows"].append(window_values)
        spans["windows"].append(window_spans)

    tensors: dict[str, Tensor] = {}
    masks: dict[str, Tensor] = {}
    span_tensors: dict[str, Tensor] = {}
    for name, width in (("packet", 2), ("runs", 4), ("windows", 4)):
        length = fixed_lengths[name]
        value = torch.zeros((len(views), length, width), dtype=torch.float32)
        mask = torch.zeros((len(views), length), dtype=torch.bool)
        location = torch.zeros((len(views), length, 2), dtype=torch.long)
        for i, row in enumerate(rows[name]):
            n = len(row)
            if n > length:
                raise ValueError(f"{name} exceeds fixed token budget")
            if n:
                value[i, :n] = torch.tensor(row, dtype=torch.float32)
                mask[i, :n] = True
                location[i, :n] = torch.tensor(spans[name][i], dtype=torch.long)
        tensors[name], masks[name], span_tensors[name] = value, mask, location
    return GeneratorTokenBatch(
        tensors["packet"], masks["packet"], tensors["runs"], masks["runs"],
        tensors["windows"], masks["windows"], span_tensors,
    )


def mask_observation_spans(
    batch: GeneratorTokenBatch, starts: Tensor, lengths: Tensor,
) -> tuple[GeneratorTokenBatch, dict[str, Tensor]]:
    """Hide every token whose raw-packet support intersects an input span.

    This prevents a masked packet patch being trivially reconstructed from an
    unmasked overlapping run or window token.
    """
    b = batch.packet.shape[0]
    if starts.shape != (b,) or lengths.shape != (b,) or (starts < 0).any() or (lengths <= 0).any():
        raise ValueError("starts/lengths must be positive per-sample vectors")
    selected: dict[str, Tensor] = {}
    values: dict[str, Tensor] = {}
    for name in ("packet", "runs", "windows"):
        bounds = batch.spans[name]
        overlap = (bounds[..., 0] < (starts + lengths)[:, None]) & (bounds[..., 1] > starts[:, None])
        selected[name] = overlap & getattr(batch, f"{name}_mask")
        values[name] = getattr(batch, name).masked_fill(selected[name][..., None], 0.0)
    masked_batch = GeneratorTokenBatch(
        values["packet"], batch.packet_mask, values["runs"], batch.runs_mask,
        values["windows"], batch.windows_mask, batch.spans,
    )
    return masked_batch, selected


class ResidualAdapter(nn.Module):
    """Bottleneck residual adapter, initially an exact identity mapping."""

    def __init__(self, d_model: int, bottleneck: int = 32) -> None:
        super().__init__()
        if d_model <= 0 or bottleneck <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.down = nn.Linear(d_model, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(self.activation(self.down(x)))


class GeneratorTokenTransformer(nn.Module):
    """Typed packet/run/window Transformer with classifier and reconstruction heads."""

    VIEW_ORDER = ("packet", "runs", "windows")

    def __init__(
        self,
        *,
        packet_dim: int = 2,
        run_dim: int = 4,
        window_dim: int = 4,
        d_model: int = 96,
        nhead: int = 4,
        layers: int = 2,
        dim_feedforward: int = 192,
        max_tokens: int = 512,
        num_classes: int = 102,
        adapter_bottleneck: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if max_tokens < 2:
            raise ValueError("max_tokens must include CLS and at least one token")
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.projections = nn.ModuleDict({
            "packet": nn.Linear(packet_dim, d_model),
            "runs": nn.Linear(run_dim, d_model),
            "windows": nn.Linear(window_dim, d_model),
        })
        self.reconstruction = nn.ModuleDict({
            "packet": nn.Linear(d_model, packet_dim),
            "runs": nn.Linear(d_model, run_dim),
            "windows": nn.Linear(d_model, window_dim),
        })
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        self.type_embedding = nn.Embedding(4, d_model)  # CLS, packet, runs, windows
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.adapter = ResidualAdapter(d_model, adapter_bottleneck)
        self.classifier = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def _tokens(self, batch: GeneratorTokenBatch) -> tuple[Tensor, Tensor, dict[str, slice]]:
        values = []
        masks = []
        types = []
        slices: dict[str, slice] = {}
        offset = 1
        for type_id, name in enumerate(self.VIEW_ORDER, start=1):
            value = getattr(batch, name)
            mask = getattr(batch, f"{name}_mask")
            if value.shape[-1] != self.projections[name].in_features:
                raise ValueError(f"{name} feature width mismatch")
            n = value.shape[1]
            projected = self.projections[name](value)
            values.append(projected)
            masks.append(mask)
            types.append((type_id, n))
            slices[name] = slice(offset, offset + n)
            offset += n
        all_values = torch.cat(values, dim=1)
        all_mask = torch.cat(masks, dim=1)
        if all_values.shape[1] + 1 > self.max_tokens:
            raise ValueError("token sequence exceeds max_tokens")
        cls = self.cls.expand(all_values.shape[0], -1, -1)
        all_values = torch.cat((cls, all_values), dim=1)
        all_mask = torch.cat((torch.ones((len(all_values), 1), dtype=torch.bool, device=all_values.device), all_mask), dim=1)
        type_ids = torch.cat([
            torch.full((n,), type_id, dtype=torch.long, device=all_values.device)
            for type_id, n in types
        ])
        type_ids = torch.cat((torch.zeros(1, dtype=torch.long, device=all_values.device), type_ids))
        all_values = all_values + self.position[:, :all_values.shape[1]] + self.type_embedding(type_ids)[None, :, :]
        return all_values, all_mask, slices

    def encode(self, batch: GeneratorTokenBatch) -> tuple[Tensor, dict[str, slice]]:
        tokens, valid, slices = self._tokens(batch)
        hidden = self.encoder(tokens, src_key_padding_mask=~valid)
        hidden = self.adapter(self.norm(hidden))
        return hidden, slices

    def forward(self, batch: GeneratorTokenBatch) -> dict[str, Tensor | dict[str, slice]]:
        hidden, slices = self.encode(batch)
        pooled = hidden[:, 0]
        return {"logits": self.classifier(pooled), "pooled": pooled, "hidden": hidden, "slices": slices}

    def masked_reconstruction_loss(
        self,
        output: Mapping[str, Tensor | dict[str, slice]],
        batch: GeneratorTokenBatch,
        masked: Mapping[str, Tensor],
    ) -> Tensor:
        """MSE only at selected valid positions; no category labels are used."""
        hidden = output["hidden"]
        slices = output["slices"]
        if not isinstance(hidden, Tensor) or not isinstance(slices, dict):
            raise TypeError("output must come from forward")
        losses = []
        for name in self.VIEW_ORDER:
            positions = masked[name]
            if positions.dtype != torch.bool or positions.shape != getattr(batch, f"{name}_mask").shape:
                raise ValueError(f"invalid masked positions for {name}")
            positions = positions & getattr(batch, f"{name}_mask")
            if positions.any():
                pred = self.reconstruction[name](hidden[:, slices[name]])
                target = getattr(batch, name)
                losses.append(F.mse_loss(pred[positions], target[positions]))
        if not losses:
            return hidden.sum() * 0.0
        return torch.stack(losses).mean()


def freeze_except_adapter(model: nn.Module) -> None:
    """Freeze the backbone/classifier for a conservative TTA update."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.adapter.parameters():
        parameter.requires_grad = True


def confident_teacher_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    threshold: float = 0.8,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """KL to a frozen teacher on confident samples; returns loss and selection mask."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("teacher and student logits must have shape [B,C]")
    if not 0.0 < threshold <= 1.0 or temperature <= 0.0:
        raise ValueError("invalid threshold or temperature")
    with torch.no_grad():
        teacher_prob = (teacher_logits / temperature).softmax(dim=1)
        confidence = teacher_prob.max(dim=1).values
        selected = confidence >= threshold
    if not selected.any():
        return student_logits.sum() * 0.0, selected
    log_prob = (student_logits[selected] / temperature).log_softmax(dim=1)
    loss = F.kl_div(log_prob, teacher_prob[selected], reduction="batchmean") * temperature**2
    return loss, selected


def view_consistency_loss(first: Tensor, second: Tensor) -> Tensor:
    """Cosine consistency for two views of the same unlabeled trace."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("consistency inputs must have equal shape [B,D]")
    return (1.0 - F.cosine_similarity(first, second, dim=1)).mean()


def pretrain_step(
    model: GeneratorTokenTransformer,
    batch: GeneratorTokenBatch,
    optimizer: torch.optim.Optimizer,
    *,
    span_length: int = 50,
) -> dict[str, float | int]:
    """One label-free step with the same raw span hidden in every view."""
    if span_length <= 0:
        raise ValueError("span_length must be positive")
    lengths = batch.spans["packet"][..., 1].max(dim=1).values
    if (lengths <= 0).any():
        raise ValueError("each trace needs at least one packet")
    span = torch.minimum(lengths, torch.full_like(lengths, span_length))
    starts = (torch.rand(len(lengths), device=lengths.device) * (lengths - span + 1)).long()
    masked_batch, selected = mask_observation_spans(batch, starts, span)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(masked_batch)
    loss = model.masked_reconstruction_loss(output, batch, selected)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite pretraining loss")
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach()), "masked_tokens": sum(int(x.sum()) for x in selected.values())}


def finetune_step(
    model: GeneratorTokenTransformer,
    batch: GeneratorTokenBatch,
    labels: Tensor,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """One supervised step; labels must be explicitly supplied by the caller."""
    if labels.ndim != 1 or labels.shape[0] != batch.packet.shape[0]:
        raise ValueError("labels must have shape [B]")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)["logits"]
    if not isinstance(logits, Tensor):
        raise TypeError("model output missing logits")
    loss = F.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite finetuning loss")
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach()), "batch_accuracy": float((logits.argmax(1) == labels).float().mean())}


def tta_adapter_step(
    student: GeneratorTokenTransformer,
    teacher: GeneratorTokenTransformer,
    clean: GeneratorTokenBatch,
    perturbed: GeneratorTokenBatch,
    optimizer: torch.optim.Optimizer,
    *,
    confidence_threshold: float = 0.8,
    consistency_weight: float = 0.1,
) -> dict[str, float | int]:
    """One unlabeled, adapter-only step; the caller controls data permissions.

    ``clean`` and ``perturbed`` must represent the same traces.  This helper
    never accepts target labels or uses an evaluation metric.  It is *not* an
    authorization to adapt on reserved query data.
    """
    if clean.packet.shape[0] != perturbed.packet.shape[0] or consistency_weight < 0:
        raise ValueError("TTA views must share batch size and weight must be nonnegative")
    if student is teacher:
        raise ValueError("teacher and student must be separate model instances")
    if any(param.requires_grad for name, param in student.named_parameters() if not name.startswith("adapter.")):
        raise ValueError("freeze non-adapter student parameters before TTA")
    if any(param.requires_grad for param in teacher.parameters()):
        raise ValueError("teacher must be frozen")
    teacher.eval()
    student.eval()  # deterministic backbone; gradients still flow to adapter
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        teacher_logits = teacher(clean)["logits"]
    student_clean = student(clean)
    student_perturbed = student(perturbed)
    kl, selected = confident_teacher_kl(
        student_perturbed["logits"], teacher_logits, threshold=confidence_threshold,
    )
    consistency = view_consistency_loss(student_clean["pooled"], student_perturbed["pooled"])
    loss = kl + consistency_weight * consistency
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite TTA loss")
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach()), "teacher_selected": int(selected.sum()),
            "consistency": float(consistency.detach())}
