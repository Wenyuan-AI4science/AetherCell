"""Training objective used by the released AetherCell perturbation models.

The specificity term compares a predicted latent displacement with the true
displacement and with a cell-context background displacement.  References are
computed from the training split only.  By default, the current observation is
removed from its cell mean (leave-one-out) to avoid target leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cell_key(value: object) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return str(value)


def topk_directional_loss(
    delta_pred: torch.Tensor,
    delta_true: torch.Tensor,
    k: int = 200,
    eps: float = 1e-8,
) -> torch.Tensor:
    """One minus centered cosine on the strongest ground-truth changes."""
    if delta_pred.shape != delta_true.shape or delta_true.ndim != 2:
        raise ValueError("delta_pred and delta_true must be equally shaped [batch, genes]")
    k = min(max(int(k), 1), delta_true.shape[1])
    idx = delta_true.abs().topk(k, dim=1).indices
    true_k = torch.gather(delta_true, 1, idx)
    pred_k = torch.gather(delta_pred, 1, idx)
    true_k = true_k - true_k.mean(dim=1, keepdim=True)
    pred_k = pred_k - pred_k.mean(dim=1, keepdim=True)
    cosine = F.cosine_similarity(true_k, pred_k, dim=1, eps=eps)
    return (1.0 - cosine).mean()


def delta_weighted_mse(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    x_control: torch.Tensor,
    quantile: float = 0.8,
    power: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MSE weighted by the within-profile ground-truth response magnitude."""
    if not (x_pred.shape == x_true.shape == x_control.shape) or x_true.ndim != 2:
        raise ValueError("prediction, truth and control must be equally shaped [batch, genes]")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    delta_true = x_true - x_control
    delta_pred = x_pred - x_control
    signal = delta_true.abs()
    threshold = torch.quantile(signal, quantile, dim=1, keepdim=True)
    weights = (signal / (threshold + eps)).pow(power)
    weights = weights / (weights.mean(dim=1, keepdim=True) + eps)
    return (weights * (delta_pred - delta_true).pow(2)).mean()


@dataclass(frozen=True)
class SpecificityReference:
    """Training-only sufficient statistics for cell-context latent means."""

    sums: Mapping[str, torch.Tensor]
    counts: Mapping[str, int]
    global_mean: torch.Tensor

    @classmethod
    def from_batches(
        cls,
        delta_batches: Iterable[tuple[torch.Tensor, Sequence[object]]],
    ) -> "SpecificityReference":
        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        global_sum: torch.Tensor | None = None
        total = 0
        for deltas, cell_ids in delta_batches:
            values = deltas.detach().float().cpu()
            if values.ndim != 2 or len(cell_ids) != values.shape[0]:
                raise ValueError("each batch must provide [batch, latent] values and matching cell IDs")
            for value, cell_id in zip(values, cell_ids):
                key = _cell_key(cell_id)
                sums[key] = sums.get(key, torch.zeros_like(value)) + value
                counts[key] = counts.get(key, 0) + 1
                global_sum = value.clone() if global_sum is None else global_sum + value
                total += 1
        if global_sum is None or total == 0:
            raise ValueError("cannot build specificity references from an empty training set")
        return cls(sums=sums, counts=counts, global_mean=global_sum / total)

    def background(
        self,
        delta_true: torch.Tensor,
        cell_ids: Sequence[object],
        leave_one_out: bool = True,
    ) -> torch.Tensor:
        """Return a context mean per sample, with a global fallback."""
        if delta_true.ndim != 2 or len(cell_ids) != delta_true.shape[0]:
            raise ValueError("delta_true must be [batch, latent] with one cell ID per row")
        rows = []
        for true_value, cell_id in zip(delta_true, cell_ids):
            key = _cell_key(cell_id)
            count = int(self.counts.get(key, 0))
            if key not in self.sums:
                mean = self.global_mean
            elif leave_one_out and count > 1:
                mean = (self.sums[key] - true_value.detach().cpu()) / (count - 1)
            elif leave_one_out and count <= 1:
                mean = self.global_mean
            else:
                mean = self.sums[key] / max(count, 1)
            rows.append(mean)
        return torch.stack(rows).to(device=delta_true.device, dtype=delta_true.dtype).detach()


class SpecificityGainLoss(nn.Module):
    """Hinge loss requiring prediction to beat a context-only baseline."""

    def __init__(self, reference: SpecificityReference, margin: float = 0.1, leave_one_out: bool = True):
        super().__init__()
        self.reference = reference
        self.margin = float(margin)
        self.leave_one_out = bool(leave_one_out)

    def forward(
        self,
        delta_pred: torch.Tensor,
        delta_true: torch.Tensor,
        cell_ids: Sequence[object],
    ) -> torch.Tensor:
        background = self.reference.background(delta_true, cell_ids, self.leave_one_out)
        prediction_error = torch.linalg.vector_norm(delta_pred - delta_true, dim=1)
        background_error = torch.linalg.vector_norm(background - delta_true, dim=1)
        return F.relu(prediction_error - background_error + self.margin).mean()


@dataclass(frozen=True)
class LossWeights:
    reconstruction: float = 0.5
    directional: float = 2.0
    weighted_mse: float = 0.3
    latent_alignment: float = 0.2
    specificity: float = 0.2


def _core_loss_parts(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    control: torch.Tensor,
    delta_z_pred: torch.Tensor,
    delta_z_true: torch.Tensor,
    top_k: int,
) -> dict[str, torch.Tensor]:
    delta_true = target - control
    delta_pred = prediction - control
    return {
        "reconstruction": F.l1_loss(prediction, target),
        "directional": topk_directional_loss(delta_pred, delta_true, top_k),
        "weighted_mse": delta_weighted_mse(prediction, target, control),
        "latent_alignment": F.mse_loss(delta_z_pred, delta_z_true),
    }


def _weighted_core_total(parts: Mapping[str, torch.Tensor], weights: LossWeights) -> torch.Tensor:
    return (
        weights.reconstruction * parts["reconstruction"]
        + weights.directional * parts["directional"]
        + weights.weighted_mse * parts["weighted_mse"]
        + weights.latent_alignment * parts["latent_alignment"]
    )


class AetherCellLoss(nn.Module):
    """Complete reconstruction + direction + magnitude + alignment + specificity objective."""

    def __init__(
        self,
        reference: SpecificityReference,
        weights: LossWeights | None = None,
        top_k: int = 200,
        specificity_margin: float = 0.1,
        leave_one_out: bool = True,
    ):
        super().__init__()
        self.weights = weights or LossWeights()
        self.top_k = int(top_k)
        self.specificity = SpecificityGainLoss(reference, specificity_margin, leave_one_out)

    def forward(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        control: torch.Tensor,
        delta_z_pred: torch.Tensor,
        delta_z_true: torch.Tensor,
        cell_ids: Sequence[object],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts = _core_loss_parts(
            prediction=prediction,
            target=target,
            control=control,
            delta_z_pred=delta_z_pred,
            delta_z_true=delta_z_true,
            top_k=self.top_k,
        )
        parts["specificity"] = self.specificity(delta_z_pred, delta_z_true, cell_ids)
        w = self.weights
        total = _weighted_core_total(parts, w) + w.specificity * parts["specificity"]
        return total, parts


class AetherCellValidationLoss(nn.Module):
    """Validation objective that deliberately excludes specificity.

    Specificity is a training regularizer. Validation reports the remaining
    reconstruction, direction, magnitude, and latent-alignment terms without
    constructing or querying a specificity reference.
    """

    def __init__(self, weights: LossWeights | None = None, top_k: int = 200):
        super().__init__()
        self.weights = weights or LossWeights()
        self.top_k = int(top_k)

    def forward(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        control: torch.Tensor,
        delta_z_pred: torch.Tensor,
        delta_z_true: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts = _core_loss_parts(
            prediction=prediction,
            target=target,
            control=control,
            delta_z_pred=delta_z_pred,
            delta_z_true=delta_z_true,
            top_k=self.top_k,
        )
        return _weighted_core_total(parts, self.weights), parts
