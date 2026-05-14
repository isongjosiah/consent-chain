"""Reconstruction-error threshold MIA for autoencoders (the ``label_only``-
equivalent in the AE setting).

For a member sample the AE typically reconstructs with *lower* error
(model overfit toward the training distribution). For a non-member the
reconstruction error is higher. A simple threshold attack therefore
suffices:

    classify x as member iff recon_error(x) < threshold

The threshold is calibrated on the auxiliary loader so that *exactly*
``fpr_target`` of auxiliary rows — which are non-members by construction
— would be classified as members. The reported score is the fraction
of the **member** loader rows whose reconstruction error sits below
that calibrated threshold.

This is the AE analogue of ``fl.mia.label_only.LabelOnlyMIA``. The class
keeps the same on-chain ``label_only`` name so
``fl.attestation.generate_ua`` is dataset-agnostic: one MIAResult
slot, swappable implementation depending on whether the target is a
classifier or an autoencoder.

Reference: Chen et al., 2020, "GAN-Leaks: A Taxonomy of Membership
Inference Attacks against Generative Models" — generalised to AE.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fl.mia.base import BaseMIA, MIARequest, MIAResult


def _unpack(batch: Any) -> torch.Tensor:
    """Loader may yield ``(X,)`` for AE setting or ``(X, y)`` for legacy."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _recon_errors(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    """Per-sample MSE reconstruction error."""
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for batch in loader:
            X = _unpack(batch).to(device, non_blocking=False)
            X_hat = model(X)
            per_sample = ((X_hat - X) ** 2).mean(dim=1)
            out.extend(per_sample.cpu().tolist())
    return np.asarray(out, dtype=np.float64)


class ReconThresholdMIA(BaseMIA):
    """Recon-error threshold attack. Shares the ``label_only`` on-chain slot."""

    name = "label_only"

    def attack(self, request: MIARequest) -> MIAResult:
        cfg = request.method_config
        fpr_target = float(cfg.get("fpr_target", 0.05))

        t0 = time.time()
        # Auxiliary: known non-members → use to calibrate the threshold
        # so exactly fpr_target of them sit *below* the threshold (the
        # would-be false positives).
        aux_errors = _recon_errors(
            request.target_model, request.auxiliary_loader, request.device
        )
        if aux_errors.size == 0:
            raise ValueError("auxiliary loader is empty; cannot calibrate threshold")
        # "Member" if recon_error < threshold, so the threshold is the
        # fpr_target-quantile of auxiliary errors.
        threshold = float(np.quantile(aux_errors, fpr_target))

        # Members.
        mem_errors = _recon_errors(
            request.target_model, request.member_loader, request.device
        )
        if mem_errors.size == 0:
            raise ValueError("member loader is empty; nothing to attack")
        score = float((mem_errors < threshold).mean())

        details: dict[str, Any] = {
            "task": "autoencoder",
            "fpr_target": fpr_target,
            "threshold": threshold,
            "member_set_size": int(mem_errors.size),
            "aux_set_size": int(aux_errors.size),
            "mean_member_recon": float(mem_errors.mean()),
            "mean_aux_recon": float(aux_errors.mean()),
            "member_recon_below_threshold": int((mem_errors < threshold).sum()),
        }
        return self._wrap(score, t0, details)
