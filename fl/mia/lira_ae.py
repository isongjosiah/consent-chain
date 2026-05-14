"""LiRA-style likelihood-ratio MIA for autoencoders.

AE analogue of ``fl.mia.lira.LiRAMIA``. Same Carlini-2022 structure
(IN/OUT reference models → per-example Gaussian fits → log-likelihood
ratio → median-α threshold) with the membership-signal feature swapped
from "logit of true-class probability" to "log reconstruction error":

  - ``log(eps + recon_error)`` is approximately Gaussian-distributed
    across reference models for a given sample (under the standard
    "models trained on similar data converge to similar
    reconstructions" assumption)
  - smaller log-recon-error under the target than under OUT references
    → likely member

Same on-chain ``lira`` name so the UA generation slot is identical for
classifier and AE targets.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fl.mia.base import BaseMIA, MIARequest, MIAResult


def _unpack(batch: Any) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _stack_loader(loader: DataLoader) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for batch in loader:
        out.append(_unpack(batch))
    return torch.cat(out, dim=0)


def _train_quick_ae(
    model: nn.Module, X: torch.Tensor, *,
    device: torch.device, epochs: int, batch_size: int, lr: float,
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            xb = X[idx].to(device)
            loss = ((model(xb) - xb) ** 2).mean()
            loss.backward()
            opt.step()


def _log_recon_errors(
    model: nn.Module, X: torch.Tensor, *,
    device: torch.device, eps: float, batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size].to(device)
            per_sample = ((model(xb) - xb) ** 2).mean(dim=1)
            log_err = torch.log(per_sample + eps)
            out.extend(log_err.cpu().tolist())
    return np.asarray(out, dtype=np.float64)


def _log_normal_pdf(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd_safe = np.maximum(sd, 1e-6)
    return -0.5 * ((x - mu) / sd_safe) ** 2 - np.log(sd_safe * math.sqrt(2 * math.pi))


class LiRAAEMIA(BaseMIA):
    name = "lira"

    def attack(self, request: MIARequest) -> MIAResult:  # noqa: C901  acceptable
        cfg = request.method_config
        n_in = int(cfg.get("num_in_ref", 4))
        n_out = int(cfg.get("num_out_ref", 4))
        eps = float(cfg.get("recon_eps", 1e-7))
        alpha = float(cfg.get("classification_alpha", 0.5))
        epochs = int(cfg.get("ref_epochs", 5))
        batch_size = int(cfg.get("ref_batch_size", 256))
        lr = float(cfg.get("ref_learning_rate", 1e-3))
        subset_fraction = float(cfg.get("ref_subset_fraction", 0.7))

        t0 = time.time()
        rng = np.random.default_rng(request.seed + 31)

        X_mem = _stack_loader(request.member_loader)
        X_aux = _stack_loader(request.auxiliary_loader)
        if X_mem.shape[0] == 0:
            raise ValueError("member loader is empty; nothing to attack")
        if X_aux.shape[0] == 0:
            raise ValueError("auxiliary loader is empty; LiRA-AE needs OUT pool data")

        # IN pool = auxiliary ∪ members; OUT pool = auxiliary only.
        X_in_pool = torch.cat([X_aux, X_mem], dim=0)

        def _train_ref(pool: torch.Tensor) -> nn.Module:
            n = pool.shape[0]
            k = max(4, int(round(n * subset_fraction)))
            idx = rng.choice(n, size=min(k, n), replace=False)
            m = request.model_factory()
            _train_quick_ae(
                m, pool[idx],
                device=request.device, epochs=epochs, batch_size=batch_size, lr=lr,
            )
            return m

        in_logs = np.stack(
            [
                _log_recon_errors(
                    _train_ref(X_in_pool), X_mem, device=request.device, eps=eps
                )
                for _ in range(n_in)
            ],
            axis=0,
        )
        out_logs = np.stack(
            [
                _log_recon_errors(
                    _train_ref(X_aux), X_mem, device=request.device, eps=eps
                )
                for _ in range(n_out)
            ],
            axis=0,
        )

        mu_in = in_logs.mean(axis=0)
        sd_in = in_logs.std(axis=0, ddof=0)
        mu_out = out_logs.mean(axis=0)
        sd_out = out_logs.std(axis=0, ddof=0)

        target_logs = _log_recon_errors(
            request.target_model, X_mem, device=request.device, eps=eps
        )
        log_p_in = _log_normal_pdf(target_logs, mu_in, sd_in)
        log_p_out = _log_normal_pdf(target_logs, mu_out, sd_out)
        # Higher LLR → target's log-recon-error looks more like the IN-pool
        # distribution → x is more likely a member.
        llr = log_p_in - log_p_out
        threshold = float(np.quantile(llr, alpha))
        score = float((llr > threshold).mean())

        details: dict[str, Any] = {
            "task": "autoencoder",
            "n_in_ref_trained": n_in,
            "n_out_ref_trained": n_out,
            "ref_epochs": epochs,
            "subset_fraction": subset_fraction,
            "alpha": alpha,
            "member_set_size": int(X_mem.shape[0]),
            "mean_llr": float(llr.mean()),
            "median_llr": float(np.median(llr)),
            "threshold": threshold,
        }
        return self._wrap(score, t0, details)
