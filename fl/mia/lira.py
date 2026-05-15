"""LiRA — Likelihood Ratio Attack (Carlini et al., 2022).

See ``docs/mia-protocol.md §4`` for the authoritative spec.

Strongest of the three MIAs we run. The expensive step is training
``N_IN + N_OUT`` reference models per attack run — at full scale this is
~8 models × ~2 h each = ~16 GPU-h. On the toy fixtures it's a few
seconds; we accept that the resulting scores are noisy at toy scale and
gate the real assessment to the D11 go/no-go (which explicitly
authorises dropping LiRA from Table 1 if scores don't distinguish
"no unlearning" from "naive retrain").

Algorithm:

  1. Build two pools:
       IN_pool  = auxiliary ∪ member (so IN refs can be trained with
                  exposure to the member set)
       OUT_pool = auxiliary only
  2. Train N_IN reference models on random subsets of IN_pool, and
     N_OUT reference models on random subsets of OUT_pool. Same
     architecture as the target.
  3. For every example in the member set, collect the model's
     true-class softmax probability, clamp to [eps, 1−eps], and apply
     the logit transform  ℓ = log(p / (1 − p)).
  4. Per example, fit Gaussians  N(μ_IN, σ_IN²)  and  N(μ_OUT, σ_OUT²)
     to the corresponding stacks of reference-model logits.
  5. Compute the per-example log-likelihood ratio
        LLR  =  log φ_IN(ℓ_target) − log φ_OUT(ℓ_target)
  6. Classify as "member" if LLR exceeds the ``classification_alpha``
     quantile of LLR values across the member set (online LiRA with
     α=0.5 = median threshold).
  7. Report the fraction of member examples classified as members.
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


def _stack_loader(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for X, y in loader:
        xs.append(X)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _train_quick(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, *,
    device: torch.device, epochs: int, batch_size: int, lr: float,
) -> None:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = criterion(model(X[idx].to(device)), y[idx].to(device))
            loss.backward()
            opt.step()


def _true_class_logits(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, *,
    device: torch.device, eps: float, batch_size: int = 64,
) -> np.ndarray:
    """Per-row logit-of-true-class-probability after eps clamping."""
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size].to(device)
            yb = y[i : i + batch_size].to(device)
            probs = torch.softmax(model(xb), dim=-1)
            p_true = probs.gather(1, yb.unsqueeze(1)).squeeze(1)
            p_clamped = p_true.clamp(min=eps, max=1.0 - eps)
            logit = torch.log(p_clamped / (1.0 - p_clamped))
            out.extend(logit.cpu().tolist())
    return np.asarray(out, dtype=np.float64)


def _log_normal_pdf(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd_safe = np.maximum(sd, 1e-6)
    return -0.5 * ((x - mu) / sd_safe) ** 2 - np.log(sd_safe * math.sqrt(2 * math.pi))


class LiRAMIA(BaseMIA):
    name = "lira"

    def attack(self, request: MIARequest) -> MIAResult:  # noqa: C901  acceptable
        cfg = request.method_config
        n_in = int(cfg.get("num_in_ref", 4))
        n_out = int(cfg.get("num_out_ref", 4))
        eps = float(cfg.get("logit_clamp_eps", 1e-7))
        alpha = float(cfg.get("classification_alpha", 0.5))
        epochs = int(cfg.get("ref_epochs", 5))
        batch_size = int(cfg.get("ref_batch_size", 32))
        lr = float(cfg.get("ref_learning_rate", 1e-3))
        subset_fraction = float(cfg.get("ref_subset_fraction", 0.7))

        t0 = time.time()
        rng = np.random.default_rng(request.seed + 31)

        X_mem, y_mem = _stack_loader(request.member_loader)
        X_aux, y_aux = _stack_loader(request.auxiliary_loader)

        if X_mem.shape[0] == 0:
            raise ValueError("member loader is empty; nothing to attack")
        if X_aux.shape[0] == 0:
            raise ValueError("auxiliary loader is empty; LiRA needs OUT-pool data")

        X_in_pool = torch.cat([X_aux, X_mem], dim=0)
        y_in_pool = torch.cat([y_aux, y_mem], dim=0)

        def _train_one_ref(pool_X: torch.Tensor, pool_y: torch.Tensor) -> nn.Module:
            n = pool_X.shape[0]
            k = max(4, int(round(n * subset_fraction)))
            idx = rng.choice(n, size=min(k, n), replace=False)
            model = request.model_factory()
            _train_quick(
                model,
                pool_X[idx],
                pool_y[idx],
                device=request.device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            return model

        # Train shadow ensemble once and reuse it to evaluate true-class
        # logits on BOTH the member set (for the attack target) and the
        # auxiliary set (for FPR-calibrated thresholding). Sharing the
        # shadows is the standard online-LiRA approximation — strict
        # offline LiRA would condition the IN/OUT split on each example
        # individually, which is intractable at this scale.
        in_logits_mem: list[np.ndarray] = []
        in_logits_aux: list[np.ndarray] = []
        for _ in range(n_in):
            m = _train_one_ref(X_in_pool, y_in_pool)
            in_logits_mem.append(
                _true_class_logits(m, X_mem, y_mem, device=request.device, eps=eps)
            )
            in_logits_aux.append(
                _true_class_logits(m, X_aux, y_aux, device=request.device, eps=eps)
            )

        out_logits_mem: list[np.ndarray] = []
        out_logits_aux: list[np.ndarray] = []
        for _ in range(n_out):
            m = _train_one_ref(X_aux, y_aux)
            out_logits_mem.append(
                _true_class_logits(m, X_mem, y_mem, device=request.device, eps=eps)
            )
            out_logits_aux.append(
                _true_class_logits(m, X_aux, y_aux, device=request.device, eps=eps)
            )

        in_mem = np.stack(in_logits_mem, axis=0)          # [N_IN, N_mem]
        out_mem = np.stack(out_logits_mem, axis=0)        # [N_OUT, N_mem]
        in_aux = np.stack(in_logits_aux, axis=0)          # [N_IN, N_aux]
        out_aux = np.stack(out_logits_aux, axis=0)        # [N_OUT, N_aux]

        def _per_row_llr(
            target_logits: np.ndarray,
            in_arr: np.ndarray, out_arr: np.ndarray,
        ) -> np.ndarray:
            mu_in = in_arr.mean(axis=0)
            sd_in = in_arr.std(axis=0, ddof=0)
            mu_out = out_arr.mean(axis=0)
            sd_out = out_arr.std(axis=0, ddof=0)
            return (
                _log_normal_pdf(target_logits, mu_in, sd_in)
                - _log_normal_pdf(target_logits, mu_out, sd_out)
            )

        target_logits_mem = _true_class_logits(
            request.target_model, X_mem, y_mem, device=request.device, eps=eps
        )
        target_logits_aux = _true_class_logits(
            request.target_model, X_aux, y_aux, device=request.device, eps=eps
        )
        llr_mem = _per_row_llr(target_logits_mem, in_mem, out_mem)
        llr_aux = _per_row_llr(target_logits_aux, in_aux, out_aux)

        # Carlini 2022 online LiRA: threshold at the (1−α) quantile of the
        # auxiliary (non-member) LLR distribution, so exactly α of non-members
        # exceed it (α-controlled FPR). Score = fraction of *member* LLRs
        # above that threshold — the empirical TPR at α-FPR. With this rule
        # 0.5 = chance, ≫ 0.5 = strong membership leakage. (The previous
        # implementation took the threshold from the member set itself,
        # making `score` mechanically equal to 1 − α.)
        threshold = float(np.quantile(llr_aux, 1.0 - alpha))
        score = float((llr_mem > threshold).mean())

        details: dict[str, Any] = {
            "n_in_ref_trained": n_in,
            "n_out_ref_trained": n_out,
            "ref_epochs": epochs,
            "subset_fraction": subset_fraction,
            "alpha": alpha,
            "member_set_size": int(X_mem.shape[0]),
            "aux_set_size": int(X_aux.shape[0]),
            "mean_llr_mem": float(llr_mem.mean()),
            "median_llr_mem": float(np.median(llr_mem)),
            "mean_llr_aux": float(llr_aux.mean()),
            "median_llr_aux": float(np.median(llr_aux)),
            "threshold": threshold,
        }
        return self._wrap(score, t0, details)
