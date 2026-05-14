"""Label-only membership inference attack (Choquette-Choo et al., 2021).

See ``docs/mia-protocol.md §5`` for the authoritative spec.

Black-box assumption: the attacker observes only argmax predictions,
not softmax confidences. The membership signal is **decision-boundary
robustness** — a member example's prediction tends to be more stable
under small input perturbations than a non-member's, because the model
has overfit a wider basin around the training point.

Per-example robustness:
    R(x, y) = (# of K perturbations of x that still predict y) / K

Membership rule:
    classify x as member if R(x, y) > threshold_{class(y)}

Per-class thresholds are calibrated on the auxiliary loader so that
exactly ``fpr_target`` (default 5 %) of auxiliary examples — which are
non-members by construction — would be flagged as members.

Two perturbation strategies, selected by input dtype:

  - Discrete (int / long inputs, e.g. CASAS observation IDs): replace
    ``discrete_replace_fraction`` of *non-pad* positions with a random
    id from the same column's value range.
  - Continuous (float inputs, e.g. N-BaIoT features): add Gaussian
    noise with std ``continuous_noise_sigma``.
"""

from __future__ import annotations

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


def _perturb_discrete(
    X: torch.Tensor,
    *,
    replace_fraction: float,
    rng: np.random.Generator,
    padding_idx: int = 0,
) -> torch.Tensor:
    """Replace a fraction of non-pad positions with a random other id."""
    Xn = X.clone()
    arr = Xn.numpy()
    n_rows, n_cols = arr.shape
    if arr.size == 0:
        return Xn
    vocab_max = int(arr.max(initial=padding_idx)) + 1
    for r in range(n_rows):
        non_pad = np.flatnonzero(arr[r] != padding_idx)
        if non_pad.size == 0:
            continue
        n_replace = max(1, int(round(non_pad.size * replace_fraction)))
        sel = rng.choice(non_pad, size=min(n_replace, non_pad.size), replace=False)
        for j in sel:
            current = int(arr[r, j])
            candidates = list(range(vocab_max))
            candidates.remove(current)
            if not candidates:
                continue
            arr[r, j] = candidates[int(rng.integers(0, len(candidates)))]
    return Xn


def _perturb_continuous(
    X: torch.Tensor,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    noise = torch.from_numpy(rng.normal(0.0, sigma, size=X.shape).astype(np.float32))
    return X + noise.to(X.dtype)


def _predict(model: nn.Module, X: torch.Tensor, device: torch.device, batch_size: int = 64) -> torch.Tensor:
    model.eval()
    out: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size].to(device)
            out.append(model(xb).argmax(-1).cpu())
    return torch.cat(out, dim=0)


def _robustness_scores(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    K: int,
    rng: np.random.Generator,
    is_discrete: bool,
    discrete_replace_fraction: float,
    continuous_noise_sigma: float,
    padding_idx: int,
    device: torch.device,
) -> np.ndarray:
    n = X.shape[0]
    stable_counts = np.zeros(n, dtype=np.int64)
    for _ in range(K):
        if is_discrete:
            Xk = _perturb_discrete(
                X, replace_fraction=discrete_replace_fraction, rng=rng,
                padding_idx=padding_idx,
            )
        else:
            Xk = _perturb_continuous(X, sigma=continuous_noise_sigma, rng=rng)
        preds = _predict(model, Xk, device=device)
        stable_counts += (preds == y).numpy().astype(np.int64)
    return stable_counts / float(K)


def _per_class_thresholds(
    robustness: np.ndarray,
    labels: np.ndarray,
    fpr_target: float,
) -> dict[int, float]:
    """For each class observed in ``labels``, pick the (1 − fpr_target)-quantile
    of that class's auxiliary-robustness scores — so exactly ``fpr_target`` of
    auxiliary examples for that class will exceed the threshold."""
    quantile = 1.0 - fpr_target
    thresholds: dict[int, float] = {}
    for cls in np.unique(labels):
        cls_scores = robustness[labels == cls]
        if cls_scores.size == 0:
            thresholds[int(cls)] = float("inf")  # never classify as member
        else:
            thresholds[int(cls)] = float(np.quantile(cls_scores, quantile))
    return thresholds


class LabelOnlyMIA(BaseMIA):
    name = "label_only"

    def attack(self, request: MIARequest) -> MIAResult:
        cfg = request.method_config
        K = int(cfg.get("perturbations", 20))
        discrete_replace_fraction = float(cfg.get("discrete_replace_fraction", 0.05))
        continuous_noise_sigma = float(cfg.get("continuous_noise_sigma", 0.01))
        fpr_target = float(cfg.get("fpr_target", 0.05))
        padding_idx = int(cfg.get("padding_idx", 0))

        t0 = time.time()
        rng = np.random.default_rng(request.seed + 23)

        X_mem, y_mem = _stack_loader(request.member_loader)
        X_aux, y_aux = _stack_loader(request.auxiliary_loader)
        is_discrete = X_mem.dtype in (torch.int32, torch.int64, torch.long)

        kwargs = dict(
            K=K,
            rng=rng,
            is_discrete=is_discrete,
            discrete_replace_fraction=discrete_replace_fraction,
            continuous_noise_sigma=continuous_noise_sigma,
            padding_idx=padding_idx,
            device=request.device,
        )

        r_aux = _robustness_scores(request.target_model, X_aux, y_aux, **kwargs)
        thresholds = _per_class_thresholds(r_aux, y_aux.numpy(), fpr_target)

        r_mem = _robustness_scores(request.target_model, X_mem, y_mem, **kwargs)
        # Classify each member example using its class threshold (∞ if class
        # never seen in auxiliary set → never flagged, which is conservative).
        thr_per_row = np.array(
            [thresholds.get(int(c), float("inf")) for c in y_mem.numpy()],
            dtype=np.float64,
        )
        classified_member = r_mem > thr_per_row
        score = float(classified_member.mean())

        details: dict[str, Any] = {
            "K": K,
            "perturbation_kind": "discrete" if is_discrete else "continuous",
            "discrete_replace_fraction": discrete_replace_fraction if is_discrete else None,
            "continuous_noise_sigma": None if is_discrete else continuous_noise_sigma,
            "fpr_target": fpr_target,
            "member_set_size": int(X_mem.shape[0]),
            "aux_set_size": int(X_aux.shape[0]),
            "mean_member_robustness": float(r_mem.mean()),
            "mean_aux_robustness": float(r_aux.mean()),
            "thresholds_per_class": thresholds,
        }
        return self._wrap(score, t0, details)
