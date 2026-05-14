"""Shadow-model membership inference for autoencoders.

AE analogue of ``fl.mia.shadow_model.ShadowModelMIA``. Same Shokri-2017
structure (S shadow models → attack classifier on confidence features
→ apply to target), but:

  - shadow models are autoencoders, trained with MSE on benign-only rows
  - the per-sample "confidence" feature is the reconstruction error
    (we additionally feed per-feature errors so the attack classifier
    can pick up which slices distinguish member vs non-member)
  - attack model is the same ``sklearn.neural_network.MLPClassifier``

Same on-chain ``shadow_model`` name as the classifier variant; the
``fl.attestation`` pipeline therefore consumes either family uniformly.

Reference: Shokri et al., 2017, generalised to AE per Hilprecht et al.,
2019.
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
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _stack_loader(loader: DataLoader) -> torch.Tensor:
    """Materialise an unlabeled loader into a single tensor."""
    out: list[torch.Tensor] = []
    for batch in loader:
        out.append(_unpack(batch))
    return torch.cat(out, dim=0)


def _per_feature_errors(
    model: nn.Module, X: torch.Tensor, *, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    """Returns per-sample squared-error vector (shape ``[N, F]``).

    Aggregated to scalar in the attack classifier; full vector kept here
    so the classifier can learn which features signal membership.
    """
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size].to(device)
            xb_hat = model(xb)
            per_feature = (xb_hat - xb) ** 2
            rows.append(per_feature.cpu().numpy())
    return np.concatenate(rows, axis=0)


def _train_shadow_ae(
    model: nn.Module,
    X: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X[idx].to(device)
            opt.zero_grad()
            loss = ((model(xb) - xb) ** 2).mean()
            loss.backward()
            opt.step()


def _feature_vector_from_per_feature_errors(per_feature_err: np.ndarray) -> np.ndarray:
    """Compact attack features per sample.

    Concatenate scalar summary statistics over the per-feature MSE
    vector — mean (the global recon error), std, max, min, and the 90/99
    quantiles. Five numbers per sample is enough for the MLP to learn
    the member-vs-non-member boundary without overfitting on small
    shadow datasets.
    """
    return np.stack(
        [
            per_feature_err.mean(axis=1),
            per_feature_err.std(axis=1),
            per_feature_err.max(axis=1),
            per_feature_err.min(axis=1),
            np.quantile(per_feature_err, 0.90, axis=1),
            np.quantile(per_feature_err, 0.99, axis=1),
        ],
        axis=1,
    )


class ShadowModelAEMIA(BaseMIA):
    name = "shadow_model"

    def attack(self, request: MIARequest) -> MIAResult:  # noqa: C901  acceptable
        cfg = request.method_config
        num_shadow = int(cfg.get("num_shadow", 5))
        attack_hidden = tuple(cfg.get("hidden_layers", (32, 16)))
        max_iter = int(cfg.get("max_iter", 200))
        epochs = int(cfg.get("shadow_epochs", 5))
        batch_size = int(cfg.get("shadow_batch_size", 256))
        lr = float(cfg.get("shadow_learning_rate", 1e-3))

        from sklearn.neural_network import MLPClassifier

        t0 = time.time()
        rng = np.random.default_rng(request.seed + 11)

        X_aux = _stack_loader(request.auxiliary_loader)
        if X_aux.shape[0] < num_shadow * 4:
            raise ValueError(
                f"auxiliary set too small for {num_shadow} shadow AEs "
                f"(have {X_aux.shape[0]} rows, need ≥ {num_shadow * 4})"
            )
        perm = rng.permutation(X_aux.shape[0])
        chunks = np.array_split(perm, num_shadow)

        feat_in: list[np.ndarray] = []
        feat_out: list[np.ndarray] = []

        for chunk in chunks:
            half = len(chunk) // 2
            in_idx = chunk[:half]
            out_idx = chunk[half : 2 * half]
            if len(in_idx) < 2 or len(out_idx) < 2:
                continue
            shadow = request.model_factory()
            _train_shadow_ae(
                shadow, X_aux[in_idx],
                device=request.device,
                epochs=epochs, batch_size=batch_size, lr=lr,
            )
            f_in = _feature_vector_from_per_feature_errors(
                _per_feature_errors(shadow, X_aux[in_idx], device=request.device)
            )
            f_out = _feature_vector_from_per_feature_errors(
                _per_feature_errors(shadow, X_aux[out_idx], device=request.device)
            )
            feat_in.append(f_in)
            feat_out.append(f_out)

        if not feat_in:
            raise ValueError("no shadow AEs produced features; auxiliary set too tiny")

        Xa_in = np.concatenate(feat_in, axis=0)
        Xa_out = np.concatenate(feat_out, axis=0)
        Xa = np.concatenate([Xa_in, Xa_out], axis=0)
        ya = np.concatenate(
            [np.ones(Xa_in.shape[0]), np.zeros(Xa_out.shape[0])], axis=0
        ).astype(np.int64)

        attack_clf = MLPClassifier(
            hidden_layer_sizes=attack_hidden,
            max_iter=max_iter,
            random_state=request.seed + 17,
        )
        attack_clf.fit(Xa, ya)

        X_mem = _stack_loader(request.member_loader)
        feat_target = _feature_vector_from_per_feature_errors(
            _per_feature_errors(request.target_model, X_mem, device=request.device)
        )
        preds = attack_clf.predict(feat_target)
        score = float(preds.mean())

        details: dict[str, Any] = {
            "task": "autoencoder",
            "num_shadow_trained": len(feat_in),
            "shadow_epochs": epochs,
            "attack_train_size": int(Xa.shape[0]),
            "attack_feature_dim": int(Xa.shape[1]),
            "member_set_size": int(X_mem.shape[0]),
            "attack_train_accuracy": float(attack_clf.score(Xa, ya)),
        }
        return self._wrap(score, t0, details)
