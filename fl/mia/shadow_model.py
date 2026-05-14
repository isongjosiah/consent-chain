"""Shadow-model membership inference attack (Shokri et al., 2017).

See ``docs/mia-protocol.md §3`` for the authoritative spec.

Outline:

  1. Build ``S`` disjoint shadow training sets from the auxiliary loader.
     Each shadow set is further split 50/50 into "in" and "out" halves.
  2. For each shadow set, train a fresh model on the "in" half. Pass
     both halves through to collect feature vectors of top-K softmax
     confidences labelled ``member=1`` / ``member=0``.
  3. Concatenate features across all shadows and train a single
     ``MLPClassifier`` (the attack model).
  4. Pass the **member** loader through ``target_model``, extract the
     same top-K confidence features, and ask the attack model what
     fraction of those examples it classifies as members. That fraction
     is the MIA score.

Notes on the toy-scale behaviour:

  - On real CASAS data, scores cluster in the ranges documented in the
    spec (~0.6–0.75 for "no unlearning", ~0.5 for naive retraining).
  - On the toy fixtures in ``fl/tests/test_mia.py`` we mostly assert
    interface shape; absolute scores are noisy at that scale.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from fl.mia.base import BaseMIA, MIARequest, MIAResult


def _stack_loader(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a DataLoader into a single (X, y) pair."""
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for X, y in loader:
        xs.append(X)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def _top_k_softmax(
    model: nn.Module,
    X: torch.Tensor,
    *,
    k: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Sorted-descending top-K softmax confidences per row."""
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size].to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1).values
            rows.append(top.cpu().numpy())
    return np.concatenate(rows, axis=0)


def _train_quick(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
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
            xb = X[idx].to(device)
            yb = y[idx].to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()


class ShadowModelMIA(BaseMIA):
    name = "shadow_model"

    def attack(self, request: MIARequest) -> MIAResult:  # noqa: C901  acceptable
        cfg = request.method_config
        num_shadow = int(cfg.get("num_shadow", 5))
        top_k = int(cfg.get("confidence_top_k", 3))
        attack_hidden = tuple(cfg.get("hidden_layers", (64, 32)))
        max_iter = int(cfg.get("max_iter", 200))
        epochs = int(cfg.get("shadow_epochs", 5))
        batch_size = int(cfg.get("shadow_batch_size", 32))
        lr = float(cfg.get("shadow_learning_rate", 1e-3))

        # Sklearn import is deferred so users without sklearn loaded see a
        # clear error only when they actually invoke the attack.
        from sklearn.neural_network import MLPClassifier

        t0 = time.time()
        rng = np.random.default_rng(request.seed + 11)

        # Materialize the auxiliary pool once, then carve into S disjoint
        # shadow training sets.
        X_aux, y_aux = _stack_loader(request.auxiliary_loader)
        if X_aux.shape[0] < num_shadow * 4:
            raise ValueError(
                f"auxiliary loader too small for {num_shadow} shadow models "
                f"(have {X_aux.shape[0]} rows, need ≥ {num_shadow * 4})"
            )
        perm = rng.permutation(X_aux.shape[0])
        chunks = np.array_split(perm, num_shadow)

        feat_in: list[np.ndarray] = []
        feat_out: list[np.ndarray] = []

        for shadow_idx, chunk in enumerate(chunks):
            half = len(chunk) // 2
            in_idx = chunk[:half]
            out_idx = chunk[half : 2 * half]
            if len(in_idx) < 2 or len(out_idx) < 2:
                continue  # too few rows for a meaningful shadow
            shadow_model = request.model_factory()
            _train_quick(
                shadow_model,
                X_aux[in_idx],
                y_aux[in_idx],
                device=request.device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            f_in = _top_k_softmax(shadow_model, X_aux[in_idx], k=top_k, device=request.device)
            f_out = _top_k_softmax(shadow_model, X_aux[out_idx], k=top_k, device=request.device)
            feat_in.append(f_in)
            feat_out.append(f_out)

        if not feat_in:
            raise ValueError("no shadow models produced features; auxiliary set too tiny")

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

        # Score the target model on the member set.
        X_mem, _ = _stack_loader(request.member_loader)
        feat_target = _top_k_softmax(
            request.target_model, X_mem, k=top_k, device=request.device
        )
        preds = attack_clf.predict(feat_target)
        score = float(preds.mean())

        details: dict[str, Any] = {
            "num_shadow_trained": len(feat_in),
            "shadow_epochs": epochs,
            "attack_train_size": int(Xa.shape[0]),
            "attack_feature_dim": int(Xa.shape[1]),
            "member_set_size": int(X_mem.shape[0]),
            "attack_train_accuracy": float(attack_clf.score(Xa, ya)),
        }
        return self._wrap(score, t0, details)
