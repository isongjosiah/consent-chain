"""Autoencoder-aware variants of the three D8 unlearning methods.

Same BaseUnlearner contract as the classifier-side methods in this
package, but the per-batch loss is MSE between reconstruction and
input (not cross-entropy against a class label), and loaders yield
``(X,)`` tuples instead of ``(X, y)``. Sharing the interface means
``fl/attestation/generate_ua.py`` consumes either family identically.

Method semantics mirror the classifier-side docstrings:

  GradientAscentAE  — train on the revoked DS's data with the loss
                      *negated*, pushing the AE *away* from
                      reconstructing it. Cheap and fast.
  SisaFedAE         — retrain a fresh AE from scratch over the
                      remaining clients only (the "naive retraining
                      stub" used until per-shard checkpoints land).
  FedEraserAE       — calibration rounds of FedAvg over the
                      remaining clients starting from the
                      pre-unlearn model. Cheaper than retraining,
                      preserves the AE's existing knowledge of
                      non-revoked clients.

The three classes are concrete BaseUnlearner subclasses with the
same name/method_id as their classifier counterparts so the
on-chain ``UnlearningMethod`` enum stays consistent.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from fl.flower.strategy import federated_average
from fl.unlearning.base import BaseUnlearner, UnlearningRequest, UnlearningResult


def _params_of(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def _load_params(model: nn.Module, params: list[np.ndarray]) -> None:
    for p, new in zip(model.parameters(), params, strict=True):
        p.data = torch.from_numpy(new.copy()).to(p.device).to(p.dtype)


def _unpack(batch: Any) -> torch.Tensor:
    """Loader may yield (X,) for AE or (X, _) for legacy compatibility."""
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


class GradientAscentAE(BaseUnlearner):
    name = "GRADIENT_ASCENT"  # shares on-chain UnlearningMethod id with the classifier variant

    def unlearn(self, request: UnlearningRequest) -> UnlearningResult:
        cfg = request.method_config
        steps = int(cfg.get("steps", 10))
        lr = float(cfg.get("learning_rate", 1e-4))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        loss_cap = cfg.get("loss_cap")

        t0 = time.time()
        model = copy.deepcopy(request.pre_unlearn_model).to(request.device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        step_losses: list[float] = []
        taken = 0
        for _ in range(steps):
            for batch in request.revoked_train_loader:
                X = _unpack(batch).to(request.device, non_blocking=False)
                opt.zero_grad()
                X_hat = model(X)
                recon = ((X_hat - X) ** 2).mean()
                if loss_cap is not None and recon.item() > float(loss_cap):
                    continue
                (-recon).backward()        # ascend on recon error
                opt.step()
                step_losses.append(recon.item())
                taken += 1
                break                       # one inner mini-batch per step

        details: dict[str, Any] = {
            "steps_requested": steps,
            "steps_taken": taken,
            "lr": lr,
            "mean_step_recon": (sum(step_losses) / len(step_losses)) if step_losses else 0.0,
            "final_step_recon": step_losses[-1] if step_losses else 0.0,
            "task": "autoencoder",
        }
        return self._wrap(request, model, t0, details)


class SisaFedAE(BaseUnlearner):
    name = "SISA_FED"

    def unlearn(self, request: UnlearningRequest) -> UnlearningResult:
        cfg = request.method_config
        rounds = int(cfg.get("retrain_rounds", 5))
        local_epochs = int(cfg.get("local_epochs", 2))
        lr = float(cfg.get("learning_rate", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        num_shards = int(cfg.get("num_shards", 3))

        t0 = time.time()
        # One fresh model per remaining client, initialised identically.
        client_models = {cid: request.model_factory().to(request.device)
                         for cid in request.remaining_loaders}
        init = copy.deepcopy(next(iter(client_models.values())).state_dict())
        for m in client_models.values():
            m.load_state_dict(init)

        history: list[float] = []
        for _ in range(rounds):
            updates: list[tuple[list, int]] = []
            for cid, loader in request.remaining_loaders.items():
                m = client_models[cid]
                m.train()
                opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
                running, seen = 0.0, 0
                for _ in range(local_epochs):
                    for batch in loader:
                        X = _unpack(batch).to(request.device, non_blocking=False)
                        opt.zero_grad()
                        loss = ((m(X) - X) ** 2).mean()
                        loss.backward()
                        opt.step()
                        running += loss.item() * X.size(0)
                        seen += X.size(0)
                history.append(running / max(seen, 1))
                updates.append((_params_of(m), len(loader.dataset)))
            avg = federated_average(updates)
            for m in client_models.values():
                _load_params(m, avg)

        post_model = next(iter(client_models.values()))
        details: dict[str, Any] = {
            "retrain_rounds": rounds,
            "local_epochs": local_epochs,
            "remaining_client_ids": sorted(request.remaining_loaders),
            "num_shards_intended": num_shards,
            "approximation_note": (
                "stub uses naive retraining over remaining clients; "
                "production version reloads per-shard SISA checkpoints."
            ),
            "mean_local_recon": (sum(history) / len(history)) if history else 0.0,
            "task": "autoencoder",
        }
        return self._wrap(request, post_model, t0, details)


class FedEraserAE(BaseUnlearner):
    name = "FEDERASER"

    def unlearn(self, request: UnlearningRequest) -> UnlearningResult:
        cfg = request.method_config
        rounds = int(cfg.get("calibration_rounds", 3))
        local_epochs = int(cfg.get("local_epochs", 1))
        lr = float(cfg.get("learning_rate", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 0.0))

        t0 = time.time()
        post = copy.deepcopy(request.pre_unlearn_model).to(request.device)
        init = post.state_dict()
        client_models = {}
        for cid in request.remaining_loaders:
            m = request.model_factory().to(request.device)
            m.load_state_dict(init)
            client_models[cid] = m

        history: list[float] = []
        for _ in range(rounds):
            updates: list[tuple[list, int]] = []
            for cid, loader in request.remaining_loaders.items():
                m = client_models[cid]
                m.train()
                opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
                running, seen = 0.0, 0
                for _ in range(local_epochs):
                    for batch in loader:
                        X = _unpack(batch).to(request.device, non_blocking=False)
                        opt.zero_grad()
                        loss = ((m(X) - X) ** 2).mean()
                        loss.backward()
                        opt.step()
                        running += loss.item() * X.size(0)
                        seen += X.size(0)
                history.append(running / max(seen, 1))
                updates.append((_params_of(m), len(loader.dataset)))
            avg = federated_average(updates)
            _load_params(post, avg)
            for m in client_models.values():
                _load_params(m, avg)

        details: dict[str, Any] = {
            "calibration_rounds": rounds,
            "local_epochs": local_epochs,
            "remaining_client_ids": sorted(request.remaining_loaders),
            "mean_calibration_recon": (sum(history) / len(history)) if history else 0.0,
            "approximation_note": (
                "stub implements calibration only; production version "
                "replays cached per-round per-client updates excluding "
                "the revoked client."
            ),
            "task": "autoencoder",
        }
        return self._wrap(request, post, t0, details)
