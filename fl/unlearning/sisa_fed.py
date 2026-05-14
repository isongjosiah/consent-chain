"""SISA-Fed unlearning — federated adaptation of Bourtoule et al., 2021.

Reference: Bourtoule, L. et al. "Machine Unlearning." IEEE S&P 2021.

Production design (target for D8 full implementation):

  Pre-training (one-time setup, before any revocation can occur):
    For each client c ∈ {0, …, C-1}:
      Split c's training data into S disjoint shards.
      Train a sub-model per shard, save its checkpoint.
    Aggregate per-shard sub-models with FedAvg as usual.

  Unlearning for revoked client c*:
    For each shard s ∈ {0, …, S-1}:
      Reload sub-model_(c*, s) from its pre-revocation checkpoint.
      Retrain sub-model_(c*, s) on shards of c* that *exclude* the
        revoked-data subset (in our setup we revoke the entire client,
        so this collapses to "skip c* entirely").
    Re-aggregate. The unlearned global model is the FedAvg of all
    sub-models from {c ≠ c*} ∪ {empty contributions from c*}.

D8 stub implementation:

  Run D7's baseline currently writes only aggregated round checkpoints,
  not per-shard sub-models — so the production SISA-Fed flow is not yet
  available. The stub below uses the **closest tractable approximation**:
  starting from a fresh ``model_factory()`` instance, retrain to
  convergence over only the remaining clients' data (the gold-standard
  baseline labeled "naive retraining" in CLAUDE.md §3.1 — present as a
  table column under both SISA-Fed and a separate Retraining row).

The stub is correct in spirit (it removes the revoked DS's contribution
entirely) but slow at scale. The full D8 implementation will add
shard-level checkpointing to fl/flower/server.py and replace the
``unlearn()`` body with the proper SISA reload-and-retrain loop.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
from torch import nn

from fl.flower.strategy import federated_average
from fl.unlearning.base import BaseUnlearner, UnlearningRequest, UnlearningResult


def _params_of(model: nn.Module) -> list:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def _load_params(model: nn.Module, params: list) -> None:
    for p, new in zip(model.parameters(), params, strict=True):
        p.data = torch.from_numpy(new).to(p.device).to(p.dtype)


class SisaFedUnlearner(BaseUnlearner):
    name = "SISA_FED"

    def unlearn(self, request: UnlearningRequest) -> UnlearningResult:
        cfg = request.method_config
        retrain_rounds = int(cfg.get("retrain_rounds", 5))
        local_epochs = int(cfg.get("local_epochs", 2))
        lr = float(cfg.get("learning_rate", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 1e-5))
        # ``num_shards`` is honoured by the production implementation; the
        # stub records it in details so reviewers can see the intended
        # configuration without inferring it.
        num_shards = int(cfg.get("num_shards", 3))

        t0 = time.time()

        # Build one fresh client model per remaining client.
        client_models: dict[int, nn.Module] = {
            cid: request.model_factory().to(request.device)
            for cid in request.remaining_loaders
        }
        # All clients start from the same fresh initialization for fairness.
        init_state = copy.deepcopy(next(iter(client_models.values())).state_dict())
        for m in client_models.values():
            m.load_state_dict(init_state)

        history_loss: list[float] = []
        criterion = nn.CrossEntropyLoss()

        for round_idx in range(retrain_rounds):
            # ---- Local fit on each remaining client. ----
            updates: list[tuple[list, int]] = []
            for cid, loader in request.remaining_loaders.items():
                model = client_models[cid]
                model.train()
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                seen = 0
                running = 0.0
                for _ in range(local_epochs):
                    for X, y in loader:
                        X = X.to(request.device, non_blocking=True)
                        y = y.to(request.device, non_blocking=True)
                        opt.zero_grad()
                        loss = criterion(model(X), y)
                        loss.backward()
                        opt.step()
                        running += loss.item() * y.size(0)
                        seen += y.size(0)
                history_loss.append(running / max(seen, 1))
                updates.append((_params_of(model), len(loader.dataset)))

            # ---- FedAvg across remaining clients only. ----
            avg = federated_average(updates)
            for m in client_models.values():
                _load_params(m, avg)

        # The aggregated model — any client_model has the same params now.
        post_model = next(iter(client_models.values()))

        details: dict[str, Any] = {
            "retrain_rounds": retrain_rounds,
            "local_epochs": local_epochs,
            "remaining_client_ids": sorted(request.remaining_loaders),
            "num_shards_intended": num_shards,
            "approximation_note": (
                "stub uses naive retraining over remaining clients; production "
                "version reloads pre-shard SISA checkpoints (TODO D8-full)."
            ),
            "mean_local_loss": (sum(history_loss) / len(history_loss)) if history_loss else 0.0,
        }
        return self._wrap(request, post_model, t0, details)
