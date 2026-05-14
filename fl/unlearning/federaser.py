"""FedEraser — federated unlearning via aggregation-history replay.

Reference: Liu, G. et al. "FedEraser: Enabling Efficient Client-Level
Data Removal from Federated Learning Models." IEEE/ACM IWQoS 2021.

Production design (target for D8 full implementation):

  During normal training, the server caches each round's per-client
  update gradients ``g_(c, r) = θ_post(c, r) − θ_pre(r)``. To unlearn
  client c*:
    1. Replay aggregation **excluding** c*'s recorded updates:
         θ̂_(r+1) = θ̂_r + average({g_(c, r) : c ≠ c*})
    2. Apply a small **calibration** training step on remaining clients
       (typically 1–3 rounds, much cheaper than a full retrain) so the
       replayed model corrects for the implicit changes c*'s historical
       updates pulled into the global parameters.

D8 stub implementation:

  D7's baseline does not currently cache per-client per-round update
  gradients (it only stores aggregated round checkpoints); the
  per-client cache will be added in the D8 follow-up that converts
  ``fl/flower/server.py`` to retain updates alongside checkpoints.
  Until then this stub implements only the **calibration step** —
  starting from the pre-unlearn aggregated model, run a small number
  of FedAvg rounds over the remaining clients. The MIA suite at D11
  will still distinguish this from "no unlearning" because the
  calibration moves the model away from any DS-i-specific behaviour
  even without the replay step; the difference vs. full FedEraser
  will appear in Table 1 as a slightly weaker MIA-drop column.
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


class FedEraserUnlearner(BaseUnlearner):
    name = "FEDERASER"

    def unlearn(self, request: UnlearningRequest) -> UnlearningResult:
        cfg = request.method_config
        calibration_rounds = int(cfg.get("calibration_rounds", 3))
        local_epochs = int(cfg.get("local_epochs", 1))
        lr = float(cfg.get("learning_rate", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 1e-5))

        t0 = time.time()

        # Start from the pre-unlearn aggregated model.
        post_model = copy.deepcopy(request.pre_unlearn_model).to(request.device)
        criterion = nn.CrossEntropyLoss()

        # One client-model per remaining client, initialised from post_model.
        client_models: dict[int, nn.Module] = {}
        init_state = post_model.state_dict()
        for cid in request.remaining_loaders:
            m = request.model_factory().to(request.device)
            m.load_state_dict(init_state)
            client_models[cid] = m

        round_losses: list[float] = []

        for round_idx in range(calibration_rounds):
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
                round_losses.append(running / max(seen, 1))
                updates.append((_params_of(model), len(loader.dataset)))

            avg = federated_average(updates)
            _load_params(post_model, avg)
            for m in client_models.values():
                _load_params(m, avg)

        details: dict[str, Any] = {
            "calibration_rounds": calibration_rounds,
            "local_epochs": local_epochs,
            "remaining_client_ids": sorted(request.remaining_loaders),
            "mean_calibration_loss": (
                sum(round_losses) / len(round_losses) if round_losses else 0.0
            ),
            "approximation_note": (
                "stub implements calibration only; production version replays "
                "cached per-round per-client updates excluding the revoked "
                "client and applies calibration on top (TODO D8-full)."
            ),
        }
        return self._wrap(request, post_model, t0, details)
