"""Flower-style federated client wrapping a per-apartment CASAS model.

Implements the same interface contract as ``flwr.client.NumPyClient``
(``get_parameters`` / ``set_parameters`` / ``fit`` / ``evaluate``) so the
client is drop-in compatible with Flower simulations should we move to
``flwr.simulation`` later. For D7 we drive rounds manually from
``fl/flower/server.py`` — keeps the orchestration small and removes the
Ray dependency Flower's simulation backend pulls in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    local_epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float


class CasasClient:
    def __init__(
        self,
        client_id: int,
        subset: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        train_cfg: TrainConfig,
        device: torch.device,
        boost_factor: float = 1.0,
        anomaly_weight: float = 0.0,
    ) -> None:
        """
        Args:
            boost_factor: scales the client's *update* (post-local-fit minus
                pre-fit parameters) before returning it for FedAvg. Default
                1.0 is the standard FedAvg client. ``n_total / n_client`` is
                the Bagdasaryan-2020 "model replacement" attack used by the
                canary attacker in D9 — guarantees the global model
                approximately equals this client's local model after
                aggregation, regardless of how strongly the other (clean)
                clients pull back.
            anomaly_weight: coefficient on the L2 anchor
                ``λ · ||θ − θ_global||²`` added to every batch's loss.
                Default 0.0 disables it. Used together with ``boost_factor``
                this is Bagdasaryan-2020's "constrain-and-scale": the
                anchor keeps the local model close to the round's global
                model so the boosted update is small and in the trigger
                direction, not in directions that degrade clean accuracy.
                Clean clients then see a near-optimal model next round
                and don't issue a massive restoration gradient that
                would wash out the canary lesson.
        """
        self.client_id = client_id
        self.subset = subset
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_cfg = train_cfg
        self.device = device
        self.boost_factor = float(boost_factor)
        self.anomaly_weight = float(anomaly_weight)
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()

    # ------- Flower NumPyClient surface ---------------------------------

    def get_parameters(self) -> list[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        # Copy the numpy array so subsequent in-place training updates on
        # p.data don't leak back through shared memory. On MPS / CUDA the
        # `.to(device)` step already forces a copy; on CPU it does not, and
        # the caller's `parameters` list would be mutated silently.
        for p, new in zip(self.model.parameters(), parameters, strict=True):
            p.data = torch.from_numpy(new.copy()).to(self.device).to(p.dtype)

    def fit(self, parameters: list[np.ndarray]) -> tuple[list[np.ndarray], int, dict]:
        # Capture incoming global parameters before fitting so we can scale the
        # delta on return (model-replacement attack support, boost_factor != 1).
        global_params = [p.copy() for p in parameters]

        self.set_parameters(parameters)
        # Snapshot the global parameters AS TENSORS (after set_parameters and
        # device placement) so the anomaly anchor stays addressable from the
        # device-resident graph during backward.
        anchor_tensors: list[torch.Tensor] = (
            [p.detach().clone() for p in self.model.parameters()]
            if self.anomaly_weight > 0.0 else []
        )

        self.model.train()
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        total_loss, total_seen, total_anomaly = 0.0, 0, 0.0
        for _ in range(self.train_cfg.local_epochs):
            for X, y in self.train_loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                opt.zero_grad()
                logits = self.model(X)
                ce_loss = self.criterion(logits, y)
                if self.anomaly_weight > 0.0:
                    # Sum of squared per-parameter deviations from the global model.
                    # Bagdasaryan §3.3: keeps the local update small in directions
                    # that don't help the trigger, which is what allows the boost
                    # to compound across rounds.
                    anomaly = sum(
                        ((p - g) ** 2).sum()
                        for p, g in zip(self.model.parameters(), anchor_tensors, strict=True)
                    )
                    loss = ce_loss + self.anomaly_weight * anomaly
                    total_anomaly += float(anomaly.detach().cpu().item()) * y.size(0)
                else:
                    loss = ce_loss
                loss.backward()
                if self.train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.grad_clip
                    )
                opt.step()
                total_loss += ce_loss.item() * y.size(0)
                total_seen += y.size(0)
        avg_loss = total_loss / max(total_seen, 1)
        avg_anomaly = total_anomaly / max(total_seen, 1) if self.anomaly_weight > 0 else 0.0

        local_params = self.get_parameters()
        if self.boost_factor != 1.0:
            # θ_returned = θ_global + γ · (θ_local − θ_global)
            local_params = [
                g + self.boost_factor * (l - g)
                for g, l in zip(global_params, local_params, strict=True)
            ]

        return (
            local_params,
            len(self.train_loader.dataset),
            {
                "client_id": self.client_id, "subset": self.subset,
                "train_loss": avg_loss, "boost_factor": self.boost_factor,
                "anomaly_weight": self.anomaly_weight,
                "anomaly_l2": avg_anomaly,
            },
        )

    def evaluate(self, parameters: list[np.ndarray]) -> tuple[float, int, dict]:
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                logits = self.model(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * y.size(0)
                total_correct += (logits.argmax(-1) == y).sum().item()
                total += y.size(0)
        return (
            total_loss / max(total, 1),
            total,
            {
                "client_id": self.client_id,
                "subset": self.subset,
                "accuracy": total_correct / max(total, 1),
            },
        )
