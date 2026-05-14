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
    ) -> None:
        self.client_id = client_id
        self.subset = subset
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_cfg = train_cfg
        self.device = device
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
        self.set_parameters(parameters)
        self.model.train()
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )
        total_loss, total_seen = 0.0, 0
        for _ in range(self.train_cfg.local_epochs):
            for X, y in self.train_loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                opt.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                if self.train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.grad_clip
                    )
                opt.step()
                total_loss += loss.item() * y.size(0)
                total_seen += y.size(0)
        avg_loss = total_loss / max(total_seen, 1)
        return (
            self.get_parameters(),
            len(self.train_loader.dataset),
            {"client_id": self.client_id, "subset": self.subset, "train_loss": avg_loss},
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
