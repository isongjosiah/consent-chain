"""Flower-style federated client for autoencoder anomaly detection.

Same surface as :class:`fl.flower.client.CasasClient` (get_parameters,
set_parameters, fit, evaluate) but the model is an autoencoder trained
on benign-only data with MSE reconstruction loss. Returns
reconstruction error in fit_info / evaluate_info instead of
classification accuracy.

Loaders provide ``(X,)`` rows (one tensor; no labels), since both the
input and target are the same row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class AETrainConfig:
    local_epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float


class NBaIoTClient:
    def __init__(
        self,
        client_id: int,
        device_name: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        train_cfg: AETrainConfig,
        device: torch.device,
        attack_loader: DataLoader | None = None,
    ) -> None:
        self.client_id = client_id
        self.device_name = device_name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.attack_loader = attack_loader
        self.train_cfg = train_cfg
        self.device = device
        self.model.to(device)
        self.criterion = nn.MSELoss()

    # ---- Flower NumPyClient surface --------------------------------------

    def get_parameters(self) -> list[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        # Copy to avoid the aliasing hazard we hit in CasasClient.
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
            for (X,) in self.train_loader:
                X = X.to(self.device, non_blocking=False)
                opt.zero_grad()
                X_hat = self.model(X)
                loss = self.criterion(X_hat, X)
                loss.backward()
                if self.train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.grad_clip
                    )
                opt.step()
                total_loss += loss.item() * X.size(0)
                total_seen += X.size(0)
        avg_loss = total_loss / max(total_seen, 1)
        return (
            self.get_parameters(),
            len(self.train_loader.dataset),
            {
                "client_id": self.client_id,
                "device": self.device_name,
                "train_recon_loss": avg_loss,
            },
        )

    def evaluate(self, parameters: list[np.ndarray]) -> tuple[float, int, dict]:
        """Returns (mean_test_recon_loss, num_test_samples, info_dict).

        ``info_dict`` also carries ``mean_attack_recon_loss`` if attack
        data is available — useful for the §V anomaly-detection
        evaluation. The reconstruction-error gap
        ``mean_attack_recon - mean_test_recon`` is the headline
        per-device anomaly-detection signal.
        """
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for (X,) in self.test_loader:
                X = X.to(self.device, non_blocking=False)
                X_hat = self.model(X)
                per_sample = ((X_hat - X) ** 2).mean(dim=1)
                total_loss += float(per_sample.sum().item())
                total += X.size(0)
        test_recon = total_loss / max(total, 1)

        attack_recon = None
        if self.attack_loader is not None:
            a_loss, a_total = 0.0, 0
            with torch.no_grad():
                for (X,) in self.attack_loader:
                    X = X.to(self.device, non_blocking=False)
                    X_hat = self.model(X)
                    per_sample = ((X_hat - X) ** 2).mean(dim=1)
                    a_loss += float(per_sample.sum().item())
                    a_total += X.size(0)
            attack_recon = a_loss / max(a_total, 1)

        return (
            test_recon,
            total,
            {
                "client_id": self.client_id,
                "device": self.device_name,
                "test_recon_loss": test_recon,
                "attack_recon_loss": attack_recon,
                "recon_gap": (attack_recon - test_recon) if attack_recon is not None else None,
            },
        )
