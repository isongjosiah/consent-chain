"""Deep autoencoder for N-BaIoT anomaly detection (Meidan et al., 2018).

Architecture per CLAUDE.md §3.1 / config/nbaiot.yaml:
  encoder: 115 → 75 → 50 → 25 → 10  (ReLU, dropout 0.1)
  decoder: 10 → 25 → 50 → 75 → 115  (ReLU, dropout 0.1, final layer linear)

Trained on **benign traffic only**. Anomalies are detected at inference
time by high reconstruction error. Per-sample reconstruction loss is
also the per-client behavioural signature we use to verify unlearning:
pre-unlearning the AE reconstructs each enrolled client's benign
traffic well; post-unlearning the revoked client's traffic should
reconstruct noticeably worse than non-revoked clients.

MLP-only by design — no LSTM, no MPS-backend bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class AutoencoderConfig:
    encoder_dims: tuple[int, ...] = (115, 75, 50, 25, 10)
    decoder_dims: tuple[int, ...] = (10, 25, 50, 75, 115)
    activation: str = "relu"
    dropout: float = 0.1


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in {"tanh"}:
        return nn.Tanh()
    raise ValueError(f"unsupported activation: {name!r}")


class NBaIoTAutoencoder(nn.Module):
    """Symmetric deep autoencoder.

    The encoder bottlenecks 115 → 10. The decoder mirrors it back.
    Last decoder layer is linear (no activation) so the output can
    take any value in the standardized feature space.
    """

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        enc_layers: list[nn.Module] = []
        for in_d, out_d in zip(cfg.encoder_dims[:-1], cfg.encoder_dims[1:]):
            enc_layers.append(nn.Linear(in_d, out_d))
            enc_layers.append(_build_activation(cfg.activation))
            if cfg.dropout > 0:
                enc_layers.append(nn.Dropout(cfg.dropout))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        last_idx = len(cfg.decoder_dims) - 2
        for i, (in_d, out_d) in enumerate(
            zip(cfg.decoder_dims[:-1], cfg.decoder_dims[1:])
        ):
            dec_layers.append(nn.Linear(in_d, out_d))
            if i < last_idx:
                dec_layers.append(_build_activation(cfg.activation))
                if cfg.dropout > 0:
                    dec_layers.append(nn.Dropout(cfg.dropout))
            # final decoder layer is linear (no activation / no dropout)
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def reconstruction_error(
        self, x: torch.Tensor, *, reduction: str = "per_sample"
    ) -> torch.Tensor:
        """MSE between input and reconstruction.

        ``reduction='per_sample'`` (default) returns one error per row —
        useful for the verification reconstruction-error histograms.
        ``reduction='mean'`` returns a scalar suitable as a training loss.
        """
        x_hat = self.forward(x)
        per_feature = (x_hat - x) ** 2
        if reduction == "per_sample":
            return per_feature.mean(dim=1)
        if reduction == "mean":
            return per_feature.mean()
        raise ValueError(f"unsupported reduction: {reduction!r}")


def build_from_yaml(model_cfg: dict[str, Any]) -> NBaIoTAutoencoder:
    """Construct from an ``nbaiot.yaml model:`` block."""
    cfg = AutoencoderConfig(
        encoder_dims=tuple(int(d) for d in model_cfg.get("encoder_dims", (115, 75, 50, 25, 10))),
        decoder_dims=tuple(int(d) for d in model_cfg.get("decoder_dims", (10, 25, 50, 75, 115))),
        activation=str(model_cfg.get("activation", "relu")),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    if cfg.encoder_dims[-1] != cfg.decoder_dims[0]:
        raise ValueError(
            f"encoder bottleneck ({cfg.encoder_dims[-1]}) must equal "
            f"decoder input ({cfg.decoder_dims[0]})"
        )
    if cfg.encoder_dims[0] != cfg.decoder_dims[-1]:
        raise ValueError(
            f"encoder input ({cfg.encoder_dims[0]}) must equal "
            f"decoder output ({cfg.decoder_dims[-1]})"
        )
    return NBaIoTAutoencoder(cfg)
