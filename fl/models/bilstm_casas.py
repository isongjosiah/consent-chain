"""BiLSTM activity-recognition model for CASAS smart-home sensor streams.

Architecture per CLAUDE.md §3.1 / config/casas.yaml:
  Embedding(vocab_size, 64) → BiLSTM(64 → 128 × 2 layers, dropout 0.3) → Linear(256, 12).

Inputs are pre-padded int sequences of length 2000 produced by
``fl/datasets/download_casas.py``. The last time-step's hidden state
feeds the classifier (pre-padding puts the most recent events at the
sequence tail, so this aligns with the genuine end-of-activity).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BiLSTMConfig:
    vocab_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    num_classes: int = 12
    padding_idx: int = 0


class BiLSTMClassifier(nn.Module):
    """Pre-padded sequence → last-step BiLSTM read-out → softmax classifier."""

    def __init__(self, cfg: BiLSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(
            cfg.vocab_size,
            cfg.embedding_dim,
            padding_idx=cfg.padding_idx,
        )
        self.lstm = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(out_dim, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] int64 sequence of observation ids
        e = self.embedding(x)                  # [B, T, E]
        out, _ = self.lstm(e)                  # [B, T, H * D]
        pooled = out[:, -1, :]                 # [B, H * D]
        return self.classifier(self.dropout(pooled))  # [B, C]


def build_from_yaml(model_cfg: dict, vocab_size: int) -> BiLSTMClassifier:
    """Construct the model from a casas.yaml ``model`` block + global vocab size."""
    cfg = BiLSTMConfig(
        vocab_size=vocab_size,
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        bidirectional=bool(model_cfg.get("bidirectional", True)),
    )
    return BiLSTMClassifier(cfg)
