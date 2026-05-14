"""Shared interface for membership inference attacks.

Each attack consumes a target model + a "member" loader (the revoked
DS's data) + an "auxiliary" loader (data the model never trained on,
used to fit or calibrate the attack), and returns a ``MIAResult``
with the in-[0,1] score plus its ×1e4 fixed-point mirror that the
on-chain ``UnlearningAttestation`` carries.

See ``docs/mia-protocol.md`` for the authoritative spec.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


MIA_SCALE = 10_000


@dataclass
class MIARequest:
    """Inputs every MIA needs.

    Attributes:
        target_model: the model being attacked (typically ``M_post``,
            the post-unlearning model under attestation).
        member_loader: the **revoked DS's** data — the set the attack
            tries to detect as "members" of the training distribution.
        auxiliary_loader: a disjoint set of data from the same
            distribution (the remaining clients' held-out test data,
            for CASAS). Used by shadow / label-only to train or
            calibrate the attack.
        model_factory: builds a fresh untrained model with the same
            architecture as ``target_model``; shadow / LiRA need this
            to train shadow / reference models.
        method_config: method-specific hyperparameters from the YAML.
        device: torch device for forward passes.
        seed: experiment-wide seed; per-attack RNG streams should be
            derived via ``seed + offset`` for reproducibility.
    """

    target_model: nn.Module
    member_loader: DataLoader
    auxiliary_loader: DataLoader
    model_factory: Callable[[], nn.Module]
    method_config: dict[str, Any] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    seed: int = 42


@dataclass
class MIAResult:
    """Output of an MIA run; fixed-point score goes on-chain."""

    method: str
    score: float                  # in [0, 1]
    score_fixed_point: int        # round(score * MIA_SCALE)
    wall_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


def to_fixed_point(score: float) -> int:
    """Convert a [0, 1] score to the on-chain ×1e4 fixed-point integer."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score out of [0, 1]: {score}")
    return int(round(score * MIA_SCALE))


class BaseMIA(abc.ABC):
    """Abstract base for membership-inference attacks."""

    name: str = ""

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set a non-empty name")

    @abc.abstractmethod
    def attack(self, request: MIARequest) -> MIAResult:
        """Compute the MIA score and return a populated ``MIAResult``."""

    def _wrap(
        self,
        score: float,
        started_at: float,
        details: dict[str, Any] | None = None,
    ) -> MIAResult:
        return MIAResult(
            method=self.name,
            score=score,
            score_fixed_point=to_fixed_point(score),
            wall_seconds=time.time() - started_at,
            details=details or {},
        )
