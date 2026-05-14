"""Shared interface for unlearning methods.

ConsentChain is **method-agnostic** by design: the on-chain protocol
only cares about ``(model_hash_pre, model_hash_post, MIA scores)``. Any
implementation that produces a post-unlearning model the three MIAs
can certify as below-threshold is acceptable. The classes here express
that contract explicitly so SISA-Fed, Gradient Ascent, and FedEraser
can be swapped in run_casas-style experiments via configuration.

UnlearningMethod ids (uint8 on-chain, see contracts-spec.md §2.1):
  SISA_FED         = 0
  GRADIENT_ASCENT  = 1
  FEDERASER        = 2
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from fl.unlearning.model_hash import hash_model


@dataclass
class UnlearningRequest:
    """Inputs an unlearner needs to produce an unlearning-of-DS_i model.

    Attributes:
        target_client_id: federation index of the revoked DS.
        revoked_train_loader: the DS's training data (some methods need
            this to compute gradients to "ascend"; others ignore it).
        remaining_loaders: ``{client_id: DataLoader}`` for the non-revoked
            clients — used by methods that retrain over the remaining
            federation (SISA-Fed, FedEraser calibration).
        pre_unlearn_model: the model at the moment of revocation
            (i.e. the model whose hash was captured by
            ``ConsentRegistry.modelHashAtRevocation``).
        model_factory: builds a fresh untrained model with the same
            architecture; SISA-Fed needs it for retrain-from-scratch.
        method_config: method-specific hyperparameters from
            ``casas.yaml`` / ``nbaiot.yaml`` ``unlearning.<method>`` block.
        device: torch device to run on.
    """

    target_client_id: int
    revoked_train_loader: DataLoader
    remaining_loaders: dict[int, DataLoader]
    pre_unlearn_model: nn.Module
    model_factory: Callable[[], nn.Module]
    method_config: dict[str, Any] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


@dataclass
class UnlearningResult:
    """Output of an unlearning run — feeds the UA hashing in fl/attestation/."""

    method: str                      # "SISA_FED" | "GRADIENT_ASCENT" | "FEDERASER"
    method_id: int                   # uint8 mirror of the on-chain enum
    post_unlearn_model: nn.Module
    model_hash_pre: str              # hash of request.pre_unlearn_model
    model_hash_post: str             # hash of post_unlearn_model
    wall_seconds: float
    details: dict[str, Any] = field(default_factory=dict)


METHOD_ID = {
    "SISA_FED": 0,
    "GRADIENT_ASCENT": 1,
    "FEDERASER": 2,
}


class BaseUnlearner(abc.ABC):
    """Abstract base for an unlearning algorithm."""

    #: ``"SISA_FED"`` / ``"GRADIENT_ASCENT"`` / ``"FEDERASER"`` — used for
    #: the ``method`` field of ``UnlearningResult`` and to look up
    #: ``METHOD_ID`` for the on-chain UA ``UnlearningMethod`` enum.
    name: str = ""

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set a non-empty name")

    @abc.abstractmethod
    def unlearn(self, request: UnlearningRequest) -> UnlearningResult:
        """Produce a model with the target DS's contribution removed."""

    # -- Helpers shared by all subclasses --------------------------------

    @staticmethod
    def _hash(model: nn.Module) -> str:
        return hash_model(model)

    def _wrap(
        self,
        request: UnlearningRequest,
        post_model: nn.Module,
        started_at: float,
        details: dict[str, Any] | None = None,
    ) -> UnlearningResult:
        return UnlearningResult(
            method=self.name,
            method_id=METHOD_ID[self.name],
            post_unlearn_model=post_model,
            model_hash_pre=self._hash(request.pre_unlearn_model),
            model_hash_post=self._hash(post_model),
            wall_seconds=time.time() - started_at,
            details=details or {},
        )
