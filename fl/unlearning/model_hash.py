"""SHA-256 hashing of PyTorch model parameters.

The hash is deterministic given identical parameter values, regardless
of save format / torch.save internals. It feeds:

  - per-round on-chain ``FLGatekeeper.updateModelHash(roundId, h)``
    (recorded against the in-flight round)
  - Unlearning Attestation ``modelHashPre`` / ``modelHashPost``
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import torch


def hash_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Stable SHA-256 over sorted (key, value-bytes) pairs."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode("utf-8"))
        t = state_dict[key].detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
    return "0x" + h.hexdigest()


def hash_model(model: torch.nn.Module) -> str:
    return hash_state_dict(model.state_dict())
