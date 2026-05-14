"""Tests for fl.unlearning.model_hash — must be deterministic + collision-sensitive."""

import torch
from torch import nn

from fl.unlearning.model_hash import hash_model, hash_state_dict


def _tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def test_hash_format_is_0x_prefixed_hex():
    h = hash_model(_tiny_model())
    assert h.startswith("0x")
    assert len(h) == 2 + 64  # sha256 is 32 bytes → 64 hex chars
    int(h, 16)  # raises if non-hex


def test_same_parameters_same_hash():
    m1 = _tiny_model(seed=7)
    m2 = _tiny_model(seed=7)
    assert hash_model(m1) == hash_model(m2)


def test_different_parameters_different_hash():
    m1 = _tiny_model(seed=7)
    m2 = _tiny_model(seed=8)
    assert hash_model(m1) != hash_model(m2)


def test_state_dict_key_order_does_not_matter():
    """Re-arranging the state_dict iteration order must not change the hash."""
    m = _tiny_model(seed=42)
    sd = m.state_dict()
    reversed_sd = dict(reversed(list(sd.items())))
    assert hash_state_dict(sd) == hash_state_dict(reversed_sd)


def test_hash_changes_after_a_single_parameter_perturbation():
    m = _tiny_model(seed=42)
    h_before = hash_model(m)
    with torch.no_grad():
        first_param = next(m.parameters())
        first_param.add_(1e-6)  # tiny shift
    h_after = hash_model(m)
    assert h_before != h_after


def test_hash_stable_across_repeated_calls():
    """Two consecutive hashes on an unchanged model must match (no state leak)."""
    m = _tiny_model(seed=1)
    assert hash_model(m) == hash_model(m)
