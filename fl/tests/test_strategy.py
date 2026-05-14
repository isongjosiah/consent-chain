"""Tests for fl.flower.strategy.federated_average and the
model-replacement boost in fl.flower.client.CasasClient."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl.flower.client import CasasClient, TrainConfig
from fl.flower.strategy import federated_average


def test_single_update_is_pass_through():
    p = [np.array([1.0, 2.0]), np.array([[3.0, 4.0]])]
    out = federated_average([(p, 100)])
    for a, b in zip(out, p, strict=True):
        np.testing.assert_allclose(a, b)


def test_equal_weights_gives_mean():
    p1 = [np.array([1.0, 1.0]), np.array([2.0])]
    p2 = [np.array([3.0, 3.0]), np.array([4.0])]
    out = federated_average([(p1, 50), (p2, 50)])
    np.testing.assert_allclose(out[0], [2.0, 2.0])
    np.testing.assert_allclose(out[1], [3.0])


def test_unequal_weights_gives_weighted_mean():
    """3:1 ratio — heavier client dominates."""
    p1 = [np.array([0.0])]
    p2 = [np.array([4.0])]
    out = federated_average([(p1, 300), (p2, 100)])
    # (0*300 + 4*100) / 400 = 1.0
    np.testing.assert_allclose(out[0], [1.0])


def test_empty_updates_raises():
    with pytest.raises(ValueError):
        federated_average([])


def test_zero_total_examples_raises():
    p = [np.array([1.0])]
    with pytest.raises(ValueError):
        federated_average([(p, 0), (p, 0)])


def test_dtype_preserved():
    p1 = [np.array([1.0], dtype=np.float32)]
    p2 = [np.array([3.0], dtype=np.float32)]
    out = federated_average([(p1, 1), (p2, 1)])
    assert out[0].dtype == np.float32
    np.testing.assert_allclose(out[0], [2.0])


def test_multilayer_alignment_preserved():
    """Order of arrays in the output matches the input layout."""
    p1 = [np.zeros((3,)), np.zeros((2, 4))]
    p2 = [np.ones((3,)), np.ones((2, 4))]
    out = federated_average([(p1, 1), (p2, 1)])
    assert out[0].shape == (3,)
    assert out[1].shape == (2, 4)
    np.testing.assert_allclose(out[0], 0.5)
    np.testing.assert_allclose(out[1], 0.5)


# ---------------------------------------------------------------------------
# CasasClient boost_factor (model-replacement attack)
# ---------------------------------------------------------------------------


def _toy_client(boost_factor: float) -> CasasClient:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 3))
    X = torch.randn(32, 4)
    y = torch.randint(0, 3, (32,))
    loader = DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)
    return CasasClient(
        client_id=0, subset="toy", model=model,
        train_loader=loader, test_loader=loader,
        train_cfg=TrainConfig(local_epochs=1, learning_rate=1e-2, weight_decay=0, grad_clip=0),
        device=torch.device("cpu"),
        boost_factor=boost_factor,
    )


def _params_arrays(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def test_boost_factor_one_is_identity():
    """boost_factor=1.0 must produce the same returned params as plain fit."""
    c = _toy_client(boost_factor=1.0)
    initial = _params_arrays(c.model)
    returned, _, _ = c.fit(initial)
    # Returned params should equal the post-fit local params (no scaling).
    post_fit = _params_arrays(c.model)
    for a, b in zip(returned, post_fit, strict=True):
        np.testing.assert_allclose(a, b)


def test_boost_factor_scales_update_linearly():
    """θ_returned = θ_global + γ · (θ_local − θ_global). Pick γ=3, verify
    the returned delta from global is exactly 3× the unscaled delta."""
    c1 = _toy_client(boost_factor=1.0)
    initial1 = _params_arrays(c1.model)
    returned1, _, _ = c1.fit(initial1)
    delta_unscaled = [r - g for r, g in zip(returned1, initial1)]

    c3 = _toy_client(boost_factor=3.0)
    initial3 = _params_arrays(c3.model)
    # Both clients are seeded identically (manual_seed(0) inside _toy_client),
    # so their local trajectories under the same data + lr are identical.
    returned3, _, _ = c3.fit(initial3)
    delta_scaled = [r - g for r, g in zip(returned3, initial3)]

    for unscaled, scaled in zip(delta_unscaled, delta_scaled, strict=True):
        np.testing.assert_allclose(scaled, 3.0 * unscaled, rtol=1e-5, atol=1e-7)


def test_boost_factor_reported_in_fit_info():
    c = _toy_client(boost_factor=2.5)
    _, _, info = c.fit(_params_arrays(c.model))
    assert info["boost_factor"] == 2.5
