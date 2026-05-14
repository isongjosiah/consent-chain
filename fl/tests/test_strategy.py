"""Tests for fl.flower.strategy.federated_average."""

import numpy as np
import pytest

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
