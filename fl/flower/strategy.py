"""Weighted FedAvg aggregation, decoupled from the round driver."""

from __future__ import annotations

import numpy as np


def federated_average(
    updates: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    """Vanilla FedAvg: weight each client's parameters by its training-set size.

    Args:
        updates: list of (parameters, num_examples) returned by ``CasasClient.fit``.

    Returns:
        Aggregated parameter list, same shapes as input.
    """
    if not updates:
        raise ValueError("federated_average requires at least one update")
    total = sum(n for _, n in updates)
    if total <= 0:
        raise ValueError("aggregate denominator is zero — no training examples")

    # Initialize accumulators to zeros, then accumulate weighted sums.
    template, _ = updates[0]
    aggregated = [np.zeros_like(p, dtype=np.float64) for p in template]
    for params, n in updates:
        w = n / total
        for i, p in enumerate(params):
            aggregated[i] += p.astype(np.float64) * w
    # Cast back to the original dtype of each tensor.
    return [a.astype(template[i].dtype) for i, a in enumerate(aggregated)]
