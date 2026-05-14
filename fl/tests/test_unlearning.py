"""Toy interface tests for the three unlearning methods.

These don't verify unlearning *quality* (that's what the MIA suite does
on real data at D11) — they verify the **interface contract**: every
method produces a valid ``UnlearningResult`` with stable hashes, an
unchanged pre-model, and an actual change in parameters.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl.unlearning.base import METHOD_ID, BaseUnlearner, UnlearningRequest
from fl.unlearning.federaser import FedEraserUnlearner
from fl.unlearning.gradient_ascent import GradientAscentUnlearner
from fl.unlearning.model_hash import hash_model
from fl.unlearning.sisa_fed import SisaFedUnlearner


def _toy_model_factory() -> Callable[[], nn.Module]:
    def f():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))

    return f


def _toy_loader(seed: int, n: int = 32) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn((n, 4), generator=g)
    y = torch.randint(0, 3, (n,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=True)


def _toy_request(
    method_config: dict | None = None,
) -> UnlearningRequest:
    factory = _toy_model_factory()
    pre_model = factory()
    # Apply a couple of training steps so the pre-unlearn model has
    # non-trivial parameters (otherwise gradient ascent on near-zero
    # gradients could leave hashes nearly identical at toy scale).
    opt = torch.optim.Adam(pre_model.parameters(), lr=1e-2)
    for X, y in _toy_loader(seed=42):
        opt.zero_grad()
        nn.CrossEntropyLoss()(pre_model(X), y).backward()
        opt.step()

    return UnlearningRequest(
        target_client_id=0,
        revoked_train_loader=_toy_loader(seed=100),
        remaining_loaders={1: _toy_loader(seed=200), 2: _toy_loader(seed=201)},
        pre_unlearn_model=pre_model,
        model_factory=factory,
        method_config=method_config or {},
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Per-method shape + non-trivial-update tests
# ---------------------------------------------------------------------------


def _assert_result_well_formed(unlearner: BaseUnlearner, request: UnlearningRequest):
    pre_hash_recorded = hash_model(request.pre_unlearn_model)
    result = unlearner.unlearn(request)
    # Interface fields populated correctly.
    assert result.method == unlearner.name
    assert result.method_id == METHOD_ID[unlearner.name]
    assert result.model_hash_pre == pre_hash_recorded
    assert result.wall_seconds > 0
    assert isinstance(result.details, dict)
    # Pre-unlearn model untouched by the unlearner (hash unchanged).
    assert hash_model(request.pre_unlearn_model) == pre_hash_recorded
    # Output model actually moved.
    assert result.model_hash_post != result.model_hash_pre
    # Output model is a distinct object (not aliasing).
    assert result.post_unlearn_model is not request.pre_unlearn_model


def test_gradient_ascent_well_formed():
    _assert_result_well_formed(
        GradientAscentUnlearner(),
        _toy_request(method_config={"steps": 5, "learning_rate": 1e-3}),
    )


def test_sisa_fed_well_formed():
    _assert_result_well_formed(
        SisaFedUnlearner(),
        _toy_request(
            method_config={"retrain_rounds": 2, "local_epochs": 1, "learning_rate": 1e-2}
        ),
    )


def test_federaser_well_formed():
    _assert_result_well_formed(
        FedEraserUnlearner(),
        _toy_request(
            method_config={"calibration_rounds": 2, "local_epochs": 1, "learning_rate": 1e-2}
        ),
    )


# ---------------------------------------------------------------------------
# Cross-method invariants
# ---------------------------------------------------------------------------


def test_method_ids_match_onchain_enum_order():
    assert METHOD_ID["SISA_FED"] == 0
    assert METHOD_ID["GRADIENT_ASCENT"] == 1
    assert METHOD_ID["FEDERASER"] == 2


def test_method_names_unique():
    names = {SisaFedUnlearner().name, GradientAscentUnlearner().name, FedEraserUnlearner().name}
    assert len(names) == 3


def test_details_capture_method_specific_metadata():
    """Each method's `details` dict must include its identifying knobs so the
    UA pre-image / paper artefacts can be reconstructed from results alone."""
    cfg_ga = {"steps": 3, "learning_rate": 1e-3}
    res_ga = GradientAscentUnlearner().unlearn(_toy_request(method_config=cfg_ga))
    assert "steps_requested" in res_ga.details
    assert "mean_step_loss" in res_ga.details

    cfg_sisa = {"retrain_rounds": 1, "local_epochs": 1}
    res_sisa = SisaFedUnlearner().unlearn(_toy_request(method_config=cfg_sisa))
    assert "retrain_rounds" in res_sisa.details
    assert "remaining_client_ids" in res_sisa.details

    cfg_fe = {"calibration_rounds": 1, "local_epochs": 1}
    res_fe = FedEraserUnlearner().unlearn(_toy_request(method_config=cfg_fe))
    assert "calibration_rounds" in res_fe.details


def test_base_unlearner_requires_name():
    """A subclass with empty name should fail at instantiation."""
    import pytest

    class _BadUnlearner(BaseUnlearner):
        name = ""

        def unlearn(self, request):
            ...

    with pytest.raises(ValueError):
        _BadUnlearner()
