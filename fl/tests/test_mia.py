"""Toy interface tests for the three MIAs.

These verify the **interface contract** (well-formed ``MIAResult``,
score in [0, 1], fixed-point ×1e4 mirror, method-name uniqueness) on
small synthetic data. Score correctness at full scale is gated by the
D11 go/no-go against the real round-60 CASAS checkpoint.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl.mia.base import MIA_SCALE, BaseMIA, MIARequest, MIAResult, to_fixed_point
from fl.mia.label_only import LabelOnlyMIA
from fl.mia.lira import LiRAMIA
from fl.mia.shadow_model import ShadowModelMIA


# ---------------------------------------------------------------------------
# Synthetic fixture: small classification problem with 3 classes
# ---------------------------------------------------------------------------


def _model_factory(input_dim: int = 4, num_classes: int = 3) -> Callable[[], nn.Module]:
    def f():
        torch.manual_seed(0)
        return nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    return f


def _train_target(member_X: torch.Tensor, member_y: torch.Tensor) -> nn.Module:
    """Train a target model heavily on the members so MIAs have signal."""
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    for _ in range(40):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(member_X), member_y)
        loss.backward()
        opt.step()
    return model


def _make_continuous_request(seed: int = 42) -> MIARequest:
    rng = np.random.default_rng(seed)
    X_mem = torch.from_numpy(rng.normal(size=(40, 4)).astype(np.float32))
    y_mem = torch.from_numpy(rng.integers(0, 3, size=(40,)).astype(np.int64))
    X_aux = torch.from_numpy(rng.normal(size=(120, 4)).astype(np.float32))
    y_aux = torch.from_numpy(rng.integers(0, 3, size=(120,)).astype(np.int64))
    target = _train_target(X_mem, y_mem)
    return MIARequest(
        target_model=target,
        member_loader=DataLoader(TensorDataset(X_mem, y_mem), batch_size=16),
        auxiliary_loader=DataLoader(TensorDataset(X_aux, y_aux), batch_size=16),
        model_factory=_model_factory(),
        method_config={},
        device=torch.device("cpu"),
        seed=seed,
    )


def _make_discrete_request(seed: int = 42) -> MIARequest:
    """For label_only's discrete-input branch (mirrors CASAS shape)."""
    rng = np.random.default_rng(seed)
    X_mem = torch.from_numpy(rng.integers(0, 10, size=(40, 20)).astype(np.int64))
    y_mem = torch.from_numpy(rng.integers(0, 3, size=(40,)).astype(np.int64))
    X_aux = torch.from_numpy(rng.integers(0, 10, size=(120, 20)).astype(np.int64))
    y_aux = torch.from_numpy(rng.integers(0, 3, size=(120,)).astype(np.int64))

    # Tiny embedding + mean-pool + linear classifier — accepts long inputs.
    class TinyEmb(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(11, 4, padding_idx=0)
            self.fc = nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(self.emb(x).mean(dim=1))

    torch.manual_seed(0)
    target = TinyEmb()
    opt = torch.optim.Adam(target.parameters(), lr=1e-1)
    for _ in range(50):
        opt.zero_grad()
        nn.CrossEntropyLoss()(target(X_mem), y_mem).backward()
        opt.step()

    return MIARequest(
        target_model=target,
        member_loader=DataLoader(TensorDataset(X_mem, y_mem), batch_size=16),
        auxiliary_loader=DataLoader(TensorDataset(X_aux, y_aux), batch_size=16),
        model_factory=TinyEmb,
        method_config={},
        device=torch.device("cpu"),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Per-method interface tests
# ---------------------------------------------------------------------------


def _assert_result_shape(result: MIAResult, expected_name: str) -> None:
    assert result.method == expected_name
    assert 0.0 <= result.score <= 1.0
    assert result.score_fixed_point == to_fixed_point(result.score)
    assert 0 <= result.score_fixed_point <= MIA_SCALE
    assert result.wall_seconds > 0
    assert isinstance(result.details, dict)
    assert result.details, f"{expected_name} must populate details"


def test_shadow_model_well_formed():
    request = _make_continuous_request()
    request.method_config.update(
        {"num_shadow": 3, "shadow_epochs": 3, "max_iter": 80, "confidence_top_k": 3}
    )
    res = ShadowModelMIA().attack(request)
    _assert_result_shape(res, "shadow_model")
    assert "num_shadow_trained" in res.details


def test_label_only_well_formed_continuous():
    request = _make_continuous_request()
    request.method_config.update({"perturbations": 5})
    res = LabelOnlyMIA().attack(request)
    _assert_result_shape(res, "label_only")
    assert res.details["perturbation_kind"] == "continuous"


def test_label_only_well_formed_discrete():
    request = _make_discrete_request()
    request.method_config.update({"perturbations": 5, "discrete_replace_fraction": 0.1})
    res = LabelOnlyMIA().attack(request)
    _assert_result_shape(res, "label_only")
    assert res.details["perturbation_kind"] == "discrete"


def test_lira_well_formed():
    request = _make_continuous_request()
    request.method_config.update(
        {"num_in_ref": 2, "num_out_ref": 2, "ref_epochs": 3, "subset_fraction": 0.7}
    )
    res = LiRAMIA().attack(request)
    _assert_result_shape(res, "lira")
    assert "mean_llr" in res.details


# ---------------------------------------------------------------------------
# Fixed-point conversion + cross-method invariants
# ---------------------------------------------------------------------------


def test_to_fixed_point_rounds_half_correctly():
    assert to_fixed_point(0.0000) == 0
    assert to_fixed_point(0.5000) == 5000
    assert to_fixed_point(0.55) == 5500
    assert to_fixed_point(0.5500) == 5500
    assert to_fixed_point(1.0) == MIA_SCALE


def test_to_fixed_point_rejects_out_of_range():
    import pytest

    with pytest.raises(ValueError):
        to_fixed_point(-0.01)
    with pytest.raises(ValueError):
        to_fixed_point(1.01)


def test_method_names_unique():
    names = {ShadowModelMIA().name, LabelOnlyMIA().name, LiRAMIA().name}
    assert len(names) == 3
    assert names == {"shadow_model", "label_only", "lira"}


def test_base_mia_requires_name():
    import pytest

    class _Bad(BaseMIA):
        name = ""

        def attack(self, request):
            ...

    with pytest.raises(ValueError):
        _Bad()
