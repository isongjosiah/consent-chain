"""Toy interface tests for the three AE-side MIAs.

Same shape as ``test_mia.py`` but the target is an autoencoder, the
loaders yield ``(X,)`` tuples, and the membership signal is
reconstruction error. Confirms each AE-MIA produces a well-formed
``MIAResult`` and uses the *same* on-chain method names as the
classifier MIAs (so ``fl.attestation.generate_ua`` accepts results
from either family without modification).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl.mia.base import MIA_SCALE, MIARequest, MIAResult, to_fixed_point
from fl.mia.lira_ae import LiRAAEMIA
from fl.mia.recon_threshold import ReconThresholdMIA
from fl.mia.shadow_model_ae import ShadowModelAEMIA


def _ae_factory(input_dim: int = 8, bottleneck: int = 2) -> Callable[[], nn.Module]:
    def f() -> nn.Module:
        torch.manual_seed(0)
        return nn.Sequential(
            nn.Linear(input_dim, 4), nn.ReLU(),
            nn.Linear(4, bottleneck), nn.ReLU(),
            nn.Linear(bottleneck, 4), nn.ReLU(),
            nn.Linear(4, input_dim),
        )
    return f


def _train_target_ae(member_X: torch.Tensor, epochs: int = 80) -> nn.Module:
    """Heavily fit on the members so the MIAs have signal."""
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(8, 4), nn.ReLU(),
        nn.Linear(4, 2), nn.ReLU(),
        nn.Linear(2, 4), nn.ReLU(),
        nn.Linear(4, 8),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(member_X) - member_X) ** 2).mean()
        loss.backward()
        opt.step()
    return model


def _make_request(seed: int = 42) -> MIARequest:
    rng = np.random.default_rng(seed)
    # Distinct distributions: members ~ N(0, 1); auxiliary ~ N(2, 1) so a
    # member-fit AE reconstructs members easily and auxiliary rows poorly.
    X_mem = torch.from_numpy(rng.normal(loc=0.0, size=(40, 8)).astype(np.float32))
    X_aux = torch.from_numpy(rng.normal(loc=2.0, size=(160, 8)).astype(np.float32))
    target = _train_target_ae(X_mem)
    return MIARequest(
        target_model=target,
        member_loader=DataLoader(TensorDataset(X_mem), batch_size=16),
        auxiliary_loader=DataLoader(TensorDataset(X_aux), batch_size=16),
        model_factory=_ae_factory(input_dim=8, bottleneck=2),
        method_config={},
        device=torch.device("cpu"),
        seed=seed,
    )


def _assert_result_shape(result: MIAResult, expected_name: str) -> None:
    assert result.method == expected_name
    assert 0.0 <= result.score <= 1.0
    assert result.score_fixed_point == to_fixed_point(result.score)
    assert 0 <= result.score_fixed_point <= MIA_SCALE
    assert result.wall_seconds > 0
    assert isinstance(result.details, dict)
    assert result.details.get("task") == "autoencoder"


def test_recon_threshold_well_formed():
    request = _make_request()
    request.method_config.update({"fpr_target": 0.05})
    res = ReconThresholdMIA().attack(request)
    _assert_result_shape(res, "label_only")
    # On strongly-distinguishable distributions the threshold attack
    # should flag *most* members. Loose lower bound to keep the test
    # robust to seed variation.
    assert res.score > 0.5, f"expected most members below threshold; got {res.score}"


def test_shadow_model_ae_well_formed():
    request = _make_request()
    request.method_config.update(
        {"num_shadow": 3, "shadow_epochs": 5, "max_iter": 80}
    )
    res = ShadowModelAEMIA().attack(request)
    _assert_result_shape(res, "shadow_model")
    assert "num_shadow_trained" in res.details


def test_lira_ae_well_formed():
    request = _make_request()
    request.method_config.update(
        {"num_in_ref": 2, "num_out_ref": 2, "ref_epochs": 5, "subset_fraction": 0.7}
    )
    res = LiRAAEMIA().attack(request)
    _assert_result_shape(res, "lira")
    assert "mean_llr" in res.details


def test_ae_mia_names_share_onchain_slots_with_classifier_mias():
    """Critical: AE and classifier MIAs use the same `name` strings so
    fl.attestation.generate_ua.build_ua() accepts either family
    without modification."""
    # Classifier MIAs:
    from fl.mia.label_only import LabelOnlyMIA
    from fl.mia.lira import LiRAMIA
    from fl.mia.shadow_model import ShadowModelMIA

    classifier_names = {ShadowModelMIA().name, LiRAMIA().name, LabelOnlyMIA().name}
    ae_names = {ShadowModelAEMIA().name, LiRAAEMIA().name, ReconThresholdMIA().name}
    assert classifier_names == ae_names == {"shadow_model", "lira", "label_only"}


def test_member_below_aux_for_recon_threshold():
    """The threshold attack relies on mean(member_recon) < mean(aux_recon).
    If this invariant fails on toy data, the test setup is broken."""
    request = _make_request()
    request.method_config.update({"fpr_target": 0.05})
    res = ReconThresholdMIA().attack(request)
    assert res.details["mean_member_recon"] < res.details["mean_aux_recon"], (
        f"member recon {res.details['mean_member_recon']:.4f} should be lower "
        f"than aux recon {res.details['mean_aux_recon']:.4f}"
    )
