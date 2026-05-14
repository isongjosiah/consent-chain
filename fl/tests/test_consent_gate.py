"""Tests for fl.flower.consent_gate.

We mock Web3 + the three contract handles so this exercises the gate's
*orchestration* (which methods get called with what arguments, error
paths, signer requirements) without touching a live chain. A separate
``test_consent_gate_integration.py`` (D9) will run the real round-trip
against a hardhat node.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fl.flower.consent_gate import (
    ConsentGate,
    ConsentState,
    RoundContext,
    purpose_hash,
    sort_purposes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_manifest() -> dict[str, Any]:
    return {
        "network": "localhost",
        "chainId": 31337,
        "contracts": {
            "ConsentRegistry": {"address": "0x" + "11" * 20},
            "UnlearningOracle": {"address": "0x" + "22" * 20},
            "FLGatekeeper": {"address": "0x" + "33" * 20},
        },
        "verifiers": [],
        "verifierThreshold": 3,
    }


@pytest.fixture
def project_root_with_artifacts(tmp_path: Path) -> Path:
    """Lay out a minimal project root with the ABI JSON files
    ConsentGate's loader expects."""
    art = tmp_path / "contracts" / "artifacts" / "contracts"
    for name in ("ConsentRegistry", "UnlearningOracle", "FLGatekeeper"):
        d = art / f"{name}.sol"
        d.mkdir(parents=True)
        (d / f"{name}.json").write_text(json.dumps({"abi": []}))
    return tmp_path


@pytest.fixture
def gate(project_root_with_artifacts: Path, monkeypatch: pytest.MonkeyPatch) -> ConsentGate:
    """ConsentGate with all web3 traffic replaced by MagicMock."""
    monkeypatch.setenv("LOCALHOST_RPC_URL", "http://stub")
    monkeypatch.setenv(
        "LOCALHOST_PRIVATE_KEY",
        # Anvil/Hardhat default key #0 — well-known test value
        "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    )
    with patch("fl.flower.consent_gate.Web3") as Web3Mock:
        instance = MagicMock()
        instance.is_connected.return_value = True
        instance.eth.contract.side_effect = [MagicMock(), MagicMock(), MagicMock()]
        instance.eth.gas_price = 0
        instance.eth.get_transaction_count.return_value = 0

        def _fake_send_raw_transaction(_raw: Any) -> bytes:
            return b"\x00" * 32

        instance.eth.send_raw_transaction.side_effect = _fake_send_raw_transaction
        instance.eth.wait_for_transaction_receipt.return_value = {
            "status": 1, "transactionHash": b"\x00" * 32, "logs": []
        }
        instance.eth.account.sign_transaction.return_value = MagicMock(
            raw_transaction=b"raw"
        )
        Web3Mock.return_value = instance
        Web3Mock.HTTPProvider = MagicMock(return_value="provider")
        # Web3.keccak is used by purpose_hash; route to a deterministic stub.
        Web3Mock.keccak.side_effect = lambda text: ("k:" + text).encode().ljust(32, b"_")

        g = ConsentGate(
            "localhost",
            project_root=project_root_with_artifacts,
            manifest=_fake_manifest(),
        )
        # Re-bind the three contract handles to fresh, addressable mocks so
        # each test can stub specific functions.
        g.registry = MagicMock()
        g.oracle = MagicMock()
        g.gatekeeper = MagicMock()
        return g


# ---------------------------------------------------------------------------
# Read-only views
# ---------------------------------------------------------------------------


def test_state_of_calls_registry(gate: ConsentGate):
    gate.registry.functions.stateOf.return_value.call.return_value = ConsentState.ACTIVE
    assert gate.state_of(7) == ConsentState.ACTIVE
    gate.registry.functions.stateOf.assert_called_once_with(7)


def test_is_eligible_passes_through_args(gate: ConsentGate):
    gate.registry.functions.isEligible.return_value.call.return_value = True
    p = [purpose_hash("adl"), purpose_hash("research")]
    p_sorted = sort_purposes(p)
    assert gate.is_eligible(1, p_sorted[0], p_sorted, 0)
    gate.registry.functions.isEligible.assert_called_once_with(1, p_sorted[0], p_sorted, 0)


def test_pending_unlearning_count(gate: ConsentGate):
    gate.gatekeeper.functions.pendingUnlearningCount.return_value.call.return_value = 2
    assert gate.pending_unlearning_count() == 2


def test_current_round_id_and_model_hash(gate: ConsentGate):
    gate.gatekeeper.functions.currentRoundId.return_value.call.return_value = 7
    gate.gatekeeper.functions.currentModelHash.return_value.call.return_value = b"\xab" * 32
    assert gate.current_round_id() == 7
    assert gate.current_model_hash() == b"\xab" * 32


def test_model_hash_at_revocation(gate: ConsentGate):
    gate.registry.functions.modelHashAtRevocation.return_value.call.return_value = b"\xcd" * 32
    assert gate.model_hash_at_revocation(3) == b"\xcd" * 32


# ---------------------------------------------------------------------------
# State-changing entry points
# ---------------------------------------------------------------------------


def _stub_build(fn):
    """Return the function-call mock so we can intercept .build_transaction."""
    fn.return_value.build_transaction.return_value = {
        "from": "0x" + "00" * 20, "nonce": 0, "gas": 1, "gasPrice": 0,
    }
    return fn


def test_update_model_hash_builds_and_sends(gate: ConsentGate):
    _stub_build(gate.gatekeeper.functions.updateModelHash)
    tx_hash = gate.update_model_hash(5, b"\x00" * 32)
    gate.gatekeeper.functions.updateModelHash.assert_called_once_with(5, b"\x00" * 32)
    assert tx_hash.startswith("0x")


def test_validate_round_packages_round_context(gate: ConsentGate):
    _stub_build(gate.gatekeeper.functions.validateRound)
    ctx = RoundContext(
        round_id=3,
        round_purpose=purpose_hash("adl"),
        expected_model_hash_pre=b"\x00" * 32,
    )
    gate.validate_round(
        ctx,
        participant_token_ids=[1, 2],
        purpose_arrays=[[ctx.round_purpose], [ctx.round_purpose]],
        purpose_indices=[0, 0],
    )
    gate.gatekeeper.functions.validateRound.assert_called_once()
    args = gate.gatekeeper.functions.validateRound.call_args.args
    assert args[0] == (3, ctx.round_purpose, b"\x00" * 32)
    assert args[1] == [1, 2]


def test_write_calls_require_signer(project_root_with_artifacts: Path, monkeypatch):
    monkeypatch.setenv("LOCALHOST_RPC_URL", "http://stub")
    monkeypatch.delenv("LOCALHOST_PRIVATE_KEY", raising=False)
    with patch("fl.flower.consent_gate.Web3") as Web3Mock:
        instance = MagicMock()
        instance.is_connected.return_value = True
        instance.eth.contract.side_effect = [MagicMock(), MagicMock(), MagicMock()]
        Web3Mock.return_value = instance
        Web3Mock.HTTPProvider = MagicMock()
        g = ConsentGate(
            "localhost",
            project_root=project_root_with_artifacts,
            manifest=_fake_manifest(),
        )
        with pytest.raises(RuntimeError, match="write tx requires signer_key"):
            g.update_model_hash(1, b"\x00" * 32)


# ---------------------------------------------------------------------------
# Construction / error paths
# ---------------------------------------------------------------------------


def test_missing_rpc_env_raises(project_root_with_artifacts: Path, monkeypatch):
    monkeypatch.delenv("LOCALHOST_RPC_URL", raising=False)
    with pytest.raises(ValueError, match="no RPC url"):
        ConsentGate(
            "localhost",
            project_root=project_root_with_artifacts,
            manifest=_fake_manifest(),
        )


def test_missing_artifact_raises(tmp_path: Path, monkeypatch):
    """Project root without `contracts/artifacts/.../*.json` must error early."""
    monkeypatch.setenv("LOCALHOST_RPC_URL", "http://stub")
    with patch("fl.flower.consent_gate.Web3") as Web3Mock:
        Web3Mock.return_value.is_connected.return_value = True
        Web3Mock.HTTPProvider = MagicMock()
        with pytest.raises(FileNotFoundError, match="missing compiled artifact"):
            ConsentGate("localhost", project_root=tmp_path, manifest=_fake_manifest())


def test_missing_manifest_raises(project_root_with_artifacts: Path, monkeypatch):
    monkeypatch.setenv("LOCALHOST_RPC_URL", "http://stub")
    with patch("fl.flower.consent_gate.Web3") as Web3Mock:
        Web3Mock.return_value.is_connected.return_value = True
        Web3Mock.HTTPProvider = MagicMock()
        # Don't pass an explicit manifest → file-based lookup will fail.
        with pytest.raises(FileNotFoundError, match="missing deployment manifest"):
            ConsentGate("localhost", project_root=project_root_with_artifacts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_sort_purposes_is_ascending_bytes():
    a = b"\x01" + b"\x00" * 31
    b = b"\x02" + b"\x00" * 31
    c = b"\x03" + b"\x00" * 31
    assert sort_purposes([c, a, b]) == [a, b, c]


def test_purpose_hash_is_32_bytes_and_deterministic():
    h1 = purpose_hash("adl-recognition")
    h2 = purpose_hash("adl-recognition")
    assert h1 == h2
    assert len(h1) == 32


def test_purpose_hash_differs_per_name():
    assert purpose_hash("a") != purpose_hash("b")
