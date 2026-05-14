"""Tests for fl.attestation — UA assembly, signing, and submission orchestration.

The critical test is :func:`test_ua_hash_matches_ethers_reference` — a
fixed-value cross-check against an ethers.js-computed hash, locking the
Python ABI encoding to the byte layout the Solidity oracle recomputes
on-chain.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from eth_account import Account

from fl.attestation.generate_ua import (
    UnlearningAttestation,
    build_ua,
    compute_ua_hash,
)
from fl.attestation.submit_ua import address_from_private_key, submit_attestation
from fl.attestation.verify_ua import (
    recover_signer,
    sign_ua_hash,
    simulate_verifier_signatures,
)
from fl.mia.base import MIAResult
from fl.unlearning.base import UnlearningResult


# ---------------------------------------------------------------------------
# UA hash byte-for-byte parity with the Solidity recomputation
# ---------------------------------------------------------------------------


def test_ua_hash_matches_ethers_reference():
    """Reference hash produced by ethers v6 ``defaultAbiCoder.encode`` over
    the same Solidity type tuple; locking this prevents silent drift in
    ``compute_ua_hash`` away from what the on-chain oracle recomputes."""
    expected = bytes.fromhex(
        "a12c6a5ebe95b73b43ab64f6f0ae1ea08408bb6fe96fcea89eee67964abcb9ab"
    )
    got = compute_ua_hash(
        chain_id=1337,
        ds_token_id=42,
        model_hash_pre=b"\xaa" * 32,
        model_hash_post=b"\xbb" * 32,
        method=1,                  # GRADIENT_ASCENT
        shadow_mia_score=5100,
        lira_score=4900,
        label_only_mia_score=5000,
        timestamp=1_715_731_200,
    )
    assert got == expected


def test_ua_hash_changes_with_chain_id():
    """Cross-chain replay protection — same UA on a different chain id
    must produce a different hash."""
    kwargs = dict(
        ds_token_id=1, model_hash_pre=b"\x01" * 32, model_hash_post=b"\x02" * 32,
        method=0, shadow_mia_score=5000, lira_score=5000, label_only_mia_score=5000,
        timestamp=100,
    )
    h_purechain = compute_ua_hash(chain_id=900_520_900_520, **kwargs)
    h_amoy = compute_ua_hash(chain_id=80_002, **kwargs)
    assert h_purechain != h_amoy


def test_ua_hash_changes_with_every_field():
    """Each input field must contribute to the hash (no silent dropping)."""
    base = dict(
        chain_id=1, ds_token_id=1,
        model_hash_pre=b"\x00" * 32, model_hash_post=b"\x00" * 32,
        method=0, shadow_mia_score=0, lira_score=0, label_only_mia_score=0,
        timestamp=0,
    )
    h0 = compute_ua_hash(**base)
    for k, v in [
        ("ds_token_id", 2),
        ("model_hash_pre", b"\xff" * 32),
        ("model_hash_post", b"\xff" * 32),
        ("method", 1),
        ("shadow_mia_score", 1),
        ("lira_score", 1),
        ("label_only_mia_score", 1),
        ("timestamp", 1),
    ]:
        bumped = {**base, k: v}
        assert compute_ua_hash(**bumped) != h0, f"hash unchanged when {k} bumped"


def test_compute_ua_hash_accepts_hex_strings_for_model_hash():
    h_bytes = compute_ua_hash(
        chain_id=1, ds_token_id=1,
        model_hash_pre=b"\xaa" * 32, model_hash_post=b"\xbb" * 32,
        method=0, shadow_mia_score=0, lira_score=0, label_only_mia_score=0,
        timestamp=0,
    )
    h_hex = compute_ua_hash(
        chain_id=1, ds_token_id=1,
        model_hash_pre="0x" + "aa" * 32, model_hash_post="0x" + "bb" * 32,
        method=0, shadow_mia_score=0, lira_score=0, label_only_mia_score=0,
        timestamp=0,
    )
    assert h_bytes == h_hex


def test_compute_ua_hash_rejects_out_of_range_inputs():
    with pytest.raises(ValueError):
        compute_ua_hash(
            chain_id=1, ds_token_id=1,
            model_hash_pre=b"\x00" * 31,  # wrong length
            model_hash_post=b"\x00" * 32,
            method=0, shadow_mia_score=0, lira_score=0, label_only_mia_score=0,
            timestamp=0,
        )
    with pytest.raises(ValueError):
        compute_ua_hash(
            chain_id=1, ds_token_id=1,
            model_hash_pre=b"\x00" * 32, model_hash_post=b"\x00" * 32,
            method=0,
            shadow_mia_score=20_000,  # > MIA_SCALE
            lira_score=0, label_only_mia_score=0,
            timestamp=0,
        )
    with pytest.raises(ValueError):
        compute_ua_hash(
            chain_id=1, ds_token_id=1,
            model_hash_pre=b"\x00" * 32, model_hash_post=b"\x00" * 32,
            method=300,  # > uint8
            shadow_mia_score=0, lira_score=0, label_only_mia_score=0,
            timestamp=0,
        )


# ---------------------------------------------------------------------------
# build_ua wiring
# ---------------------------------------------------------------------------


def _mia_result(method: str, score: float) -> MIAResult:
    fp = int(round(score * 10_000))
    return MIAResult(method=method, score=score, score_fixed_point=fp, wall_seconds=0.01)


def _unlearning_result() -> UnlearningResult:
    return UnlearningResult(
        method="GRADIENT_ASCENT",
        method_id=1,
        post_unlearn_model=torch.nn.Linear(1, 1),
        model_hash_pre="0x" + "aa" * 32,
        model_hash_post="0x" + "bb" * 32,
        wall_seconds=0.5,
        details={},
    )


def test_build_ua_packages_fields_correctly():
    ua = build_ua(
        chain_id=900_520_900_520,
        ds_token_id=5,
        unlearning_result=_unlearning_result(),
        mia_results={
            "shadow_model": _mia_result("shadow_model", 0.51),
            "lira": _mia_result("lira", 0.49),
            "label_only": _mia_result("label_only", 0.50),
        },
        timestamp=1_700_000_000,
    )
    assert ua.ds_token_id == 5
    assert ua.method == 1  # GRADIENT_ASCENT
    assert ua.shadow_mia_score == 5100
    assert ua.lira_score == 4900
    assert ua.label_only_mia_score == 5000
    assert ua.timestamp == 1_700_000_000
    # ua_hash matches a direct recomputation.
    expected_hash = compute_ua_hash(
        chain_id=900_520_900_520, ds_token_id=5,
        model_hash_pre=b"\xaa" * 32, model_hash_post=b"\xbb" * 32,
        method=1, shadow_mia_score=5100, lira_score=4900,
        label_only_mia_score=5000, timestamp=1_700_000_000,
    )
    assert ua.ua_hash == expected_hash


def test_build_ua_requires_all_three_mia_methods():
    with pytest.raises(KeyError, match="missing methods"):
        build_ua(
            chain_id=1, ds_token_id=1,
            unlearning_result=_unlearning_result(),
            mia_results={"shadow_model": _mia_result("shadow_model", 0.5)},
        )


def test_ua_to_contract_tuple_layout():
    ua = UnlearningAttestation(
        ds_token_id=1,
        model_hash_pre=b"\x01" * 32, model_hash_post=b"\x02" * 32,
        method=2, shadow_mia_score=100, lira_score=200, label_only_mia_score=300,
        timestamp=42, ua_hash=b"\x09" * 32,
    )
    tup = ua.to_contract_tuple()
    # Order must match the Solidity struct: dsTokenId, pre, post, method,
    # shadow, lira, labelOnly, timestamp, uaHash.
    assert tup == (1, b"\x01" * 32, b"\x02" * 32, 2, 100, 200, 300, 42, b"\x09" * 32)


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


_TEST_KEY = (
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)
_TEST_ADDR = Account.from_key(_TEST_KEY).address


def test_sign_ua_hash_produces_65_byte_signature():
    sig = sign_ua_hash(b"\xab" * 32, _TEST_KEY)
    assert isinstance(sig, bytes)
    assert len(sig) == 65


def test_signature_roundtrips_to_signer_address():
    h = b"\xcd" * 32
    sig = sign_ua_hash(h, _TEST_KEY)
    recovered = recover_signer(h, sig)
    assert recovered == _TEST_ADDR


def test_simulate_verifier_signatures_in_order():
    keys = [
        "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
        "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cde8b0c7",
        "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
    ]
    h = b"\xef" * 32
    sigs = simulate_verifier_signatures(h, keys)
    assert len(sigs) == 3
    recovered = [recover_signer(h, s) for s in sigs]
    expected = [Account.from_key(k).address for k in keys]
    assert recovered == expected


def test_sign_rejects_wrong_size_hash():
    with pytest.raises(ValueError):
        sign_ua_hash(b"\x00" * 31, _TEST_KEY)


def test_recover_rejects_wrong_size_hash():
    sig = sign_ua_hash(b"\x00" * 32, _TEST_KEY)
    with pytest.raises(ValueError):
        recover_signer(b"\x00" * 31, sig)


# ---------------------------------------------------------------------------
# Submission orchestration (mocked)
# ---------------------------------------------------------------------------


def _stub_gate():
    g = MagicMock()
    g.account = Account.from_key(_TEST_KEY)
    g.w3.eth.get_transaction_count.return_value = 0
    g.w3.eth.gas_price = 0
    g.w3.eth.send_raw_transaction.return_value = b"\x00" * 32
    g.w3.eth.wait_for_transaction_receipt.return_value = {
        "status": 1, "transactionHash": b"\x00" * 32, "logs": []
    }
    g.w3.eth.account.sign_transaction.return_value = MagicMock(
        raw_transaction=b"raw", rawTransaction=b"raw"
    )
    return g


def _ua():
    return UnlearningAttestation(
        ds_token_id=1,
        model_hash_pre=b"\xaa" * 32, model_hash_post=b"\xbb" * 32,
        method=1, shadow_mia_score=5100, lira_score=4900, label_only_mia_score=5000,
        timestamp=100, ua_hash=b"\xcc" * 32,
    )


def test_submit_attestation_calls_oracle_with_tuple_and_sigs():
    gate = _stub_gate()
    ua = _ua()
    sigs = [b"\x00" * 65, b"\x11" * 65, b"\x22" * 65]
    submit_attestation(gate, ua, sigs)
    gate.oracle.functions.submitAttestation.assert_called_once_with(
        ua.to_contract_tuple(), sigs
    )


def test_submit_attestation_returns_hex_tx_hash():
    gate = _stub_gate()
    tx = submit_attestation(gate, _ua(), [b"\x00" * 65, b"\x11" * 65, b"\x22" * 65])
    assert tx.startswith("0x")
    assert len(tx) == 2 + 64


def test_submit_attestation_validates_signature_size():
    gate = _stub_gate()
    with pytest.raises(ValueError, match="65"):
        submit_attestation(gate, _ua(), [b"\x00" * 64])


def test_submit_attestation_requires_account():
    gate = MagicMock()
    gate.account = None
    with pytest.raises(RuntimeError, match="signer key"):
        submit_attestation(gate, _ua(), [b"\x00" * 65, b"\x11" * 65, b"\x22" * 65])


def test_address_from_private_key():
    assert address_from_private_key(_TEST_KEY) == _TEST_ADDR
