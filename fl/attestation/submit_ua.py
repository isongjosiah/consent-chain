"""Submit a packaged UA + signatures to the on-chain oracle.

Thin wrapper around the ``ConsentGate``'s ``oracle`` contract handle —
takes the assembled :class:`UnlearningAttestation` and signatures, and
calls ``submitAttestation`` with the right tuple layout.

Round-trip with the on-chain oracle (contracts-spec.md §4.5):
  1. Oracle recomputes the ``uaHash`` from the tuple fields →
     ``UaReplay`` if mismatch.
  2. Oracle checks ``seenUaHash`` → ``UaReplay`` if already accepted.
  3. Oracle checks ``shadowMia < τ``, ``lira < τ``, ``labelOnly < τ`` →
     ``MiaThresholdExceeded`` if any fails.
  4. Oracle checks ``ua.modelHashPre == registry.modelHashAtRevocation``
     → ``ModelHashMismatch`` if not.
  5. Oracle checks ``stateOf(dsTokenId) ∈ {REVOKED, EXPIRED}`` →
     ``WrongConsentState`` if not.
  6. Oracle recovers each signature, requires distinct registered
     verifiers ≥ ``verifierThreshold`` → ``InsufficientSignatures`` /
     ``NotVerifier`` / ``DuplicateSigner``.
  7. State transitions: ``seenUaHash[hash]=true``,
     ``lastUaHash[tokenId]=hash``, ``registry.transitionToUnlearned``,
     ``AttestationAccepted`` emitted.

The CLI is reserved for E3+ ("revoke client 3 at round 60 → submit
UA"); for one-off interactive use, import ``submit_attestation`` from
your own driver.
"""

from __future__ import annotations

import logging
from typing import Iterable

from eth_account import Account
from web3.exceptions import ContractLogicError

from fl.attestation.generate_ua import UnlearningAttestation
from fl.flower.consent_gate import ConsentGate

LOGGER = logging.getLogger("submit_ua")


def submit_attestation(
    gate: ConsentGate,
    ua: UnlearningAttestation,
    signatures: Iterable[bytes],
    *,
    gas_limit: int = 600_000,
) -> str:
    """Send the ``submitAttestation`` transaction. Returns the 0x-prefixed
    tx hash on success; raises ``ContractLogicError`` if the oracle
    reverts.

    The caller is responsible for collecting signatures (see
    ``fl/attestation/verify_ua.py``) and for ensuring the consent token
    is in ``REVOKED`` or ``EXPIRED`` state before calling this.
    """
    if not gate.account:
        raise RuntimeError("submit_attestation requires a signer key on the gate")
    sig_list = list(signatures)
    if not sig_list:
        raise ValueError("at least one signature required")
    for i, s in enumerate(sig_list):
        if len(s) != 65:
            raise ValueError(f"signature[{i}] is {len(s)} bytes; expected 65")

    fn = gate.oracle.functions.submitAttestation(
        ua.to_contract_tuple(), sig_list
    )
    nonce = gate.w3.eth.get_transaction_count(gate.account.address)
    tx = fn.build_transaction(
        {
            "from": gate.account.address,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": gate.w3.eth.gas_price,
        }
    )
    signed = gate.w3.eth.account.sign_transaction(tx, gate.account.key)
    raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
    tx_hash = gate.w3.eth.send_raw_transaction(raw)
    receipt = gate.w3.eth.wait_for_transaction_receipt(tx_hash)
    if receipt["status"] != 1:
        raise ContractLogicError(f"submitAttestation reverted: 0x{tx_hash.hex()}")
    LOGGER.info(
        "UA accepted: dsTokenId=%d uaHash=0x%s tx=0x%s",
        ua.ds_token_id, ua.ua_hash.hex(), tx_hash.hex(),
    )
    return f"0x{tx_hash.hex()}"


def address_from_private_key(private_key: str) -> str:
    """Convenience helper used by the test driver to look up which
    verifier a given key corresponds to."""
    return Account.from_key(private_key).address
