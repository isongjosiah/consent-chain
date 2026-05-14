"""ECDSA signing helpers for the Verifier Committee.

The on-chain ``UnlearningOracle.submitAttestation`` recovers ECDSA
signers via OpenZeppelin's ``ECDSA.recover`` after applying
``MessageHashUtils.toEthSignedMessageHash`` — i.e. each signer must
sign the 32-byte ``uaHash`` with the standard "\x19Ethereum Signed
Message:\n32" prefix.

For the BCCA prototype (CLAUDE.md §3.1 — ECDSA multi-sig 3-of-5), the
VC's "independent re-runs" are simulated by signing the same ``uaHash``
with each verifier's pre-registered private key. Production swaps this
for BLS threshold sigs (journal scope, contracts-spec.md §11).
"""

from __future__ import annotations

from typing import Iterable

from eth_account import Account
from eth_account.messages import encode_defunct


def sign_ua_hash(ua_hash: bytes, private_key: str) -> bytes:
    """Produce a 65-byte ECDSA signature compatible with ``ECDSA.recover``
    after the Ethereum-prefixed message hash."""
    if len(ua_hash) != 32:
        raise ValueError(f"ua_hash must be 32 bytes; got {len(ua_hash)}")
    message = encode_defunct(primitive=ua_hash)
    signed = Account.sign_message(message, private_key=private_key)
    sig = signed.signature
    if len(sig) != 65:
        raise RuntimeError(f"unexpected signature length: {len(sig)}")
    return bytes(sig)


def recover_signer(ua_hash: bytes, signature: bytes) -> str:
    """Inverse of ``sign_ua_hash`` — returns the recovered EOA address."""
    if len(ua_hash) != 32:
        raise ValueError(f"ua_hash must be 32 bytes; got {len(ua_hash)}")
    message = encode_defunct(primitive=ua_hash)
    return Account.recover_message(message, signature=signature)


def simulate_verifier_signatures(
    ua_hash: bytes, verifier_keys: Iterable[str]
) -> list[bytes]:
    """Sign with each key in order. Returns the list to pass as the
    ``signatures`` argument of ``submitAttestation``."""
    return [sign_ua_hash(ua_hash, k) for k in verifier_keys]
