"""Off-chain assembly of an Unlearning Attestation (UA).

This is the bridge between the unlearning + MIA scripts and the
on-chain ``UnlearningOracle.submitAttestation`` call: it packages
unlearning + MIA outputs into the struct contracts-spec.md §4 / §2.2
defines, and computes the canonical ``uaHash`` the oracle recomputes
on the way in.

Byte-for-byte matches the Solidity recomputation in
``UnlearningOracle.submitAttestation``:

  bytes32 computed = keccak256(abi.encode(
      block.chainid,    // uint256
      ua.dsTokenId,     // uint256
      ua.modelHashPre,  // bytes32
      ua.modelHashPost, // bytes32
      ua.method,        // uint8
      ua.shadowMiaScore,    // uint256
      ua.liraScore,         // uint256
      ua.labelOnlyMiaScore, // uint256
      ua.timestamp          // uint256
  ));
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Mapping

from eth_abi import encode as abi_encode
from web3 import Web3

from fl.mia.base import MIAResult, MIA_SCALE, to_fixed_point
from fl.unlearning.base import METHOD_ID, UnlearningResult


_UA_SOLIDITY_TYPES: tuple[str, ...] = (
    "uint256",  # chainid
    "uint256",  # dsTokenId
    "bytes32",  # modelHashPre
    "bytes32",  # modelHashPost
    "uint8",    # method
    "uint256",  # shadowMiaScore
    "uint256",  # liraScore
    "uint256",  # labelOnlyMiaScore
    "uint256",  # timestamp
)


@dataclass(frozen=True)
class UnlearningAttestation:
    """Off-chain mirror of the on-chain ``IUnlearningOracle.UnlearningAttestation``
    struct (contracts-spec.md §2.2)."""

    ds_token_id: int
    model_hash_pre: bytes        # 32 bytes
    model_hash_post: bytes       # 32 bytes
    method: int                  # uint8 — UnlearningMethod enum
    shadow_mia_score: int        # ×1e4 fixed-point
    lira_score: int              # ×1e4
    label_only_mia_score: int    # ×1e4
    timestamp: int               # unix seconds
    ua_hash: bytes               # 32 bytes — computed via compute_ua_hash

    def to_contract_tuple(self) -> tuple:
        """Order matching the Solidity struct layout for the contract call."""
        return (
            self.ds_token_id,
            self.model_hash_pre,
            self.model_hash_post,
            self.method,
            self.shadow_mia_score,
            self.lira_score,
            self.label_only_mia_score,
            self.timestamp,
            self.ua_hash,
        )

    def to_json(self) -> str:
        d = asdict(self)
        d["model_hash_pre"] = "0x" + self.model_hash_pre.hex()
        d["model_hash_post"] = "0x" + self.model_hash_post.hex()
        d["ua_hash"] = "0x" + self.ua_hash.hex()
        return json.dumps(d, indent=2, sort_keys=True)


def _to_bytes32(h: bytes | str) -> bytes:
    """Accept ``bytes`` or a ``"0x…"``/raw-hex string; return 32 bytes."""
    if isinstance(h, str):
        if h.startswith("0x") or h.startswith("0X"):
            h = h[2:]
        h = bytes.fromhex(h)
    if len(h) != 32:
        raise ValueError(f"expected 32-byte hash, got {len(h)} bytes")
    return h


def compute_ua_hash(
    chain_id: int,
    ds_token_id: int,
    model_hash_pre: bytes | str,
    model_hash_post: bytes | str,
    method: int,
    shadow_mia_score: int,
    lira_score: int,
    label_only_mia_score: int,
    timestamp: int,
) -> bytes:
    """Canonical ``uaHash`` — must match ``UnlearningOracle.submitAttestation``
    byte-for-byte. Cross-chain replay protection: ``chain_id`` is bound into
    the preimage so the same field set produces different hashes on
    PureChain and Polygon Amoy."""
    pre = _to_bytes32(model_hash_pre)
    post = _to_bytes32(model_hash_post)
    if not 0 <= method <= 255:
        raise ValueError(f"method must fit in uint8: {method}")
    for name, v in (
        ("shadow_mia_score", shadow_mia_score),
        ("lira_score", lira_score),
        ("label_only_mia_score", label_only_mia_score),
    ):
        if not 0 <= v <= MIA_SCALE:
            raise ValueError(f"{name} must be in [0, {MIA_SCALE}]; got {v}")
    encoded = abi_encode(
        list(_UA_SOLIDITY_TYPES),
        [
            chain_id,
            ds_token_id,
            pre,
            post,
            method,
            shadow_mia_score,
            lira_score,
            label_only_mia_score,
            timestamp,
        ],
    )
    return bytes(Web3.keccak(encoded))


def build_ua(
    *,
    chain_id: int,
    ds_token_id: int,
    unlearning_result: UnlearningResult,
    mia_results: Mapping[str, MIAResult],
    timestamp: int | None = None,
) -> UnlearningAttestation:
    """Assemble an attestation from real unlearning + MIA outputs.

    ``mia_results`` must contain entries for ``"shadow_model"``,
    ``"label_only"``, **and** ``"lira"``. If LiRA was dropped per the
    D11 fallback, pass a zeroed-out ``MIAResult`` (score=0) — the
    contract still accepts it because ``0 < miaThreshold`` and the
    paper text discloses the deviation.
    """
    required = {"shadow_model", "label_only", "lira"}
    missing = required - mia_results.keys()
    if missing:
        raise KeyError(f"mia_results missing methods: {sorted(missing)}")

    method_name = unlearning_result.method
    if method_name not in METHOD_ID:
        raise ValueError(f"unknown unlearning method: {method_name!r}")

    ts = int(timestamp if timestamp is not None else time.time())
    shadow = mia_results["shadow_model"].score_fixed_point
    lira = mia_results["lira"].score_fixed_point
    label_only = mia_results["label_only"].score_fixed_point

    ua_hash = compute_ua_hash(
        chain_id=chain_id,
        ds_token_id=ds_token_id,
        model_hash_pre=unlearning_result.model_hash_pre,
        model_hash_post=unlearning_result.model_hash_post,
        method=METHOD_ID[method_name],
        shadow_mia_score=shadow,
        lira_score=lira,
        label_only_mia_score=label_only,
        timestamp=ts,
    )
    return UnlearningAttestation(
        ds_token_id=ds_token_id,
        model_hash_pre=_to_bytes32(unlearning_result.model_hash_pre),
        model_hash_post=_to_bytes32(unlearning_result.model_hash_post),
        method=METHOD_ID[method_name],
        shadow_mia_score=shadow,
        lira_score=lira,
        label_only_mia_score=label_only,
        timestamp=ts,
        ua_hash=ua_hash,
    )
