"""On-chain consent gate consumed by the FL coordinator.

Bridges ``fl/flower/server.py`` to the three Solidity contracts deployed
in D5/D6: it wraps the read-only views needed during round selection
(``ConsentRegistry.stateOf`` / ``modelHashAtRevocation``,
``FLGatekeeper.pendingUnlearningCount``) and the two state-changing
calls the coordinator must make each round
(``FLGatekeeper.validateRound`` + ``updateModelHash``).

Inputs are loaded from:
  - ``contracts/deployments/{network}.json`` — addresses + verifier set
    (written by ``scripts/deploy.ts``)
  - ``contracts/artifacts/contracts/{Contract}.sol/{Contract}.json`` —
    ABIs produced by ``npx hardhat compile``
  - ``.env`` for ``{NETWORK}_RPC_URL`` and the coordinator's private
    key (``{NETWORK}_DEPLOYER_PRIVATE_KEY`` — the gatekeeper's owner)

D9 will compose this into ``run_casas.py`` so revocations placed via
``revoke_consent`` flow through to the live contracts; D7's baseline
runs offline and does *not* construct a ConsentGate.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError
from eth_account import Account

LOGGER = logging.getLogger("consent_gate")

# Maps the network name (matching contracts/deployments/<name>.json) to the
# .env keys this module reads. Keep in sync with .env.example.
_NETWORK_ENV = {
    "purechain": ("PURECHAIN_RPC_URL", "PURECHAIN_DEPLOYER_PRIVATE_KEY"),
    "amoy": ("AMOY_RPC_URL", "AMOY_DEPLOYER_PRIVATE_KEY"),
    "localhost": ("LOCALHOST_RPC_URL", "LOCALHOST_PRIVATE_KEY"),
}


# Subset of CASAS / N-BaIoT lifecycle states, mirroring IConsentRegistry.
class ConsentState:
    INACTIVE = 0
    ACTIVE = 1
    SUSPENDED = 2
    REVOKED = 3
    EXPIRED = 4
    UNLEARNED = 5


@dataclass(frozen=True)
class RoundContext:
    round_id: int
    round_purpose: bytes      # 32-byte purpose hash
    expected_model_hash_pre: bytes  # 32 bytes; 0x00...00 to skip the binding check


def _project_root() -> Path:
    """Repo root assuming this file is at ``fl/flower/consent_gate.py``."""
    return Path(__file__).resolve().parents[2]


def _load_abi(project_root: Path, contract_name: str) -> list[dict[str, Any]]:
    path = (
        project_root
        / "contracts"
        / "artifacts"
        / "contracts"
        / f"{contract_name}.sol"
        / f"{contract_name}.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"missing compiled artifact: {path}. Run `npx hardhat compile` first."
        )
    return json.loads(path.read_text())["abi"]


def _load_manifest(project_root: Path, network: str) -> dict[str, Any]:
    path = project_root / "contracts" / "deployments" / f"{network}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing deployment manifest: {path}. Run `npm run deploy:{network}` first."
        )
    return json.loads(path.read_text())


class ConsentGate:
    """Coordinator-side façade over the three Solidity contracts."""

    def __init__(
        self,
        network: str,
        *,
        rpc_url: str | None = None,
        signer_key: str | None = None,
        project_root: Path | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self.network = network
        root = project_root or _project_root()
        self.manifest = manifest or _load_manifest(root, network)

        env_rpc_key, env_priv_key = _NETWORK_ENV.get(
            network, (f"{network.upper()}_RPC_URL", f"{network.upper()}_PRIVATE_KEY")
        )
        rpc = rpc_url or os.environ.get(env_rpc_key)
        if not rpc:
            raise ValueError(
                f"no RPC url provided for {network!r} (env {env_rpc_key} unset)"
            )
        self.w3 = Web3(Web3.HTTPProvider(rpc))
        if not self.w3.is_connected():
            raise ConnectionError(f"web3 could not reach {rpc!r}")

        priv = signer_key or os.environ.get(env_priv_key)
        if priv:
            self.account = Account.from_key(priv)
            LOGGER.info("signer: %s", self.account.address)
        else:
            self.account = None
            LOGGER.info("read-only mode (no signer key)")

        # Contract handles
        self.registry: Contract = self.w3.eth.contract(
            address=self.manifest["contracts"]["ConsentRegistry"]["address"],
            abi=_load_abi(root, "ConsentRegistry"),
        )
        self.oracle: Contract = self.w3.eth.contract(
            address=self.manifest["contracts"]["UnlearningOracle"]["address"],
            abi=_load_abi(root, "UnlearningOracle"),
        )
        self.gatekeeper: Contract = self.w3.eth.contract(
            address=self.manifest["contracts"]["FLGatekeeper"]["address"],
            abi=_load_abi(root, "FLGatekeeper"),
        )

    # -- Internal: transaction send + receipt wait ------------------------

    def _send(self, fn_call) -> str:
        if not self.account:
            raise RuntimeError("write tx requires signer_key — re-init ConsentGate with one")
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        tx = fn_call.build_transaction(
            {
                "from": self.account.address,
                "nonce": nonce,
                # PureChain enforces gasPrice=0; Amoy expects a non-zero
                # priority. Let web3 estimate when we can; PureChain
                # overrides at the network layer so this is safe.
                "gas": 500_000,
                "gasPrice": self.w3.eth.gas_price,
            }
        )
        signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
        # web3 ≥ 6.13 renamed rawTransaction → raw_transaction; tolerate either.
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        tx_hash = self.w3.eth.send_raw_transaction(raw)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt["status"] != 1:
            raise ContractLogicError(f"tx reverted: 0x{tx_hash.hex()}")
        return f"0x{tx_hash.hex()}"

    # -- Read-only views --------------------------------------------------

    def state_of(self, token_id: int) -> int:
        return self.registry.functions.stateOf(token_id).call()

    def is_eligible(
        self,
        token_id: int,
        round_purpose: bytes,
        purpose_array: Iterable[bytes],
        purpose_index: int,
    ) -> bool:
        return self.registry.functions.isEligible(
            token_id, round_purpose, list(purpose_array), purpose_index
        ).call()

    def model_hash_at_revocation(self, token_id: int) -> bytes:
        return self.registry.functions.modelHashAtRevocation(token_id).call()

    def pending_unlearning_count(self) -> int:
        return self.gatekeeper.functions.pendingUnlearningCount().call()

    def current_round_id(self) -> int:
        return self.gatekeeper.functions.currentRoundId().call()

    def current_model_hash(self) -> bytes:
        return self.gatekeeper.functions.currentModelHash().call()

    # -- State-changing entry points -------------------------------------

    def update_model_hash(self, round_id: int, model_hash: bytes) -> str:
        return self._send(self.gatekeeper.functions.updateModelHash(round_id, model_hash))

    def validate_round(
        self,
        ctx: RoundContext,
        participant_token_ids: list[int],
        purpose_arrays: list[list[bytes]],
        purpose_indices: list[int],
    ) -> str:
        ctx_tuple = (ctx.round_id, ctx.round_purpose, ctx.expected_model_hash_pre)
        return self._send(
            self.gatekeeper.functions.validateRound(
                ctx_tuple,
                participant_token_ids,
                purpose_arrays,
                purpose_indices,
            )
        )

    # -- DS-side helpers (used by the FL test driver, not the coordinator) -

    def grant_consent(
        self,
        purposes_sorted: list[bytes],
        ttl: int = 0,
        signer_key: str | None = None,
    ) -> tuple[str, int]:
        """Mint a consent NFT for the supplied (already sorted-ascending)
        purpose hashes and return ``(tx_hash, token_id)``.

        ``signer_key`` lets the caller use a DS-specific key instead of
        the gatekeeper coordinator's. If ``None``, falls back to
        ``self.account`` from construction.
        """
        signer = Account.from_key(signer_key) if signer_key else self.account
        if not signer:
            raise RuntimeError("grant_consent requires a signer key")
        fn = self.registry.functions.grantConsent(purposes_sorted, ttl)
        nonce = self.w3.eth.get_transaction_count(signer.address)
        tx = fn.build_transaction(
            {
                "from": signer.address,
                "nonce": nonce,
                "gas": 400_000,
                "gasPrice": self.w3.eth.gas_price,
            }
        )
        signed = self.w3.eth.account.sign_transaction(tx, signer.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        tx_hash = self.w3.eth.send_raw_transaction(raw)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt["status"] != 1:
            raise ContractLogicError(f"grantConsent reverted: 0x{tx_hash.hex()}")
        # Decode the ConsentGranted log to recover the tokenId.
        events = self.registry.events.ConsentGranted().process_receipt(receipt)
        if not events:
            raise RuntimeError("ConsentGranted event not in receipt")
        return f"0x{tx_hash.hex()}", int(events[0]["args"]["tokenId"])

    def revoke_consent(self, token_id: int, signer_key: str | None = None) -> str:
        signer = Account.from_key(signer_key) if signer_key else self.account
        if not signer:
            raise RuntimeError("revoke_consent requires a signer key")
        fn = self.registry.functions.revokeConsent(token_id)
        nonce = self.w3.eth.get_transaction_count(signer.address)
        tx = fn.build_transaction(
            {
                "from": signer.address,
                "nonce": nonce,
                "gas": 250_000,
                "gasPrice": self.w3.eth.gas_price,
            }
        )
        signed = self.w3.eth.account.sign_transaction(tx, signer.key)
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        tx_hash = self.w3.eth.send_raw_transaction(raw)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt["status"] != 1:
            raise ContractLogicError(f"revokeConsent reverted: 0x{tx_hash.hex()}")
        return f"0x{tx_hash.hex()}"


# ---------------------------------------------------------------------------
# Convenience: purpose-hash + sorted-purposes helper
# ---------------------------------------------------------------------------


def purpose_hash(name: str) -> bytes:
    """Match the contract-side keccak256 of UTF-8 encoded purpose names."""
    return Web3.keccak(text=name)


def sort_purposes(purposes: Iterable[bytes]) -> list[bytes]:
    """Strict-ascending byte sort, matching ``_validatePurposes`` in
    ``ConsentRegistry.sol``."""
    return sorted(purposes, key=lambda b: b)
