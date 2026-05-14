"""E6.E2E — full §7 sequence on the live PureChain contracts.

Drives ConsentGate against the deployed contracts to:

    1. Compute SHA-256 hashes of the pre-unlearn (round_030.pt) and
       post-unlearn (gradient_ascent default) N-BaIoT autoencoder
       checkpoints — these become ``modelHashPre`` / ``modelHashPost``
       in the on-chain UA.
    2. Set gatekeeper.currentModelHash = modelHashPre (so the registry
       captures the right pre-hash when revoke fires).
    3. Mint a consent NFT representing the revoked DS (client 5).
    4. Revoke it.
    5. Build the UA from the real MIA scores recorded in
       results/nbaiot_unlearned/<method>/mia_scores.json — these are
       the exact numbers the off-chain MIA suite produced.
    6. Sign uaHash with 3 of the 5 registered verifier keys.
    7. submitAttestation → triggers registry.transitionToUnlearned and
       gatekeeper.unblock in one transaction.
    8. Verify token state is UNLEARNED on-chain.

Usage::

    python -m fl.experiments.run_onchain_attestation
    python -m fl.experiments.run_onchain_attestation --method sisa_fed
    python -m fl.experiments.run_onchain_attestation --network purechain --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from dotenv import load_dotenv
from eth_account import Account

# Load .env from the repo root so PURECHAIN_RPC_URL / VC_VERIFIER_*_PRIVATE_KEY
# etc. are visible to the on-chain plumbing. Hardhat handles this on its TS
# side; this is the Python equivalent.
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

from fl.attestation.generate_ua import build_ua
from fl.attestation.submit_ua import submit_attestation
from fl.attestation.verify_ua import simulate_verifier_signatures
from fl.flower.consent_gate import ConsentGate, purpose_hash, sort_purposes
from fl.mia.base import MIAResult
from fl.unlearning.base import UnlearningResult
from fl.unlearning.model_hash import hash_state_dict

LOGGER = logging.getLogger("run_onchain_attestation")


def _hash_checkpoint(path: Path) -> str:
    sd = torch.load(path, map_location="cpu")
    return hash_state_dict(sd)


def _load_unlearning_result(method_dir: Path, pre_hash: str, post_hash: str) -> UnlearningResult:
    """Reconstruct an UnlearningResult from the saved JSON.

    The ``post_unlearn_model`` field is set to ``None`` — build_ua only
    reads the hashes and the method name."""
    info = json.loads((method_dir / "unlearning_result.json").read_text())
    return UnlearningResult(
        method=info["method"],
        method_id=info["method_id"],
        post_unlearn_model=None,  # type: ignore[arg-type]
        model_hash_pre=pre_hash,
        model_hash_post=post_hash,
        wall_seconds=float(info["wall_seconds"]),
        details=info.get("details", {}),
    )


def _load_mia_results(method_dir: Path) -> dict[str, MIAResult]:
    raw = json.loads((method_dir / "mia_scores.json").read_text())
    return {
        name: MIAResult(
            method=v["method"],
            score=float(v["score"]),
            score_fixed_point=int(v["score_fixed_point"]),
            wall_seconds=float(v.get("wall_seconds", 0.0)),
            details=v.get("details", {}),
        )
        for name, v in raw.items()
    }


def _gather_verifier_keys(threshold: int) -> list[str]:
    keys: list[str] = []
    for i in range(1, 6):
        k = os.environ.get(f"VC_VERIFIER_{i}_PRIVATE_KEY")
        if k:
            keys.append(k)
    if len(keys) < threshold:
        raise SystemExit(
            f"need ≥{threshold} verifier keys in env (have {len(keys)}). "
            f"Set VC_VERIFIER_1..5_PRIVATE_KEY in .env."
        )
    return keys[:threshold]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="End-to-end on-chain attestation.")
    p.add_argument("--network", default="purechain",
                   choices=["purechain", "amoy", "localhost"])
    p.add_argument("--config", type=Path, default=Path("fl/config/nbaiot.yaml"))
    p.add_argument("--method", default="gradient_ascent",
                   choices=["gradient_ascent", "sisa_fed", "federaser"],
                   help="Which post-unlearn model to attest.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every step but skip on-chain writes.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(args.config.read_text())

    # --- Compute model hashes --------------------------------------------
    round_n = int(cfg["federation"]["revocation"]["revoke_at_round"])
    pre_ckpt = Path(cfg["logging"]["results_dir"]) / "model_checkpoints" / f"round_{round_n:03d}.pt"
    post_ckpt = Path("results/nbaiot_unlearned") / args.method / "post_unlearn_model.pt"
    if not pre_ckpt.exists():
        raise SystemExit(f"missing pre-unlearn checkpoint: {pre_ckpt}")
    if not post_ckpt.exists():
        raise SystemExit(f"missing post-unlearn checkpoint: {post_ckpt}")

    pre_hash = _hash_checkpoint(pre_ckpt)
    post_hash = _hash_checkpoint(post_ckpt)
    LOGGER.info("pre-unlearn  hash: %s  (%s)", pre_hash, pre_ckpt.name)
    LOGGER.info("post-unlearn hash: %s  (%s)", post_hash, post_ckpt.name)

    # --- Load MIA scores + UnlearningResult -----------------------------
    method_dir = Path("results/nbaiot_unlearned") / args.method
    ur = _load_unlearning_result(method_dir, pre_hash, post_hash)
    mia_results = _load_mia_results(method_dir)
    LOGGER.info(
        "MIA scores (×1e4): shadow=%d  lira=%d  label_only=%d",
        mia_results["shadow_model"].score_fixed_point,
        mia_results["lira"].score_fixed_point,
        mia_results["label_only"].score_fixed_point,
    )

    # --- Open the on-chain gate -----------------------------------------
    gate = ConsentGate(args.network)
    chain_id = int(gate.w3.eth.chain_id)
    threshold = int(gate.oracle.functions.verifierThreshold().call())
    mia_threshold = int(gate.oracle.functions.miaThreshold().call())
    LOGGER.info(
        "connected to %s chainid=%d miaThreshold=%d verifierThreshold=%d",
        args.network, chain_id, mia_threshold, threshold,
    )
    if not gate.account:
        raise SystemExit("no signer key configured; cannot drive on-chain writes")

    # --- 1. Set gatekeeper.currentModelHash to pre-unlearn hash --------
    current_round = int(gate.gatekeeper.functions.currentRoundId().call())
    current_hash = bytes(gate.gatekeeper.functions.currentModelHash().call())
    LOGGER.info(
        "gatekeeper state: currentRoundId=%d currentModelHash=0x%s",
        current_round, current_hash.hex(),
    )
    pre_hash_bytes = bytes.fromhex(pre_hash[2:])
    if current_hash != pre_hash_bytes:
        if args.dry_run:
            LOGGER.info("[dry-run] would call updateModelHash(%d, %s)",
                        current_round, pre_hash)
        else:
            tx = gate.update_model_hash(current_round, pre_hash_bytes)
            LOGGER.info("updateModelHash tx: %s", tx)

    # --- 2. Mint a fresh consent token (DS = the deployer account) ----
    purposes = sort_purposes([
        purpose_hash("anomaly-detection"),
        purpose_hash("research"),
    ])
    deployer_key = os.environ.get("PURECHAIN_DEPLOYER_PRIVATE_KEY")
    if not deployer_key:
        raise SystemExit("PURECHAIN_DEPLOYER_PRIVATE_KEY required")
    ds_address = Account.from_key(deployer_key).address
    LOGGER.info("data subject address: %s", ds_address)

    if args.dry_run:
        LOGGER.info("[dry-run] would call grantConsent(purposes=%s, ttl=0)",
                    [p.hex() for p in purposes])
        token_id = 9999  # placeholder for dry-run
    else:
        tx_hash, token_id = gate.grant_consent(purposes, ttl=0, signer_key=deployer_key)
        LOGGER.info("grantConsent tx: %s  tokenId=%d", tx_hash, token_id)

    # --- 3. Revoke (registry captures modelHashPre into modelHashAtRevocation) -
    if args.dry_run:
        LOGGER.info("[dry-run] would call revokeConsent(tokenId=%d)", token_id)
    else:
        rev_tx = gate.revoke_consent(token_id, signer_key=deployer_key)
        LOGGER.info("revokeConsent tx: %s", rev_tx)

        # Sanity-check the registry captured our hash.
        captured = bytes(gate.model_hash_at_revocation(token_id))
        LOGGER.info("registry.modelHashAtRevocation = 0x%s", captured.hex())
        if captured != pre_hash_bytes:
            raise SystemExit(
                f"modelHashAtRevocation mismatch: expected 0x{pre_hash_bytes.hex()}, "
                f"got 0x{captured.hex()}"
            )

    # --- 4. Build UA + sign + submit ------------------------------------
    ua = build_ua(
        chain_id=chain_id,
        ds_token_id=token_id,
        unlearning_result=ur,
        mia_results=mia_results,
        timestamp=int(time.time()),
    )
    LOGGER.info("uaHash: 0x%s", ua.ua_hash.hex())
    LOGGER.info("UA struct: dsTokenId=%d method=%d shadow=%d lira=%d labelOnly=%d ts=%d",
                ua.ds_token_id, ua.method,
                ua.shadow_mia_score, ua.lira_score, ua.label_only_mia_score, ua.timestamp)

    verifier_keys = _gather_verifier_keys(threshold)
    sigs = simulate_verifier_signatures(ua.ua_hash, verifier_keys)
    LOGGER.info("collected %d/%d verifier signatures (threshold=%d)",
                len(sigs), 5, threshold)

    if args.dry_run:
        LOGGER.info("[dry-run] would call submitAttestation(ua, signatures[])")
    else:
        tx = submit_attestation(gate, ua, sigs)
        LOGGER.info("submitAttestation tx: %s", tx)

    # --- 5. Verify the consent token state ------------------------------
    if not args.dry_run:
        final_state = int(gate.state_of(token_id))
        LOGGER.info("final ConsentRegistry.stateOf(%d) = %d (5 = UNLEARNED)",
                    token_id, final_state)
        if final_state != 5:
            raise SystemExit(
                f"unexpected final state {final_state}; expected UNLEARNED (5)"
            )
        LOGGER.info("✓ end-to-end on-chain attestation accepted")

    # --- 6. Persist artefacts -------------------------------------------
    out_dir = method_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    artefact = {
        "network": args.network,
        "chain_id": chain_id,
        "method": args.method,
        "ds_token_id": int(token_id),
        "ds_address": ds_address,
        "model_hash_pre": pre_hash,
        "model_hash_post": post_hash,
        "mia_scores_fixed_point": {
            name: mia_results[name].score_fixed_point
            for name in ("shadow_model", "lira", "label_only")
        },
        "ua_hash": "0x" + ua.ua_hash.hex(),
        "ua_timestamp": ua.timestamp,
        "verifier_threshold": threshold,
        "signatures": [sig.hex() for sig in sigs],
        "dry_run": args.dry_run,
    }
    (out_dir / "onchain_attestation.json").write_text(
        json.dumps(artefact, indent=2) + "\n"
    )
    LOGGER.info("artefact written: %s", out_dir / "onchain_attestation.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
