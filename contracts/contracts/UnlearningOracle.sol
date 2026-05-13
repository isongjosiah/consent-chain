// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable, Ownable2Step} from "@openzeppelin/contracts/access/Ownable2Step.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {MessageHashUtils} from "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";
import {EnumerableSet} from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

import {IConsentRegistry} from "./interfaces/IConsentRegistry.sol";
import {IUnlearningOracle} from "./interfaces/IUnlearningOracle.sol";

/// @title UnlearningOracle — Verifier-Committee-gated UA acceptance.
/// @notice See docs/contracts-spec.md §4 for the authoritative specification.
contract UnlearningOracle is Ownable2Step, IUnlearningOracle {
    using EnumerableSet for EnumerableSet.AddressSet;

    // ---------------------------------------------------------------------
    // Constants (§2.3)
    // ---------------------------------------------------------------------

    uint256 public constant MIA_SCALE = 1e4;
    uint256 public constant DEFAULT_MIA_THRESHOLD = 5500; // 0.55
    uint256 public constant DEFAULT_VERIFIER_THRESHOLD = 3;

    // ---------------------------------------------------------------------
    // Immutable wiring (§4.2)
    // ---------------------------------------------------------------------

    IConsentRegistry public immutable override registry;

    // ---------------------------------------------------------------------
    // Mutable state (§4.2)
    // ---------------------------------------------------------------------

    uint256 public override miaThreshold;
    uint256 public override verifierThreshold;

    EnumerableSet.AddressSet private _verifiers;

    mapping(bytes32 uaHash => bool seen) public override seenUaHash;
    mapping(uint256 dsTokenId => bytes32 uaHash) public override lastUaHash;

    // ---------------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------------

    constructor(IConsentRegistry registry_, address initialOwner) Ownable(initialOwner) {
        registry = registry_;
        miaThreshold = DEFAULT_MIA_THRESHOLD;
        verifierThreshold = DEFAULT_VERIFIER_THRESHOLD;
        emit ThresholdsUpdated(DEFAULT_MIA_THRESHOLD, DEFAULT_VERIFIER_THRESHOLD);
    }

    // ---------------------------------------------------------------------
    // Verifier management (§4.5, owner-only)
    // ---------------------------------------------------------------------

    function addVerifier(address v) external override onlyOwner {
        if (!_verifiers.add(v)) revert VerifierAlreadyRegistered(v);
        emit VerifierAdded(v);
    }

    function removeVerifier(address v) external override onlyOwner {
        if (!_verifiers.remove(v)) revert VerifierNotRegistered(v);
        emit VerifierRemoved(v);
    }

    function setMiaThreshold(uint256 t) external override onlyOwner {
        if (t == 0 || t > MIA_SCALE) revert InvalidMiaThreshold();
        miaThreshold = t;
        emit ThresholdsUpdated(t, verifierThreshold);
    }

    function setVerifierThreshold(uint256 t) external override onlyOwner {
        if (t == 0 || t > _verifiers.length()) revert InvalidVerifierThreshold();
        verifierThreshold = t;
        emit ThresholdsUpdated(miaThreshold, t);
    }

    // ---------------------------------------------------------------------
    // submitAttestation (§4.5–§4.7)
    // ---------------------------------------------------------------------

    function submitAttestation(
        UnlearningAttestation calldata ua,
        bytes[] calldata signatures
    ) external override {
        // 1. Recompute uaHash with chain binding (§4.6) and match the
        //    self-declared hash. Mismatch is treated as a replay attempt.
        bytes32 computed = keccak256(
            abi.encode(
                block.chainid,
                ua.dsTokenId,
                ua.modelHashPre,
                ua.modelHashPost,
                ua.method,
                ua.shadowMiaScore,
                ua.liraScore,
                ua.labelOnlyMiaScore,
                ua.timestamp
            )
        );
        if (computed != ua.uaHash) revert UaReplay(ua.uaHash);

        // 2. Replay protection: a hash may be accepted at most once.
        if (seenUaHash[ua.uaHash]) revert UaReplay(ua.uaHash);

        // 3. All three MIA scores must be strictly below the threshold τ.
        uint256 tau = miaThreshold;
        if (ua.shadowMiaScore >= tau) revert MiaThresholdExceeded(ua.shadowMiaScore, tau);
        if (ua.liraScore >= tau) revert MiaThresholdExceeded(ua.liraScore, tau);
        if (ua.labelOnlyMiaScore >= tau) revert MiaThresholdExceeded(ua.labelOnlyMiaScore, tau);

        // 4. The pre-unlearning model hash must equal the one captured in
        //    the consent record at revoke/expire time. This prevents a UA
        //    produced against one model checkpoint from retiring a token
        //    that was revoked against a different one.
        bytes32 expectedPre = registry.modelHashAtRevocation(ua.dsTokenId);
        if (expectedPre != ua.modelHashPre) {
            revert ModelHashMismatch(expectedPre, ua.modelHashPre);
        }

        // 5. The token must be in REVOKED or EXPIRED — the only two states
        //    from which the registry can transition to UNLEARNED.
        IConsentRegistry.ConsentState s = registry.stateOf(ua.dsTokenId);
        if (
            s != IConsentRegistry.ConsentState.REVOKED
                && s != IConsentRegistry.ConsentState.EXPIRED
        ) revert WrongConsentState(s);

        // 6. Verify ≥ verifierThreshold distinct, registered ECDSA signers.
        bytes32 ethHash = MessageHashUtils.toEthSignedMessageHash(ua.uaHash);
        uint256 nSigs = signatures.length;
        address[] memory seen = new address[](nSigs);
        uint256 distinct;
        for (uint256 i; i < nSigs; ++i) {
            address signer = ECDSA.recover(ethHash, signatures[i]);
            if (!_verifiers.contains(signer)) revert NotVerifier(signer);
            for (uint256 j; j < distinct; ++j) {
                if (seen[j] == signer) revert DuplicateSigner(signer);
            }
            seen[distinct++] = signer;
        }
        if (distinct < verifierThreshold) {
            revert InsufficientSignatures(distinct, verifierThreshold);
        }

        // 7. Commit. State first, then external call into the registry,
        //    then the event — CEI.
        seenUaHash[ua.uaHash] = true;
        lastUaHash[ua.dsTokenId] = ua.uaHash;
        registry.transitionToUnlearned(ua.dsTokenId, ua.uaHash);
        emit AttestationAccepted(
            ua.dsTokenId,
            ua.uaHash,
            ua.method,
            ua.shadowMiaScore,
            ua.liraScore,
            ua.labelOnlyMiaScore
        );
    }

    // ---------------------------------------------------------------------
    // Verifier-set views
    // ---------------------------------------------------------------------

    function isVerifier(address v) external view override returns (bool) {
        return _verifiers.contains(v);
    }

    function verifierCount() external view override returns (uint256) {
        return _verifiers.length();
    }

    function verifierAt(uint256 index) external view override returns (address) {
        return _verifiers.at(index);
    }
}
