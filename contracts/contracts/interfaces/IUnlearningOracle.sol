// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IConsentRegistry} from "./IConsentRegistry.sol";

/// @title IUnlearningOracle — verifiable-unlearning attestation gateway.
/// @notice See docs/contracts-spec.md §4 for the authoritative specification.
///         Accepts an Unlearning Attestation (UA) signed by ≥ verifierThreshold
///         distinct registered verifiers, validates MIA-score bounds and the
///         model-hash binding to the consent record, then drives the
///         ConsentRegistry into the UNLEARNED terminal state.
interface IUnlearningOracle {
    // ---------------------------------------------------------------------
    // Types
    // ---------------------------------------------------------------------

    enum UnlearningMethod {
        SISA_FED, // 0
        GRADIENT_ASCENT, // 1
        FEDERASER // 2
    }

    struct UnlearningAttestation {
        uint256 dsTokenId;
        bytes32 modelHashPre;
        bytes32 modelHashPost;
        UnlearningMethod method;
        uint256 shadowMiaScore; // fixed-point ×1e4 (5000 = 0.5000)
        uint256 liraScore;
        uint256 labelOnlyMiaScore;
        uint256 timestamp;
        bytes32 uaHash; // keccak256(abi.encode(chainid, dsTokenId, ..., timestamp))
    }

    // ---------------------------------------------------------------------
    // Errors (§4.3)
    // ---------------------------------------------------------------------

    error NotVerifier(address signer);
    error DuplicateSigner(address signer);
    error InsufficientSignatures(uint256 have, uint256 need);
    error MiaThresholdExceeded(uint256 score, uint256 threshold);
    error UaReplay(bytes32 uaHash);
    error ModelHashMismatch(bytes32 expected, bytes32 got);
    error InvalidVerifierThreshold();
    error InvalidMiaThreshold();
    error VerifierAlreadyRegistered(address v);
    error VerifierNotRegistered(address v);
    error WrongConsentState(IConsentRegistry.ConsentState state);

    // ---------------------------------------------------------------------
    // Events (§4.4)
    // ---------------------------------------------------------------------

    event VerifierAdded(address indexed verifier);
    event VerifierRemoved(address indexed verifier);
    event ThresholdsUpdated(uint256 miaThreshold, uint256 verifierThreshold);
    event AttestationAccepted(
        uint256 indexed dsTokenId,
        bytes32 indexed uaHash,
        UnlearningMethod method,
        uint256 shadowMia,
        uint256 lira,
        uint256 labelOnly
    );

    // ---------------------------------------------------------------------
    // State-changing functions (§4.5)
    // ---------------------------------------------------------------------

    function submitAttestation(
        UnlearningAttestation calldata ua,
        bytes[] calldata signatures
    ) external;

    function addVerifier(address v) external;
    function removeVerifier(address v) external;

    function setMiaThreshold(uint256 t) external;
    function setVerifierThreshold(uint256 t) external;

    // ---------------------------------------------------------------------
    // Views
    // ---------------------------------------------------------------------

    function registry() external view returns (IConsentRegistry);
    function miaThreshold() external view returns (uint256);
    function verifierThreshold() external view returns (uint256);
    function isVerifier(address v) external view returns (bool);
    function verifierCount() external view returns (uint256);
    function verifierAt(uint256 index) external view returns (address);
    function seenUaHash(bytes32 uaHash) external view returns (bool);
    function lastUaHash(uint256 dsTokenId) external view returns (bytes32);
}
