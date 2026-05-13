// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title IConsentRegistry — ConsentChain dynamic-consent state machine.
/// @notice See docs/contracts-spec.md §3 for the authoritative specification.
///         Implementations MUST be soulbound (no transfers) and treat
///         UNLEARNED as an absorbing terminal state.
interface IConsentRegistry {
    // ---------------------------------------------------------------------
    // Types
    // ---------------------------------------------------------------------

    enum ConsentState {
        INACTIVE, // 0 — s₀, also the default for an unminted tokenId
        ACTIVE, // 1
        SUSPENDED, // 2
        REVOKED, // 3
        EXPIRED, // 4
        UNLEARNED // 5 — terminal / absorbing
    }

    struct ConsentRecord {
        address dsOwner;
        ConsentState state;
        bytes32 purposeSet;
        uint256 ttl;
        uint256 grantedAt;
        uint256 lastTransitionAt;
        bytes32 modelHashAtRevocation;
    }

    // ---------------------------------------------------------------------
    // Errors (custom errors only — see CLAUDE.md §13.2)
    // ---------------------------------------------------------------------

    error NotTokenOwner(uint256 tokenId, address caller);
    error NotAuthorizedOracle(address caller);
    error OracleAlreadySet();
    error GatekeeperAlreadySet();
    error GatekeeperNotSet();
    error UnknownToken(uint256 tokenId);
    error InvalidStateTransition(ConsentState from, ConsentState to);
    error EmptyPurposeSet();
    error PurposeSetTooLarge(uint256 size);
    error PurposesNotSorted();
    error TtlInThePast(uint256 ttl, uint256 nowTs);
    error SoulboundTransferDisabled();
    error TerminalState(uint256 tokenId);
    error ModelHashAlreadyBound(uint256 tokenId);

    // ---------------------------------------------------------------------
    // Events (full audit trail — every transition must emit)
    // ---------------------------------------------------------------------

    event ConsentGranted(
        uint256 indexed tokenId, address indexed dsOwner, bytes32 indexed purposeSet, uint256 ttl
    );
    event ConsentModified(uint256 indexed tokenId, bytes32 purposeSet, uint256 ttl);
    event ConsentSuspended(uint256 indexed tokenId, address indexed by);
    event ConsentResumed(uint256 indexed tokenId);
    event ConsentRevoked(uint256 indexed tokenId, bytes32 modelHashAtRevocation);
    event ConsentExpired(uint256 indexed tokenId, bytes32 modelHashAtRevocation);
    event UnlearningRequired(uint256 indexed tokenId, bytes32 modelHashPre);
    event UnlearningCompleted(uint256 indexed tokenId, bytes32 uaHash);
    event OracleBound(address indexed oracle);
    event GatekeeperBound(address indexed gatekeeper);

    // ---------------------------------------------------------------------
    // State-changing functions (§3.5)
    // ---------------------------------------------------------------------

    function grantConsent(bytes32[] calldata purposes, uint256 ttl)
        external
        returns (uint256 tokenId);

    function modifyConsent(uint256 tokenId, bytes32[] calldata purposes, uint256 ttl) external;

    function suspendConsent(uint256 tokenId) external;

    function resumeConsent(uint256 tokenId) external;

    function revokeConsent(uint256 tokenId) external;

    function checkExpiry(uint256 tokenId) external;

    function transitionToUnlearned(uint256 tokenId, bytes32 uaHash) external;

    function setOracle(address oracle_) external;

    function setGatekeeper(address gatekeeper_) external;

    // ---------------------------------------------------------------------
    // Views (§3.6)
    // ---------------------------------------------------------------------

    function recordOf(uint256 tokenId) external view returns (ConsentRecord memory);

    function stateOf(uint256 tokenId) external view returns (ConsentState);

    function modelHashAtRevocation(uint256 tokenId) external view returns (bytes32);

    function isEligible(
        uint256 tokenId,
        bytes32 roundPurpose,
        bytes32[] calldata purposeArray,
        uint256 index
    ) external view returns (bool);

    function oracle() external view returns (address);

    function gatekeeper() external view returns (address);
}
