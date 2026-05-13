// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IConsentRegistry} from "./IConsentRegistry.sol";

/// @title IFLGatekeeper — pre-round consent gate for the FL orchestrator.
/// @notice See docs/contracts-spec.md §5 for the authoritative specification.
///         The coordinator must call `validateRound` before each FL round.
///         Any outstanding unlearning obligation blocks the next round —
///         this is the liveness lever that implements the "degrades to
///         no training, not unconsented training" property.
interface IFLGatekeeper {
    // ---------------------------------------------------------------------
    // Types
    // ---------------------------------------------------------------------

    struct RoundContext {
        uint256 roundId;
        bytes32 roundPurpose;
        bytes32 expectedModelHashPre; // 0x0 = skip the per-round model binding
    }

    // ---------------------------------------------------------------------
    // Errors (§5.3)
    // ---------------------------------------------------------------------

    error NotRegistry();
    error NotOracle();
    error EmptyParticipants();
    error TooManyParticipants(uint256 n);
    error LengthMismatch();
    error PurposeArrayWrong(uint256 tokenId);
    error ParticipantIneligible(uint256 tokenId, IConsentRegistry.ConsentState state);
    error PendingUnlearningBlocksRound(uint256 outstanding);
    error RoundIdNotMonotonic(uint256 given, uint256 current);
    error UnknownPendingToken(uint256 tokenId);
    error ExpectedModelHashMismatch(bytes32 expected, bytes32 got);
    error InvalidDeltaMax();
    error NotYetTimedOut(uint256 deadlineBlock, uint256 currentBlock);

    // ---------------------------------------------------------------------
    // Events (§5.4)
    // ---------------------------------------------------------------------

    event ModelHashUpdated(uint256 indexed roundId, bytes32 modelHash);
    event RoundValidated(uint256 indexed roundId, uint256 eligibleCount);
    event UnlearningPendingRecorded(uint256 indexed tokenId, uint256 sinceBlock);
    event UnlearningUnblocked(uint256 indexed tokenId);
    event ComplianceViolation(uint256 indexed tokenId, uint256 deadlineBlock);
    event DeltaMaxUpdated(uint256 blocks);

    // ---------------------------------------------------------------------
    // State-changing functions (§5.5)
    // ---------------------------------------------------------------------

    function validateRound(
        RoundContext calldata ctx,
        uint256[] calldata participantTokenIds,
        bytes32[][] calldata purposeArrays,
        uint256[] calldata purposeIndices
    ) external returns (uint256[] memory eligible);

    function updateModelHash(uint256 roundId, bytes32 modelHash) external;

    function recordPendingUnlearning(uint256 tokenId) external; // onlyRegistry
    function unblock(uint256 tokenId) external; // onlyRegistry

    function reportTimeout(uint256 tokenId) external; // permissionless

    function setDeltaMax(uint256 blocks_) external; // onlyOwner

    // ---------------------------------------------------------------------
    // Views
    // ---------------------------------------------------------------------

    function registry() external view returns (IConsentRegistry);
    function currentModelHash() external view returns (bytes32);
    function currentRoundId() external view returns (uint256);
    function deltaMaxBlocks() external view returns (uint256);
    function pendingUnlearningSince(uint256 tokenId) external view returns (uint256);
    function pendingUnlearningCount() external view returns (uint256);
}
