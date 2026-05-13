// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Ownable, Ownable2Step} from "@openzeppelin/contracts/access/Ownable2Step.sol";

import {IConsentRegistry} from "./interfaces/IConsentRegistry.sol";
import {IFLGatekeeper} from "./interfaces/IFLGatekeeper.sol";

/// @title FLGatekeeper — pre-round consent gate.
/// @notice See docs/contracts-spec.md §5 for the authoritative specification.
contract FLGatekeeper is Ownable2Step, IFLGatekeeper {
    // ---------------------------------------------------------------------
    // Constants (§2.3)
    // ---------------------------------------------------------------------

    uint256 public constant DEFAULT_DELTA_MAX = 1000; // blocks
    uint256 public constant MAX_PARTICIPANTS_PER_ROUND = 1024;

    // ---------------------------------------------------------------------
    // Immutable wiring (§5.2)
    // ---------------------------------------------------------------------

    IConsentRegistry public immutable override registry;

    // ---------------------------------------------------------------------
    // Mutable state (§5.2)
    // ---------------------------------------------------------------------

    bytes32 public override currentModelHash;
    uint256 public override currentRoundId;
    uint256 public override deltaMaxBlocks;

    mapping(uint256 tokenId => uint256 sinceBlock) public override pendingUnlearningSince;
    uint256 public override pendingUnlearningCount;

    // ---------------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------------

    constructor(IConsentRegistry registry_, address initialOwner) Ownable(initialOwner) {
        registry = registry_;
        deltaMaxBlocks = DEFAULT_DELTA_MAX;
        emit DeltaMaxUpdated(DEFAULT_DELTA_MAX);
    }

    // ---------------------------------------------------------------------
    // Authorization modifiers
    // ---------------------------------------------------------------------

    modifier onlyRegistry() {
        if (msg.sender != address(registry)) revert NotRegistry();
        _;
    }

    // ---------------------------------------------------------------------
    // Round validation (§5.5–§5.6)
    // ---------------------------------------------------------------------

    function validateRound(
        RoundContext calldata ctx,
        uint256[] calldata participantTokenIds,
        bytes32[][] calldata purposeArrays,
        uint256[] calldata purposeIndices
    ) external override onlyOwner returns (uint256[] memory eligible) {
        uint256 n = participantTokenIds.length;
        if (n == 0) revert EmptyParticipants();
        if (n > MAX_PARTICIPANTS_PER_ROUND) revert TooManyParticipants(n);
        if (purposeArrays.length != n || purposeIndices.length != n) revert LengthMismatch();

        if (ctx.roundId <= currentRoundId) revert RoundIdNotMonotonic(ctx.roundId, currentRoundId);

        // Liveness lever (§4.2 property 2 / §5.6 step 4): any outstanding
        // unlearning obligation blocks all rounds. The system degrades to
        // "no training" rather than "unconsented training".
        if (pendingUnlearningCount != 0) {
            revert PendingUnlearningBlocksRound(pendingUnlearningCount);
        }

        // Optional binding: caller can require we're operating on top of a
        // specific checkpoint. 0x0 disables the check.
        if (ctx.expectedModelHashPre != bytes32(0) && ctx.expectedModelHashPre != currentModelHash) {
            revert ExpectedModelHashMismatch(ctx.expectedModelHashPre, currentModelHash);
        }

        eligible = new uint256[](n);
        IConsentRegistry reg = registry;
        for (uint256 i; i < n; ++i) {
            uint256 tokenId = participantTokenIds[i];
            bytes32[] calldata purposes = purposeArrays[i];
            uint256 index = purposeIndices[i];
            if (!reg.isEligible(tokenId, ctx.roundPurpose, purposes, index)) {
                // Surface the state for debugging; if the token is unknown,
                // registry.stateOf reverts UnknownToken on its own.
                revert ParticipantIneligible(tokenId, reg.stateOf(tokenId));
            }
            eligible[i] = tokenId;
        }

        currentRoundId = ctx.roundId;
        emit RoundValidated(ctx.roundId, n);
    }

    /// @notice Commit the model hash produced by the round currently in
    ///         flight. Must be called for the round most recently advanced
    ///         to by validateRound; cannot skip ahead and cannot rewrite.
    function updateModelHash(uint256 roundId, bytes32 modelHash) external override onlyOwner {
        if (roundId != currentRoundId) revert RoundIdNotMonotonic(roundId, currentRoundId);
        currentModelHash = modelHash;
        emit ModelHashUpdated(roundId, modelHash);
    }

    // ---------------------------------------------------------------------
    // Pending-unlearning lifecycle (registry-driven)
    // ---------------------------------------------------------------------

    function recordPendingUnlearning(uint256 tokenId) external override onlyRegistry {
        // Defense in depth: don't double-count if registry somehow
        // re-records the same token. The registry's state-machine guards
        // already prevent this (revoke requires ACTIVE, expire requires
        // ACTIVE), but a buggy upgrade should not corrupt the counter.
        if (pendingUnlearningSince[tokenId] == 0) {
            pendingUnlearningSince[tokenId] = block.number;
            unchecked {
                pendingUnlearningCount += 1;
            }
            emit UnlearningPendingRecorded(tokenId, block.number);
        }
    }

    function unblock(uint256 tokenId) external override onlyRegistry {
        if (pendingUnlearningSince[tokenId] == 0) revert UnknownPendingToken(tokenId);
        pendingUnlearningSince[tokenId] = 0;
        unchecked {
            pendingUnlearningCount -= 1;
        }
        emit UnlearningUnblocked(tokenId);
    }

    // ---------------------------------------------------------------------
    // Timeout reporting (permissionless audit signal)
    // ---------------------------------------------------------------------

    function reportTimeout(uint256 tokenId) external override {
        uint256 since = pendingUnlearningSince[tokenId];
        if (since == 0) revert UnknownPendingToken(tokenId);
        uint256 deadline = since + deltaMaxBlocks;
        if (block.number <= deadline) revert NotYetTimedOut(deadline, block.number);
        emit ComplianceViolation(tokenId, deadline);
    }

    // ---------------------------------------------------------------------
    // Configuration
    // ---------------------------------------------------------------------

    function setDeltaMax(uint256 blocks_) external override onlyOwner {
        if (blocks_ == 0) revert InvalidDeltaMax();
        deltaMaxBlocks = blocks_;
        emit DeltaMaxUpdated(blocks_);
    }
}
