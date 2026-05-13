// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title MockFLGatekeeper — test double used until FLGatekeeper ships on D4.
/// @dev Implements only the surface ConsentRegistry calls:
///      - currentModelHash() view
///      - recordPendingUnlearning(uint256) state-changing
///      Exposes setters and counters so tests can assert behavior.
contract MockFLGatekeeper {
    bytes32 public currentModelHash;
    mapping(uint256 tokenId => uint256 count) public pendingCalls;
    mapping(uint256 tokenId => uint256 count) public unblockCalls;
    uint256 public totalPendingCalls;
    uint256 public totalUnblockCalls;
    uint256 public lastPendingTokenId;
    uint256 public lastUnblockedTokenId;

    event PendingRecorded(uint256 indexed tokenId);
    event Unblocked(uint256 indexed tokenId);

    function setCurrentModelHash(bytes32 h) external {
        currentModelHash = h;
    }

    function recordPendingUnlearning(uint256 tokenId) external {
        pendingCalls[tokenId] += 1;
        totalPendingCalls += 1;
        lastPendingTokenId = tokenId;
        emit PendingRecorded(tokenId);
    }

    function unblock(uint256 tokenId) external {
        unblockCalls[tokenId] += 1;
        totalUnblockCalls += 1;
        lastUnblockedTokenId = tokenId;
        emit Unblocked(tokenId);
    }
}
