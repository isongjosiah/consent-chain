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
    uint256 public totalPendingCalls;
    uint256 public lastPendingTokenId;

    event PendingRecorded(uint256 indexed tokenId);

    function setCurrentModelHash(bytes32 h) external {
        currentModelHash = h;
    }

    function recordPendingUnlearning(uint256 tokenId) external {
        pendingCalls[tokenId] += 1;
        totalPendingCalls += 1;
        lastPendingTokenId = tokenId;
        emit PendingRecorded(tokenId);
    }
}
