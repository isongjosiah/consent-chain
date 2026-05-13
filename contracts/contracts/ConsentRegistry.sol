// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC721} from "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import {Ownable, Ownable2Step} from "@openzeppelin/contracts/access/Ownable2Step.sol";

import {IConsentRegistry} from "./interfaces/IConsentRegistry.sol";

/// @dev Minimal subset of FLGatekeeper used by the registry. The full
///      interface is defined alongside FLGatekeeper.sol (see docs/contracts-spec.md §5).
interface IFLGatekeeperMin {
    function currentModelHash() external view returns (bytes32);
    function recordPendingUnlearning(uint256 tokenId) external;
    function unblock(uint256 tokenId) external;
}

/// @title ConsentRegistry — soulbound dynamic-consent NFTs with CLSM semantics.
/// @notice See docs/contracts-spec.md §3 for the authoritative specification.
contract ConsentRegistry is ERC721, Ownable2Step, IConsentRegistry {
    // ---------------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------------

    uint256 public constant MAX_PURPOSE_SET_SIZE = 16;

    // ---------------------------------------------------------------------
    // Storage
    // ---------------------------------------------------------------------

    mapping(uint256 tokenId => ConsentRecord record) private _records;
    uint256 private _nextTokenId; // first minted id is 1; 0 reserved for "unminted"

    address public override oracle;
    address public override gatekeeper;

    // ---------------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------------

    constructor(address initialOwner)
        ERC721("ConsentChain Consent", "CC-CONSENT")
        Ownable(initialOwner)
    {}

    // ---------------------------------------------------------------------
    // One-shot wiring (§3.5)
    // ---------------------------------------------------------------------

    function setOracle(address oracle_) external override onlyOwner {
        if (oracle != address(0)) revert OracleAlreadySet();
        oracle = oracle_;
        emit OracleBound(oracle_);
    }

    function setGatekeeper(address gatekeeper_) external override onlyOwner {
        if (gatekeeper != address(0)) revert GatekeeperAlreadySet();
        gatekeeper = gatekeeper_;
        emit GatekeeperBound(gatekeeper_);
    }

    // ---------------------------------------------------------------------
    // State machine — §3.5 / §3.8
    // ---------------------------------------------------------------------

    function grantConsent(bytes32[] calldata purposes, uint256 ttl)
        external
        override
        returns (uint256 tokenId)
    {
        bytes32 ps = _validatePurposes(purposes);
        if (ttl != 0 && ttl <= block.timestamp) revert TtlInThePast(ttl, block.timestamp);

        unchecked {
            tokenId = ++_nextTokenId;
        }
        _records[tokenId] = ConsentRecord({
            dsOwner: msg.sender,
            state: ConsentState.ACTIVE,
            purposeSet: ps,
            ttl: ttl,
            grantedAt: block.timestamp,
            lastTransitionAt: block.timestamp,
            modelHashAtRevocation: bytes32(0)
        });

        _mint(msg.sender, tokenId);
        emit ConsentGranted(tokenId, msg.sender, ps, ttl);
    }

    function modifyConsent(uint256 tokenId, bytes32[] calldata purposes, uint256 ttl)
        external
        override
    {
        _assertDsOwner(tokenId);
        _assertState(tokenId, ConsentState.ACTIVE);

        bytes32 ps = _validatePurposes(purposes);
        if (ttl != 0 && ttl <= block.timestamp) revert TtlInThePast(ttl, block.timestamp);

        ConsentRecord storage rec = _records[tokenId];
        rec.purposeSet = ps;
        rec.ttl = ttl;
        rec.lastTransitionAt = block.timestamp;
        emit ConsentModified(tokenId, ps, ttl);
    }

    function suspendConsent(uint256 tokenId) external override {
        _assertTokenExists(tokenId);
        address dsOwner = _records[tokenId].dsOwner;
        if (msg.sender != dsOwner && msg.sender != owner()) {
            revert NotTokenOwner(tokenId, msg.sender);
        }
        _assertState(tokenId, ConsentState.ACTIVE);
        _setState(tokenId, ConsentState.SUSPENDED);
        emit ConsentSuspended(tokenId, msg.sender);
    }

    function resumeConsent(uint256 tokenId) external override {
        _assertDsOwner(tokenId);
        _assertState(tokenId, ConsentState.SUSPENDED);
        _setState(tokenId, ConsentState.ACTIVE);
        emit ConsentResumed(tokenId);
    }

    function revokeConsent(uint256 tokenId) external override {
        _assertDsOwner(tokenId);
        _assertState(tokenId, ConsentState.ACTIVE);

        bytes32 h = _captureCurrentModelHash(tokenId);
        ConsentRecord storage rec = _records[tokenId];
        rec.state = ConsentState.REVOKED;
        rec.lastTransitionAt = block.timestamp;
        IFLGatekeeperMin(gatekeeper).recordPendingUnlearning(tokenId);

        emit ConsentRevoked(tokenId, h);
        emit UnlearningRequired(tokenId, h);
    }

    function checkExpiry(uint256 tokenId) external override {
        _assertTokenExists(tokenId);
        ConsentRecord storage rec = _records[tokenId];
        if (rec.state == ConsentState.UNLEARNED) revert TerminalState(tokenId);
        if (rec.state != ConsentState.ACTIVE) {
            revert InvalidStateTransition(rec.state, ConsentState.EXPIRED);
        }
        if (rec.ttl == 0 || rec.ttl > block.timestamp) {
            revert InvalidStateTransition(rec.state, ConsentState.EXPIRED);
        }

        bytes32 h = _captureCurrentModelHash(tokenId);
        rec.state = ConsentState.EXPIRED;
        rec.lastTransitionAt = block.timestamp;
        IFLGatekeeperMin(gatekeeper).recordPendingUnlearning(tokenId);

        emit ConsentExpired(tokenId, h);
        emit UnlearningRequired(tokenId, h);
    }

    function transitionToUnlearned(uint256 tokenId, bytes32 uaHash) external override {
        if (msg.sender != oracle) revert NotAuthorizedOracle(msg.sender);
        _assertTokenExists(tokenId);
        ConsentRecord storage rec = _records[tokenId];
        if (rec.state == ConsentState.UNLEARNED) revert TerminalState(tokenId);
        if (rec.state != ConsentState.REVOKED && rec.state != ConsentState.EXPIRED) {
            revert InvalidStateTransition(rec.state, ConsentState.UNLEARNED);
        }
        rec.state = ConsentState.UNLEARNED;
        rec.lastTransitionAt = block.timestamp;
        // Decrement the gatekeeper's pendingUnlearningCount in the same
        // transaction that retires the token (spec §3.5 / §7). The
        // gatekeeper must be wired by now — revokeConsent / checkExpiry
        // would have failed earlier with GatekeeperNotSet otherwise.
        IFLGatekeeperMin(gatekeeper).unblock(tokenId);
        emit UnlearningCompleted(tokenId, uaHash);
    }

    // ---------------------------------------------------------------------
    // Views — §3.6
    // ---------------------------------------------------------------------

    function recordOf(uint256 tokenId) external view override returns (ConsentRecord memory) {
        _assertTokenExists(tokenId);
        return _records[tokenId];
    }

    function stateOf(uint256 tokenId) external view override returns (ConsentState) {
        _assertTokenExists(tokenId);
        return _records[tokenId].state;
    }

    function modelHashAtRevocation(uint256 tokenId) external view override returns (bytes32) {
        _assertTokenExists(tokenId);
        return _records[tokenId].modelHashAtRevocation;
    }

    function isEligible(
        uint256 tokenId,
        bytes32 roundPurpose,
        bytes32[] calldata purposeArray,
        uint256 index
    ) external view override returns (bool) {
        if (_ownerOf(tokenId) == address(0)) return false;
        ConsentRecord storage rec = _records[tokenId];
        if (rec.state != ConsentState.ACTIVE) return false;
        if (rec.ttl != 0 && rec.ttl <= block.timestamp) return false;
        if (index >= purposeArray.length) return false;
        if (purposeArray[index] != roundPurpose) return false;
        if (keccak256(abi.encode(purposeArray)) != rec.purposeSet) return false;
        return true;
    }

    // ---------------------------------------------------------------------
    // Soulbound enforcement (§3.7)
    // ---------------------------------------------------------------------

    function _update(address to, uint256 tokenId, address auth)
        internal
        override
        returns (address)
    {
        // Allow mint (from == 0). Block transfers AND burns — UNLEARNED is
        // a state, not a deletion (CLAUDE.md §13.2, contracts-spec.md §3.7).
        address from = _ownerOf(tokenId);
        if (from != address(0)) revert SoulboundTransferDisabled();
        return super._update(to, tokenId, auth);
    }

    function approve(address, uint256) public pure override(ERC721) {
        revert SoulboundTransferDisabled();
    }

    function setApprovalForAll(address, bool) public pure override(ERC721) {
        revert SoulboundTransferDisabled();
    }

    // ---------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------

    function _validatePurposes(bytes32[] calldata purposes) internal pure returns (bytes32) {
        uint256 n = purposes.length;
        if (n == 0) revert EmptyPurposeSet();
        if (n > MAX_PURPOSE_SET_SIZE) revert PurposeSetTooLarge(n);
        for (uint256 i = 1; i < n; ++i) {
            // Strict ordering implies no duplicates.
            if (purposes[i] <= purposes[i - 1]) revert PurposesNotSorted();
        }
        return keccak256(abi.encode(purposes));
    }

    function _assertTokenExists(uint256 tokenId) internal view {
        if (_ownerOf(tokenId) == address(0)) revert UnknownToken(tokenId);
    }

    function _assertDsOwner(uint256 tokenId) internal view {
        _assertTokenExists(tokenId);
        if (_records[tokenId].dsOwner != msg.sender) revert NotTokenOwner(tokenId, msg.sender);
    }

    /// @dev Reverts TerminalState first so UNLEARNED is unambiguously absorbing,
    ///      then InvalidStateTransition for any other mismatch.
    function _assertState(uint256 tokenId, ConsentState expected) internal view {
        ConsentState s = _records[tokenId].state;
        if (s == ConsentState.UNLEARNED) revert TerminalState(tokenId);
        if (s != expected) revert InvalidStateTransition(s, expected);
    }

    function _setState(uint256 tokenId, ConsentState next) internal {
        _records[tokenId].state = next;
        _records[tokenId].lastTransitionAt = block.timestamp;
    }

    /// @dev Capture the gatekeeper's current model hash on first
    ///      revoke/expire transition. Write-once: subsequent calls revert.
    function _captureCurrentModelHash(uint256 tokenId) internal returns (bytes32 h) {
        if (gatekeeper == address(0)) revert GatekeeperNotSet();
        ConsentRecord storage rec = _records[tokenId];
        if (rec.modelHashAtRevocation != bytes32(0)) revert ModelHashAlreadyBound(tokenId);
        h = IFLGatekeeperMin(gatekeeper).currentModelHash();
        rec.modelHashAtRevocation = h;
    }
}
