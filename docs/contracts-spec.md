# `docs/contracts-spec.md` — On-chain Specifications

**Authoritative** for the three Solidity contracts. If this conflicts with
`CLAUDE.md §4`, this file wins and CLAUDE.md must be updated to match
(per `CLAUDE.md §13.1`).

- Target Solidity: `^0.8.24`
- Dependencies: `@openzeppelin/contracts@5.x`
- EVM: paris-compatible (PureChain + Polygon Amoy)
- Test framework: Hardhat + ethers v6 + chai matchers

Three contracts, deployed in this order, with these constructor dependencies:

```
ConsentRegistry      (no deps)
        ▲
        │ address registry
UnlearningOracle ────┐
                     │
                     ▼
              FLGatekeeper(address registry, address oracle)
```

`UnlearningOracle` needs the registry address; `FLGatekeeper` needs both.
After deployment, the registry's `oracle` reference is bound via
`registry.setOracle(oracle)` (one-shot, owner-only) so revocations can later
trigger the registry → `UnlearningRequired` event → oracle pickup loop, and
the oracle's authorization to call `transitionToUnlearned` is enforced.

---

## 1. Design principles

1. **Custom errors only.** No `require` strings anywhere. (CLAUDE.md §13.2)
2. **Soulbound** consent NFTs implemented by overriding OZ v5
   `ERC721._update` — not the deprecated `_beforeTokenTransfer`.
3. **UNLEARNED is absorbing.** No function may transition out of it. A
   property-based test enforces this for every public function.
4. **Replay protection** for UAs: `uaHash` binds `(chainid, dsTokenId,
   modelHashPre, modelHashPost, method, scores, timestamp)` and is stored
   on-chain; resubmission of the same hash reverts.
5. **No arbitrary calls / no upgradability** in the conference prototype.
   Document the production path (proxy + timelock) in `§VI Discussion`.
6. **Events are the audit trail.** Every state-changing call emits an event
   that names the consent token, the operation, and the actor.

---

## 2. Shared types

### 2.1 Enums

```solidity
enum ConsentState {
    INACTIVE,    // 0  s₀, also the value for an unminted token
    ACTIVE,      // 1
    SUSPENDED,   // 2
    REVOKED,     // 3
    EXPIRED,     // 4
    UNLEARNED    // 5  terminal / absorbing
}

enum UnlearningMethod {
    SISA_FED,        // 0
    GRADIENT_ASCENT, // 1
    FEDERASER        // 2
}
```

### 2.2 Structs

```solidity
struct ConsentRecord {
    address dsOwner;                // data subject; immutable per token
    ConsentState state;
    bytes32 purposeSet;             // hash of an allowed-purpose set; see §2.4
    uint256 ttl;                    // unix timestamp; 0 = no expiry
    uint256 grantedAt;
    uint256 lastTransitionAt;
    bytes32 modelHashAtRevocation;  // set on REVOKED/EXPIRED; consumed by oracle
}

struct UnlearningAttestation {
    uint256 dsTokenId;
    bytes32 modelHashPre;
    bytes32 modelHashPost;
    UnlearningMethod method;
    uint256 shadowMiaScore;         // fixed-point ×1e4 (5000 = 0.5000)
    uint256 liraScore;
    uint256 labelOnlyMiaScore;
    uint256 timestamp;              // assertor's clock; bounded by chain freshness check
    bytes32 uaHash;                 // keccak256 of all fields above, plus block.chainid
}

struct RoundContext {
    uint256 roundId;
    bytes32 roundPurpose;           // single purpose hash; must lie in each DS purposeSet
    bytes32 expectedModelHashPre;   // optional binding to a specific checkpoint; 0x0 = skip
}
```

### 2.3 Constants

```solidity
uint256 internal constant MIA_SCALE = 1e4;          // fixed-point divisor
uint256 internal constant DEFAULT_MIA_THRESHOLD = 5500; // 0.55
uint256 internal constant DEFAULT_DELTA_MAX = 1000;     // blocks
uint256 internal constant MAX_PURPOSE_SET_SIZE = 16;    // purposes per consent
uint256 internal constant MAX_PARTICIPANTS_PER_ROUND = 1024; // hard gas cap
```

### 2.4 Purpose-set hashing

Purposes are short ASCII identifiers (e.g. `"adl-recognition"`,
`"anomaly-detection"`). On grant/modify, the DS supplies an array
`bytes32[] purposes`. The contract stores
`purposeSet = keccak256(abi.encode(sortAscending(purposes)))`.

`validateRound` accepts a single `roundPurpose` and a per-DS membership
proof (`bytes32[] purposeArray` + index). The contract recomputes the
sorted hash and checks `purposeArray[index] == roundPurpose`. This avoids
on-chain enumeration and keeps `validateRound` O(N · log P) where P is
purposes per DS.

---

## 3. `ConsentRegistry.sol`

Soulbound ERC-721 over `ConsentRecord`. One token per `(dsOwner, purposeSet
intent)`; a DS may hold multiple tokens for different deployments.

### 3.1 Inheritance

```solidity
contract ConsentRegistry is ERC721, Ownable2Step
```

`ERC721Enumerable` is **not** used (saves ~30k gas per transfer; we don't
need enumeration on-chain — off-chain indexers walk events).

### 3.2 Custom errors

```solidity
error NotTokenOwner(uint256 tokenId, address caller);
error NotAuthorizedOracle(address caller);
error OracleAlreadySet();
error UnknownToken(uint256 tokenId);
error InvalidStateTransition(ConsentState from, ConsentState to);
error EmptyPurposeSet();
error PurposeSetTooLarge(uint256 size);
error PurposesNotSorted();
error TtlInThePast(uint256 ttl, uint256 nowTs);
error SoulboundTransferDisabled();
error TerminalState(uint256 tokenId);
error ModelHashAlreadyBound(uint256 tokenId);
```

### 3.3 Storage

```solidity
mapping(uint256 => ConsentRecord) private _records;
uint256 private _nextTokenId;
address public oracle;          // address(0) until setOracle
address public gatekeeper;      // address(0) until setGatekeeper (read-only ref)
```

### 3.4 Events

```solidity
event ConsentGranted(
    uint256 indexed tokenId,
    address indexed dsOwner,
    bytes32 indexed purposeSet,
    uint256 ttl
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
```

`UnlearningRequired` is the trigger off-chain UA submitters watch for.

### 3.5 External functions

NatSpec stub for each. Pre/postconditions are normative.

```solidity
/// @notice Mint a new consent NFT.
/// @dev Caller becomes dsOwner; the soulbound owner cannot be changed.
/// @param purposes Sorted ascending array (no duplicates), length 1..MAX_PURPOSE_SET_SIZE.
/// @param ttl Unix expiry; 0 means non-expiring.
/// @return tokenId Newly minted token id.
/// Preconditions:  purposes sorted strictly ascending, |purposes|∈[1, MAX], ttl==0||ttl>now
/// Postconditions: state==ACTIVE, purposeSet=hash(purposes), dsOwner=msg.sender
function grantConsent(bytes32[] calldata purposes, uint256 ttl) external returns (uint256 tokenId);

/// @notice Replace purposeSet and/or TTL of an existing ACTIVE token.
/// @dev Only the dsOwner may call.
/// Preconditions:  current state==ACTIVE, caller==dsOwner, new purposes valid as in grantConsent
function modifyConsent(uint256 tokenId, bytes32[] calldata purposes, uint256 ttl) external;

/// @notice Pause inclusion of this DS without revoking.
/// @dev Owner OR contract owner (regulator) may call.
function suspendConsent(uint256 tokenId) external;

/// @notice Reverse a suspension.
/// @dev Only dsOwner may call.
function resumeConsent(uint256 tokenId) external;

/// @notice Revoke consent. Captures current model hash from FLGatekeeper.
/// @dev Only dsOwner may call. Sets modelHashAtRevocation and emits UnlearningRequired.
/// Postconditions: state==REVOKED, modelHashAtRevocation==gatekeeper.currentModelHash()
function revokeConsent(uint256 tokenId) external;

/// @notice Anyone may flip an ACTIVE token to EXPIRED once TTL has passed.
/// @dev Permissionless to allow the gatekeeper or watchers to fire it.
function checkExpiry(uint256 tokenId) external;

/// @notice Oracle-only: mark this token UNLEARNED. Terminal.
/// @dev Caller must equal `oracle`; reverts if state ∉ {REVOKED, EXPIRED}.
///      Also calls `gatekeeper.unblock(tokenId)` so the gatekeeper's
///      `pendingUnlearningCount` is decremented in the same transaction —
///      mirrors the `recordPendingUnlearning` callback fired from
///      `revokeConsent` / `checkExpiry`.
function transitionToUnlearned(uint256 tokenId, bytes32 uaHash) external;

// --- Owner-only one-shot wiring ---
function setOracle(address oracle_) external onlyOwner;
function setGatekeeper(address gatekeeper_) external onlyOwner;
```

### 3.6 View functions

```solidity
function recordOf(uint256 tokenId) external view returns (ConsentRecord memory);
function stateOf(uint256 tokenId) external view returns (ConsentState);
function modelHashAtRevocation(uint256 tokenId) external view returns (bytes32);

/// @notice Membership test used by FLGatekeeper.
/// @param purposeArray Sorted ascending purposes the caller claims for the token.
/// @param index Position in purposeArray that should equal `roundPurpose`.
/// @return true iff (a) recompute(purposeArray) == record.purposeSet,
///                  (b) purposeArray[index] == roundPurpose,
///                  (c) record.state == ACTIVE,
///                  (d) ttl == 0 || ttl > block.timestamp.
function isEligible(
    uint256 tokenId,
    bytes32 roundPurpose,
    bytes32[] calldata purposeArray,
    uint256 index
) external view returns (bool);
```

### 3.7 Soulbound enforcement

```solidity
function _update(address to, uint256 tokenId, address auth)
    internal
    override
    returns (address)
{
    address from = _ownerOf(tokenId);
    // Allow mint (from == 0). Block all other movement, including burn —
    // UNLEARNED is a state, not a deletion, per CLAUDE.md §13.2.
    if (from != address(0)) {
        revert SoulboundTransferDisabled();
    }
    return super._update(to, tokenId, auth);
}
```

### 3.8 Transition table — normative

| From       | Function                                       | To         | Guard                                                              |
|------------|------------------------------------------------|------------|--------------------------------------------------------------------|
| INACTIVE   | grantConsent                                   | ACTIVE     | purposes valid; ttl future or 0                                    |
| ACTIVE     | modifyConsent                                  | ACTIVE     | caller == dsOwner; new purposes/ttl valid                          |
| ACTIVE     | suspendConsent                                 | SUSPENDED  | caller ∈ {dsOwner, contract owner}                                 |
| ACTIVE     | revokeConsent                                  | REVOKED    | caller == dsOwner; binds modelHashAtRevocation; emits UnlearningRequired |
| ACTIVE     | checkExpiry                                    | EXPIRED    | block.timestamp >= ttl && ttl != 0; binds modelHashAtRevocation    |
| SUSPENDED  | resumeConsent                                  | ACTIVE     | caller == dsOwner                                                  |
| REVOKED    | transitionToUnlearned                          | UNLEARNED  | caller == oracle; tokenId not already UNLEARNED                    |
| EXPIRED    | transitionToUnlearned                          | UNLEARNED  | caller == oracle; tokenId not already UNLEARNED                    |
| UNLEARNED  | *any*                                          | —          | reverts TerminalState                                              |

All other (state, function) pairs MUST revert `InvalidStateTransition`.

---

## 4. `UnlearningOracle.sol`

### 4.1 Inheritance

```solidity
contract UnlearningOracle is Ownable2Step
```

ECDSA verification via `@openzeppelin/contracts/utils/cryptography/ECDSA.sol`
and `MessageHashUtils.toEthSignedMessageHash`.

### 4.2 Storage

```solidity
IConsentRegistry public immutable registry;
uint256 public miaThreshold;       // default DEFAULT_MIA_THRESHOLD
uint256 public verifierThreshold;  // default 3
mapping(address => bool) public isVerifier;
address[] private _verifierList;   // for enumeration in events
mapping(bytes32 => bool) public seenUaHash;
mapping(uint256 => bytes32) public lastUaHash; // dsTokenId → uaHash
```

### 4.3 Custom errors

```solidity
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
error WrongConsentState(ConsentState state);
```

### 4.4 Events

```solidity
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
```

### 4.5 External functions

```solidity
/// @notice Submit an Unlearning Attestation with ≥ verifierThreshold ECDSA signatures.
/// @param ua  Off-chain-assembled attestation. ua.uaHash is recomputed and checked.
/// @param signatures  Length must be ≥ verifierThreshold. Each must recover to a registered,
///                    distinct verifier address.
/// Reverts if:
///   - ua.uaHash != computed hash
///   - seenUaHash[uaHash]
///   - any MIA score >= miaThreshold
///   - ua.modelHashPre != registry.modelHashAtRevocation(ua.dsTokenId)
///   - registry.stateOf(dsTokenId) ∉ {REVOKED, EXPIRED}
///   - fewer than verifierThreshold distinct, registered signers
///   - any signature does not recover to a registered verifier
/// On success:
///   - seenUaHash[uaHash] = true
///   - lastUaHash[dsTokenId] = uaHash
///   - calls registry.transitionToUnlearned(dsTokenId, uaHash)
///   - emits AttestationAccepted
function submitAttestation(
    UnlearningAttestation calldata ua,
    bytes[] calldata signatures
) external;

// --- Verifier management (owner-only) ---
function addVerifier(address v) external onlyOwner;
function removeVerifier(address v) external onlyOwner;

// --- Threshold management (owner-only) ---
function setMiaThreshold(uint256 t) external onlyOwner;            // require 1 ≤ t ≤ MIA_SCALE
function setVerifierThreshold(uint256 t) external onlyOwner;       // require 1 ≤ t ≤ |verifiers|
```

### 4.6 UA-hash computation

```solidity
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
```

`block.chainid` in the preimage prevents cross-chain replay between PureChain
and Polygon Amoy (the same contracts are deployed on both).

### 4.7 Signature verification (sketch)

```solidity
address[] memory seen = new address[](signatures.length);
uint256 distinct;
bytes32 ethHash = MessageHashUtils.toEthSignedMessageHash(ua.uaHash);
for (uint256 i; i < signatures.length; ++i) {
    address signer = ECDSA.recover(ethHash, signatures[i]);
    if (!isVerifier[signer]) revert NotVerifier(signer);
    for (uint256 j; j < distinct; ++j) {
        if (seen[j] == signer) revert DuplicateSigner(signer);
    }
    seen[distinct++] = signer;
}
if (distinct < verifierThreshold) revert InsufficientSignatures(distinct, verifierThreshold);
```

---

## 5. `FLGatekeeper.sol`

The pre-round gate. Maintains the current model hash, validates participant
eligibility per round, and tracks pending-unlearning timeouts that surface
the `ComplianceViolation` event referenced in CLAUDE.md §4.4.

### 5.1 Inheritance

```solidity
contract FLGatekeeper is Ownable2Step
```

### 5.2 Storage

```solidity
IConsentRegistry public immutable registry;
address public immutable oracle;

bytes32 public currentModelHash;
uint256 public currentRoundId;
uint256 public deltaMaxBlocks;      // default DEFAULT_DELTA_MAX

/// dsTokenId → block.number at which UnlearningRequired was observed
mapping(uint256 => uint256) public pendingUnlearningSince;
uint256 public pendingUnlearningCount;
```

### 5.3 Custom errors

```solidity
error NotRegistry();
error NotOracle();
error EmptyParticipants();
error TooManyParticipants(uint256 n);
error LengthMismatch();
error PurposeArrayWrong(uint256 tokenId);
error ParticipantIneligible(uint256 tokenId, ConsentState state);
error PendingUnlearningBlocksRound(uint256 outstanding);
error RoundIdNotMonotonic(uint256 given, uint256 current);
error UnknownPendingToken(uint256 tokenId);
```

### 5.4 Events

```solidity
event ModelHashUpdated(uint256 indexed roundId, bytes32 modelHash);
event RoundValidated(uint256 indexed roundId, uint256 eligibleCount);
event UnlearningPendingRecorded(uint256 indexed tokenId, uint256 sinceBlock);
event UnlearningUnblocked(uint256 indexed tokenId);
event ComplianceViolation(uint256 indexed tokenId, uint256 deadlineBlock);
```

### 5.5 External functions

```solidity
/// @notice Coordinator-only (owner). Validate that every participant has ACTIVE
///         consent for `ctx.roundPurpose` and that no token has a pending
///         unlearning obligation.
/// @param participantTokenIds  Consent NFT ids, one per FL client.
/// @param purposeArrays        Sorted-ascending purpose lists per token.
/// @param purposeIndices       Index into purposeArrays[i] equal to roundPurpose.
/// @return eligible            participantTokenIds filtered to ACTIVE+matching.
function validateRound(
    RoundContext calldata ctx,
    uint256[] calldata participantTokenIds,
    bytes32[][] calldata purposeArrays,
    uint256[] calldata purposeIndices
) external returns (uint256[] memory eligible);

/// @notice Coordinator-only. Commit the model hash produced by this round.
function updateModelHash(uint256 roundId, bytes32 modelHash) external onlyOwner;

/// @notice Called by ConsentRegistry on revoke/expire to record the obligation.
function recordPendingUnlearning(uint256 tokenId) external;     // onlyRegistry

/// @notice Called by ConsentRegistry inside `transitionToUnlearned`
///         so the gatekeeper's pendingUnlearningCount is updated in
///         the same transaction that retires the token. The oracle
///         does not call this directly — keeping the oracle ignorant
///         of the gatekeeper preserves the L3/L4 layering in §4.1.
function unblock(uint256 tokenId) external;                     // onlyRegistry

/// @notice Anyone may emit ComplianceViolation for a token whose pending
///         obligation has aged beyond deltaMaxBlocks. Does NOT change state;
///         it surfaces the audit signal for off-chain monitors.
function reportTimeout(uint256 tokenId) external;

function setDeltaMax(uint256 blocks_) external onlyOwner;
```

### 5.6 `validateRound` semantics

1. Revert `EmptyParticipants` / `TooManyParticipants` if `n` outside
   `[1, MAX_PARTICIPANTS_PER_ROUND]`.
2. Revert `LengthMismatch` unless all three arrays have length `n`.
3. Revert `RoundIdNotMonotonic` unless `ctx.roundId > currentRoundId`.
4. Revert `PendingUnlearningBlocksRound(pendingUnlearningCount)` if any
   outstanding obligation exists. This is the **liveness lever**: as long as
   even one revocation is unresolved, the FL run is blocked. Combined with
   `reportTimeout`, this implements the "degrades to no training, not
   unconsented training" property (CLAUDE.md §4.2 property 2).
5. For each participant, call `registry.isEligible(...)`. Aggregate into
   `eligible[]`. Whether an ineligible token reverts the whole call or is
   skipped is a **design choice** — we **revert**, matching strict
   pre-round gating; the coordinator must rebuild the participant set
   before retrying. Document this in the paper.
6. Update `currentRoundId`, emit `RoundValidated`.

Gas note (the scalability story): `validateRound` is O(N · P_avg) where
`N` = participants and `P_avg` = average purposes per DS. On Polygon Amoy
this generates the linear gas-vs-N curve in Figure 1. On PureChain it is
free; we still measure wall-clock latency to show identical correctness.

---

## 6. Interface contracts (in `contracts/interfaces/`)

`IConsentRegistry`, `IUnlearningOracle`, `IFLGatekeeper` — each exposes the
`external` surface above plus the relevant views. The concrete contracts
implement them; tests and the Python `web3` client consume only the
interfaces.

---

## 7. End-to-end sequence (single revocation)

```
DS                ConsentRegistry           FLGatekeeper           UnlearningOracle           Coordinator
 │ grantConsent      │                          │                       │                         │
 ├──────────────────►│  mint NFT, ACTIVE        │                       │                         │
 │                   │  ConsentGranted          │                       │                         │
 │                   │                          │                       │  (round k pre-flight)   │
 │                   │                          │◄──────────────────────┼──── validateRound(...)──┤
 │                   │  isEligible(...) ←───────┤                       │                         │
 │                   │                          │  eligible[] ─────────►│ run round k             │
 │                   │                          │◄────────────── updateModelHash(k, H_k) ─────────┤
 │ revokeConsent     │                          │                       │                         │
 ├──────────────────►│  REVOKED, bind H_k       │                       │                         │
 │                   ├── recordPendingUnlearning(t) ──►│ +pendingCount  │                         │
 │                   │  UnlearningRequired(t,H_k)  emit                 │                         │
 │                   │                          │                       │◄──── off-chain UA ──────┤
 │                   │                          │                       │  submitAttestation()    │
 │                   │  transitionToUnlearned ◄─┼───────────────────────┤                         │
 │                   │  UNLEARNED state set     │                       │  AttestationAccepted    │
 │                   ├── unblock(t) ───────────►│  pendingCount -= 1    │                         │
 │                   │  UnlearningCompleted emit│                       │                         │
 │                   │                          │  validateRound(k+1) ok│                         │
```

---

## 8. Guard-condition matrix

| Function                                  | Caller         | Reverts (selected)                                          |
|-------------------------------------------|----------------|-------------------------------------------------------------|
| `grantConsent`                            | any            | `EmptyPurposeSet`, `PurposeSetTooLarge`, `PurposesNotSorted`, `TtlInThePast` |
| `modifyConsent`                           | dsOwner        | `NotTokenOwner`, `InvalidStateTransition`                   |
| `suspendConsent`                          | dsOwner/owner  | `NotTokenOwner`, `InvalidStateTransition`                   |
| `resumeConsent`                           | dsOwner        | `NotTokenOwner`, `InvalidStateTransition`                   |
| `revokeConsent`                           | dsOwner        | `NotTokenOwner`, `InvalidStateTransition`, `ModelHashAlreadyBound` |
| `checkExpiry`                             | any            | `InvalidStateTransition` (if state ≠ ACTIVE or ttl in future) |
| `transitionToUnlearned`                   | oracle         | `NotAuthorizedOracle`, `InvalidStateTransition`, `TerminalState` |
| `submitAttestation`                       | any            | `UaReplay`, `MiaThresholdExceeded`, `ModelHashMismatch`, `InsufficientSignatures`, `NotVerifier`, `DuplicateSigner`, `WrongConsentState` |
| `addVerifier` / `removeVerifier`          | owner          | `VerifierAlreadyRegistered`, `VerifierNotRegistered`         |
| `validateRound`                           | owner          | `EmptyParticipants`, `TooManyParticipants`, `LengthMismatch`, `RoundIdNotMonotonic`, `PendingUnlearningBlocksRound`, `ParticipantIneligible`, `PurposeArrayWrong` |
| `updateModelHash`                         | owner          | `RoundIdNotMonotonic`                                       |
| `recordPendingUnlearning`                 | registry       | `NotRegistry`                                               |
| `unblock`                                 | registry       | `NotRegistry`, `UnknownPendingToken`                        |
| `reportTimeout`                           | any            | `UnknownPendingToken`                                       |

---

## 9. Threat-model bindings (cross-ref CLAUDE.md §5)

| Adversary               | Defense surface                                                       |
|-------------------------|-----------------------------------------------------------------------|
| Lazy Orchestrator       | `PendingUnlearningBlocksRound`; `reportTimeout` emits `ComplianceViolation` after `deltaMaxBlocks` |
| Deceptive DC            | `MiaThresholdExceeded`; oracle verifies all three scores < τ; VC re-runs MIA off-chain before signing |
| Consent forger          | `NotTokenOwner`; `_update` blocks transfers; dsOwner set at mint and immutable |
| Replay attacker         | `UaReplay`; `uaHash` binds `block.chainid` and `modelHashPre`; `seenUaHash` mapping is permanent |
| Colluding participants  | Out of scope (acknowledged in paper §VI); no on-chain countermeasure  |

---

## 10. Test matrix → file layout

| File                                  | Coverage                                                                          |
|---------------------------------------|-----------------------------------------------------------------------------------|
| `test/ConsentRegistry.test.ts`        | Every row of §3.8 (happy path + revert). Soulbound: transferFrom / safeTransferFrom revert with `SoulboundTransferDisabled`. Property: from UNLEARNED, every external function reverts. |
| `test/UnlearningOracle.test.ts`       | UA hash recomputation; replay; threshold of signatures; below/above MIA threshold; verifier add/remove; cross-chain hash differs by `chainid`. |
| `test/FLGatekeeper.test.ts`           | validateRound happy path; pending blocks round; timeout emits ComplianceViolation; purposeArray mismatch; monotonic round id. |
| `test/Integration.test.ts`            | Full §7 sequence in one test: grant → round k → revoke → submit UA → unlearned → round k+1 unblocked. |

Coverage target: **≥ 80 % branches** per `CLAUDE.md §13.2`. Enforced in CI.

---

## 11. Open items deferred from this spec

- **BLS threshold signatures** in place of ECDSA multi-sig — journal upgrade,
  not conference scope (CLAUDE.md §11).
- **PoA²-native VC integration** — validators sign UAs as part of block
  production. Requires PureChain consensus hook; out of scope.
- **ZK-SNARK consent eligibility proofs** — hide *who* consented while
  proving membership. Journal scope.
- **Upgradability** — production deployment would put each contract behind
  a transparent proxy + timelock; the conference prototype is non-upgradeable
  and we explicitly state this in §VI.
- **Per-purpose granular revocation** — revoke a subset of purposes without
  full token revocation. Spec sketch: `partialRevoke(tokenId, droppedPurposes)`
  that recomputes purposeSet but does not transition state. Tracked as
  follow-on work; out of scope for conference.

---

## 12. Implementation order (D2–D6)

| Day | Deliverable                                                       |
|-----|-------------------------------------------------------------------|
| D2 (May 14) | `ConsentRegistry.sol` + interface + unit tests (§3, §3.8) |
| D3 (May 15) | `UnlearningOracle.sol` + interface + unit tests (§4)      |
| D4 (May 16) | `FLGatekeeper.sol` + interface + `Integration.test.ts` (§5, §7)   |
| D5 (May 17) | Deploy to PureChain via `scripts/deploy.ts`; smoke-test full §7 sequence on-chain. |
| D6 (May 18) | Deploy to Polygon Amoy; run `scripts/measure-gas.ts`; record `results/blockchain/gas_comparison.json`. |

— End of `contracts-spec.md` —
