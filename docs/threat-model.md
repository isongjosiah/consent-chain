# `docs/threat-model.md` — Threat Model

**Authoritative** for the adversary set, trust assumptions, and
residual-risk acknowledgments. If this conflicts with `CLAUDE.md §5`,
this file wins.

ConsentChain protects the integrity of consent and the verifiability of
machine unlearning under a specific adversary model. This document
specifies who is trusted, who is adversarial, what each adversarial
party can and cannot do, and where the protocol stops short of formal
guarantees (with the rationale for each acknowledged gap).

---

## 1. System entities

| Symbol | Entity | Role |
|---|---|---|
| **DS_i** | Data Subject _i_ | Holds the private key of a soulbound Consent NFT. The only party that can grant, modify, suspend, resume, or revoke their own consent. In the IoT framing, DS_i may also be the device owner (smart-home resident, fleet operator). |
| **DC** | Data Controller / FL Orchestrator | Operates the federated round loop. Must consult `FLGatekeeper.validateRound` before starting any round and must trigger Unlearning Protocol within `Δ_max` blocks of any `revokeConsent`. |
| **FL_k** | FL participant device _k_ | One per DS in the IoT framing. Runs local SGD over local data; submits a gradient/weight update each round. |
| **VC = {v_1, … v_n}** | Verifier Committee | Five (n=5) independent off-chain entities, each holding one ECDSA key registered with `UnlearningOracle.addVerifier()`. Each VC member independently runs the three MIA attacks against any UA they sign. Threshold t = 3-of-5 default. |
| **Registry / Oracle / Gatekeeper** | On-chain contracts | `ConsentRegistry`, `UnlearningOracle`, `FLGatekeeper` — see `docs/contracts-spec.md`. |
| **Chain** | The PoA² / PoS network | PureChain (primary, gas-free, permissioned validator set) or Polygon Amoy (comparison, gas-based, permissionless). |
| **Auditor** | Anyone | Anyone with read access to the chain. Can monitor every consent state transition, every UA, every ComplianceViolation event. Public auditability is a first-class property. |

---

## 2. Trust assumptions

The protocol's guarantees rest on six assumptions. Each is stated
explicitly so reviewers can challenge whichever they find weakest.

**T1 — DS private key custody.** DS_i's private key is held by DS_i
alone. Compromise of the key is equivalent to identity theft and is
out of scope (standard wallet security). The system provides no
mechanism that compensates for a leaked DS key.

**T2 — Verifier Committee honest-majority.** Strictly more than n − t
verifiers are honest at any given time. With n=5, t=3 the protocol
tolerates two faulty or colluding verifiers. Verifier faults are
detectable by anyone who independently re-runs the MIA on a published
UA — a faulty signature on a UA whose MIA scores are above τ is
public evidence on-chain.

**T3 — Network synchrony.** The DC and DSs can reach the chain within
some bounded latency Δ_block. The compliance-violation timeout
`Δ_max = 1000 blocks` is set well above expected network delays
(roughly 20-30 minutes at typical PoS block times).

**T4 — MIA-based forgetting bound.** A model that passes three
independent MIA tests below τ does not memorise any single revoked
training row to a degree that meaningfully exceeds the false-positive
rate of the MIAs themselves. This is an _empirical_ bound, not a
formal one. §VI Discussion expands on the limits of this assumption
(distribution-shift confounds, weak-signal regimes, etc.).

**T5 — Chain liveness.** The PoA² validator set on PureChain (or the
PoS validator set on Polygon Amoy) produces blocks within bounded
delays. A liveness failure of the chain itself stalls the entire
protocol — but this is the same as for any blockchain-based system.

**T6 — Contract immutability.** Once deployed, the three Solidity
contracts cannot be upgraded. Bugs in the contract code translate
directly to vulnerabilities. The contracts are not behind a proxy.
For the BCCA 2026 prototype this is acceptable; production
deployments should consider a transparent-proxy upgrade pattern with
a long timelock.

---

## 3. Adversary model — in scope

Five adversarial parties, with capabilities, intent, defense, and
residual risk.

### 3.1 Lazy FL Orchestrator (LFO)

**Capability:** Operates the DC. Has full control over which clients
participate in each round, when rounds start, and what model state
gets aggregated. Can silently ignore a revocation event and continue
training the model on the revoked DS's data.

**Intent:** Avoid the cost of unlearning. Continue training as if no
revocation occurred.

**Defense:**
1. `FLGatekeeper.validateRound` reverts whenever
   `pendingUnlearningCount > 0`. The LFO cannot start any new round
   until every outstanding revocation is cleared by a valid UA.
2. The chain enforces `Δ_max`: after that window expires, any auditor
   can call `FLGatekeeper.reportTimeout(tokenId)` and the
   `ComplianceViolation` event becomes part of the immutable on-chain
   record.

**Residual risk:** The LFO could train _offline_ (off the chain) on the
revoked data and never submit the result via `validateRound`. The
protocol does not prevent off-chain misuse — but the resulting model
cannot be deployed in any audited setting because there is no UA on
chain associating the deployed model hash with a valid consent state.
§VI acknowledges this as a "trust-the-controller-or-don't" boundary
that any audit-based system shares.

**Empirical evidence:** §IV.2 demo (`results/blockchain/compliance_violation_demo_purechain.json`)
shows the end-to-end timeout sequence executing on PureChain.

### 3.2 Deceptive Data Controller (DDC)

**Capability:** Submits a UA claiming an unlearning was performed,
without having actually run it (or having run an ineffective unlearning).

**Intent:** Bypass the verification step. Claim compliance without
paying the compute cost.

**Defense:** The three independent MIA attacks plus the verifier
committee. To pass:
- All three MIA scores must be strictly below τ = 5500/10000 = 0.55.
- ≥ t=3 distinct registered verifiers must sign the UA hash.
- The verifiers re-run the MIAs independently; a verifier that
  rubber-stamps a UA whose MIA scores are above τ is publicly
  identifiable (every UA is on-chain with its scores and signatures).

A DDC who controls the DC but not the verifier set cannot fabricate
acceptable scores: the chain stores both the scores _and_ the
recovered signers, and any auditor can re-run the MIA against the
published `modelHashPost` to verify.

**Residual risk:** A DDC who also corrupts ≥ t verifiers can mint a
fraudulent UA. This is the security threshold of any t-of-n protocol;
T2 specifies the bound. Mitigation: enlarge n, raise t, or pivot to
BLS threshold signatures (§VI future work).

**Empirical evidence:** §IV.3 demo (`verifier_threshold_sweep_purechain.json`)
shows that submitting a UA with t-1 signatures reverts on every t∈{2,3,4,5}.

### 3.3 Colluding FL participants (CFP)

**Capability:** Two or more participating devices share information
off-chain. They can cache observed model updates to reconstruct
information about a peer's local data.

**Intent:** Re-identify a revoked DS's contribution from the aggregated
model state, defeating the consent-revocation guarantee.

**Defense:** _None at the consent layer._ ConsentChain enforces that
the model is updated (post-revocation) to no-longer-memorise the
revoked DS's data, but it does not protect against cross-participant
inference attacks operating on cached round outputs.

**Residual risk:** Acknowledged as out-of-scope at the consent layer.
Mitigation belongs to a complementary layer:
- Differential privacy on FL updates (e.g., DP-FedAvg) limits the
  per-round information leak.
- Secure aggregation (e.g., BatchCrypt, SecAgg) prevents individual
  updates from being observable.

Both are orthogonal to ConsentChain and are noted as future-work
extensions in §VI.

### 3.4 Consent Forger (CF)

**Capability:** Possesses control over an EVM-compatible wallet that is
_not_ DS_i's. Wants to grant, revoke, or modify consent on behalf of DS_i.

**Intent:** Grant unwanted consent (e.g., to mint a Consent NFT for a
DS without their knowledge) or revoke a legitimate consent (e.g., to
trigger a forced unlearning).

**Defense:** Soulbound ERC-721 + `dsOwner` enforcement.
1. The Consent NFT cannot be transferred (`_update()` reverts on any
   `from != address(0)` transfer; see `docs/contracts-spec.md`).
2. All consent-state transitions require `msg.sender == dsOwner` (the
   address recorded at `grantConsent` time). A forger's `revokeConsent`
   call against a token they don't own reverts with `NotTokenOwner`.

**Residual risk:** None at the protocol layer. Residual risk equals
the security of DS_i's wallet — same as T1.

### 3.5 Replay Attacker (RA)

**Capability:** Has read access to the chain and observes a previously
submitted UA (with all its signatures). Wants to re-submit the same UA
against a different revocation event or against the same token after
state transitions.

**Intent:** Use an old UA to satisfy a new revocation requirement
without performing new unlearning compute.

**Defense:** Three pieces of state baked into the UA hash:
1. **dsTokenId** — every token has a unique id; replaying against a
   different token fails uaHash recomputation.
2. **modelHashPre** — the UA's pre-hash must equal
   `registry.modelHashAtRevocation(tokenId)`, which is captured
   write-once at the moment of revoke. A second revocation event
   on a different model state captures a different hash.
3. **chainid** — included in the hashed encoding so a UA from
   PureChain cannot replay on Amoy or any other EVM chain.

The combination means: an old UA only re-accepts on the exact
(chain, token, pre-hash) tuple it was minted for. Once the token is
in UNLEARNED state, `transitionToUnlearned` reverts on any second call
(`TerminalState`), closing the loop.

**Residual risk:** None at the protocol layer.

---

## 4. Out of scope

The following are acknowledged adversaries / failure modes that the
protocol does **not** defend against. Each is listed with the rationale
for why it's deferred.

### 4.1 Compromised DS private key

If DS_i's wallet is compromised, the attacker can act as DS_i (revoke
consent, grant new consent under different purposes, etc.). The
protocol assumes T1 (wallet-security baseline) and provides no recovery
mechanism. Future work could add a guardian-based recovery scheme
analogous to social-recovery wallets; out of scope for BCCA 2026.

### 4.2 Honest-but-curious verifier (HBV)

The current ECDSA multi-sig scheme leaks the identity of the t
verifiers who signed each UA. A determined attacker could correlate
verifier signing patterns to infer which verifiers are likely to
collude, or to mount targeted bribery. BLS threshold signatures
(§VI future work) collapse the t signatures to one aggregate, hiding
which specific verifiers signed. Acknowledged as a journal-version
upgrade.

### 4.3 Exact-unlearning formal guarantees

The §IV.3 invariant is `MIA scores < τ`, not "the revoked DS's data has
been provably erased." Approximate unlearning (gradient ascent,
FedEraser) does not formally erase information — it reduces it below
an MIA-detection threshold. CLAUDE.md §5's "approximate unlearning +
MIA threshold τ is a pragmatic bound" matches GDPR Article 17's
"reasonable measures" standard but does not satisfy a stronger
information-theoretic erasure requirement. Naive retraining (Table 1
"Naive Retrain" row) is the exact-unlearning gold standard included
for reference; ConsentChain accepts approximate methods because they
are economically viable at scale.

### 4.4 Side-channel attacks on MIA evaluation

A verifier's MIA evaluation runs off-chain. An adversary with access to
a verifier's machine could observe cache timings, GPU memory patterns,
or other side channels and bias the verifier's reported scores. The
protocol assumes verifiers operate evaluation in trusted environments
(or at minimum, that any two corrupted verifiers cannot influence ≥ t
total signatures). Defenses against side channels are out of scope.

### 4.5 PoA² validator-set compromise

PureChain runs on a permissioned PoA² validator set. If a majority of
PoA² validators are compromised, they can refuse to include
transactions, censor specific addresses, or rewrite recent history.
The protocol treats the chain as trusted at the consensus layer (T5).
For deployments where this assumption is unacceptable, the Polygon
Amoy comparison baseline demonstrates that ConsentChain operates
unchanged on a permissionless PoS chain — at the cost of per-operation
gas (Table 2).

### 4.6 Post-unlearning data reconstruction from public checkpoints

Even after unlearning, an attacker with access to a sequence of public
model checkpoints `M_0, M_1, … M_k` could potentially reconstruct the
delta attributable to a specific DS's contribution by differential
analysis of the checkpoint sequence. This is a research-grade attack
on FL itself, orthogonal to consent enforcement, and well outside the
BCCA 2026 scope.

---

## 5. Decision: which mitigations to ship in the conference prototype

| Mitigation | Conference (BCCA 2026) | Journal extension |
|---|---|---|
| Soulbound NFT + `dsOwner` check | ✅ shipped | — |
| Three-MIA threshold gate | ✅ shipped | — |
| ECDSA t-of-n verifier signatures | ✅ shipped, t=3, n=5 | replace with BLS threshold sigs |
| `Δ_max` compliance timeout | ✅ shipped (default 1000 blocks) | adaptive Δ_max tied to chain congestion |
| chainid binding for replay protection | ✅ shipped | — |
| Differential privacy on FL updates | ❌ out of scope | complementary layer for CFP defense |
| Secure aggregation | ❌ out of scope | complementary layer for CFP defense |
| Wallet recovery for lost DS keys | ❌ out of scope | guardian-based recovery |
| PoA²-native verifier integration | ❌ out of scope | consensus-layer UA validation |
| ZK consent proofs | ❌ out of scope | privacy-preserving eligibility checks |

---

**End of `docs/threat-model.md`.**
