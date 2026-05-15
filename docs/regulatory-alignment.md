# `docs/regulatory-alignment.md` — Regulatory Alignment

**Authoritative** for the mapping between ConsentChain mechanisms and
the regulatory obligations they satisfy. If this conflicts with
`CLAUDE.md §6 (limitations)` or §VI of the paper, this file wins.

ConsentChain is designed against three regulatory regimes:

- **GDPR** — Regulation (EU) 2016/679 (General Data Protection Regulation).
- **EU AI Act** — Regulation (EU) 2024/1689 (effective August 2026; some
  high-risk-AI provisions phase in earlier).
- **CCPA / CPRA** — Cal. Civ. Code §§ 1798.100 et seq. (California
  Consumer Privacy Act, as amended by the California Privacy Rights Act).

Each section below states the regulation's requirement in plain
language, the precise article citation, the ConsentChain mechanism
that addresses it, and the on-chain or off-chain evidence a regulator
would inspect to verify compliance. Where the alignment is partial
("reasonable measures" rather than "complete erasure"), this document
flags the gap and points to §VI of the paper.

---

## 1. GDPR Article 7 — Conditions for consent

> _"Where processing is based on consent, the controller shall be able
> to demonstrate that the data subject has consented to processing of
> his or her personal data."_ — GDPR Art. 7(1)
>
> _"The data subject shall have the right to withdraw his or her
> consent at any time. The withdrawal of consent shall not affect the
> lawfulness of processing based on consent before its withdrawal."_
> — GDPR Art. 7(3)

| Requirement | ConsentChain mechanism | Evidence |
|---|---|---|
| Consent must be demonstrably given by the data subject. | `ConsentRegistry.grantConsent(purposes, ttl)` requires `msg.sender == dsOwner` and is recorded on-chain with the DS's address as `dsOwner`, the purpose set, and the TTL. | `ConsentGranted(tokenId, dsOwner, purposeSet, ttl)` event; `stateOf(tokenId) == ACTIVE`. |
| Consent must be granular per processing purpose. | The Consent NFT carries a `bytes32[] purposes` array. Each FL round binds to a single `roundPurpose`; rounds whose purpose isn't in the DS's purpose set are blocked at `validateRound`. | The `purposeSet` hash field in `ConsentRecord` + the `isEligible(tokenId, roundPurpose, purposeArray, index)` check. |
| Withdrawal must be as easy as giving consent. | `revokeConsent(tokenId)` is a one-call transaction by `dsOwner`. No counter-signature required. | `ConsentRevoked` event + immediate state transition to `REVOKED`. |
| Processing prior to withdrawal remains lawful. | The protocol does not retroactively invalidate model versions trained before revocation. The captured `modelHashAtRevocation` marks the dividing line between "lawful past" and "must-unlearn future." | `ConsentRegistry.modelHashAtRevocation(tokenId)` returns the hash of the model state at revocation. |

**Empirical evidence:** Every `grantConsent` / `revokeConsent` flow is
exercised end-to-end in `results/nbaiot_unlearned/<method>/onchain_attestation.json`.

---

## 2. GDPR Article 17 — Right to erasure ("right to be forgotten")

> _"The data subject shall have the right to obtain from the
> controller the erasure of personal data concerning him or her
> without undue delay …"_ — GDPR Art. 17(1)
>
> _"… the controller, taking account of available technology and the
> cost of implementation, shall take reasonable steps … to inform
> controllers which are processing the personal data that the data
> subject has requested the erasure …"_ — GDPR Art. 17(2)

This is the article ConsentChain centrally targets. The "right to be
forgotten" applied to a trained model means the model itself must
"forget" the revoked data subject's contribution.

| Requirement | ConsentChain mechanism | Evidence |
|---|---|---|
| Erasure on request, without undue delay. | `Δ_max = 1000 blocks` (configurable) defines the maximum window between revocation and either a valid UA or a `ComplianceViolation` event. At Polygon's ~2 s block time, this is ≈ 33 minutes — well within "undue delay" interpretations from EU DPAs (which range from days to weeks). | `FLGatekeeper.pendingUnlearningSince[tokenId]` + `Δ_max` + the `ComplianceViolation` event. §IV.2 demo: `results/blockchain/compliance_violation_demo_purechain.json`. |
| The erasure must be _demonstrable_. | The Unlearning Attestation captures `modelHashPre`, `modelHashPost`, three independent MIA scores, and 3-of-5 verifier signatures. Every UA is permanently on-chain with its accept/reject status. | `AttestationAccepted(tokenId, uaHash)` event; the artefact `onchain_attestation.json` in each method directory. |
| "Reasonable steps … taking account of available technology and the cost of implementation." | ConsentChain combines approximate unlearning (cheap) with MIA verification (cheap and re-runnable by anyone). Approximate unlearning is what GDPR Art. 17 actually requires for FL deployments — exact retraining is economically impractical at IoT scale. | §V Table 1 (MIA scores); §V Table 2 (gas comparison — IoT-scale cost projection). |

**Acknowledged gap:** GDPR Art. 17 does not numerically specify
"erasure quality." The MIA-threshold approach in ConsentChain is an
empirical bound that may not satisfy a stricter information-theoretic
erasure requirement. §VI Discussion explains why this is the right
trade-off for IoT-scale FL: exact retraining is infeasible, and
approximate methods + verifiable MIA evidence offer the best currently
available "reasonable measure."

**Empirical evidence:** §IV.2 demo + the on-chain UA submissions for
the three approximate methods that pass the τ-gate (gradient_ascent,
sisa_fed, federaser in `results/nbaiot_unlearned/<method>/`) plus
the CASAS sisa_fed UA at `results/casas_unlearned/sisa_fed/`.
Naive retraining never produces an on-chain UA: its shadow MIA on
the revoked client's distribution exceeds τ even though the model
never saw that client's data, so the gate (correctly) rejects the
submission — a distribution-shift confound discussed in §VI.

---

## 3. GDPR Article 21 — Right to object

> _"The data subject shall have the right to object, on grounds
> relating to his or her particular situation, at any time to
> processing of personal data concerning him or her …"_ — GDPR Art. 21(1)

Article 21 differs from Article 17 in that the DS may want to _pause_
processing without permanently erasing the data. ConsentChain
supports both paths.

| Requirement | ConsentChain mechanism | Evidence |
|---|---|---|
| The DS may pause processing without erasing. | `suspendConsent(tokenId)` transitions the token from `ACTIVE` to `SUSPENDED`. While suspended, `validateRound` excludes the token from the eligible set (`isEligible` returns false on non-ACTIVE state). No UA is required — the model stays as-is. | `ConsentSuspended` event; `stateOf(tokenId) == SUSPENDED`. |
| Suspension is reversible. | `resumeConsent(tokenId)` transitions back to `ACTIVE`. | `ConsentResumed` event. |
| Suspension does not require unlearning. | `pendingUnlearningCount` is unaffected by suspend/resume; only revoke triggers the unlearning obligation. | The recordPendingUnlearning hook in the registry fires only on REVOKED / EXPIRED transitions. |

---

## 4. EU AI Act Article 15 — Accuracy, robustness and cybersecurity

> _"High-risk AI systems shall be designed and developed in such a way
> that they achieve, in the light of their intended purpose, an
> appropriate level of accuracy, robustness and cybersecurity, and
> perform consistently in those respects throughout their lifecycle."_
> — EU AI Act Art. 15(1)

For FL systems deployed in regulated domains (healthcare, hiring,
critical infrastructure), Article 15 demands lifecycle robustness
evidence. ConsentChain contributes to this in two ways.

| Requirement | ConsentChain mechanism | Evidence |
|---|---|---|
| Lifecycle robustness — model state at any point must be auditable. | Every FL round commits its `modelHash` to `FLGatekeeper.currentModelHash` via `updateModelHash`. The chain is the source of truth for "which model hash was canonical at block X." | The full E7/E8 gas measurements include `updateModelHash` as a normal round operation; `results/nbaiot/metrics.json` records the per-round hash. |
| Cybersecurity — tampering with the consent or unlearning state must be detectable. | Soulbound NFTs prevent transfer of consent records; UA submission requires t-of-n verifier signatures; replay protection binds UAs to (chain, token, modelHashPre). | §IV.3 sweep (verifier_threshold_sweep) + the threat model §3.4 (Consent Forger) + §3.5 (Replay Attacker). |
| Continuous performance monitoring. | The `ComplianceViolation` event is the on-chain signal that the system is operating in degraded mode. Standard monitoring tools (subgraph + Webhook) can pick this up and page the operator. | §IV.2 demo. |

**Note on Art. 15 timing:** the high-risk AI provisions of the EU AI
Act enter into force in August 2026. ConsentChain is positioned for
deployments that need to be compliant by that date.

---

## 5. EU AI Act Article 26 — Obligations of deployers

> _"Deployers of high-risk AI systems shall take appropriate technical
> and organisational measures to ensure they use such systems in
> accordance with the instructions for use accompanying the systems …"_
> — EU AI Act Art. 26(1)
>
> _"Deployers shall monitor the operation of the high-risk AI system
> on the basis of the instructions for use and, when relevant, inform
> the providers …"_ — Art. 26(5)

| Requirement | ConsentChain mechanism | Evidence |
|---|---|---|
| Documented technical measures around consent. | The three contracts (with NatSpec in `docs/contracts-spec.md`), the formal CLSM definition, the VUP protocol, and this regulatory-alignment doc collectively form the "technical measures" the deployer must document. | The repository as deployed. |
| Continuous monitoring. | `ComplianceViolation`, `AttestationAccepted`, and every consent-state transition are emitted as events. Off-the-shelf indexers (The Graph, custom subgraphs, Tenderly listeners) can ingest them and alert. | The §IV.2 + §IV.3 demos exercise every monitoring-relevant code path. |
| Cooperation with regulators. | The chain is publicly readable. Regulators can run their own node, ingest events, and verify any UA's MIA scores by re-running the protocol off-chain against the published `modelHashPost`. | The per-method `onchain_attestation.json` artefacts at `results/{nbaiot,casas}_unlearned/<method>/` contain every tx hash, MIA score, model hash, and verifier signature a regulator needs to reproduce the audit independently. |

---

## 6. CCPA § 1798.105 — Right to delete

> _"A consumer shall have the right to request that a business delete
> any personal information about the consumer which the business has
> collected from the consumer."_ — Cal. Civ. Code § 1798.105(a)

CCPA's right-to-delete maps closely to GDPR Article 17. ConsentChain's
unlearning mechanism satisfies both with the same on-chain artefacts.

| Requirement | ConsentChain mechanism | Evidence |
|---|---|---|
| Deletion on consumer request. | `revokeConsent(tokenId)` is the consumer-initiated trigger; the rest of the protocol mirrors GDPR Art. 17 (see §2 above). | Same as §2. |
| Deletion within 45 days of request. | `Δ_max` at 1000 blocks (≈ 33 min on Polygon at 2 s/block; faster on PureChain) is well under 45 days. | Same as §2. |
| Service providers must also delete. | In the FL framing, the "service providers" are the participating devices (FL_k). The `FLGatekeeper.validateRound` revert on `pendingUnlearningCount > 0` forces every participant to halt training until the UA is accepted — i.e., propagates the deletion obligation to every downstream party. | §IV.2 demo + the integration test in `contracts/test/Integration.test.ts`. |

---

## 7. Cross-regime comparison — single artefact, multiple compliances

The strength of the design is that one Unlearning Attestation
simultaneously discharges:

- **GDPR Art. 17** (erasure on request, with verification),
- **EU AI Act Art. 15 + 26** (lifecycle robustness + deployer monitoring),
- **CCPA § 1798.105** (consumer right to delete, including service-provider propagation).

The deployer does not need separate compliance workflows per regime.
The chain is the single source of truth, and every regulator that wants
to verify a specific compliance event can read the same `AttestationAccepted`
or `ComplianceViolation` event from the same on-chain log.

---

## 8. What ConsentChain does **not** discharge

To be honest with reviewers and regulators, the following obligations
are **outside** the scope of ConsentChain and must be handled by
adjacent systems in any production deployment:

- **GDPR Art. 13/14 (information notices).** The act of giving a DS
  the information required to make an informed consent decision is a
  UX concern, not a chain concern. ConsentChain records the _result_
  of consent, not the briefing that preceded it.
- **GDPR Art. 35 (DPIAs).** The Data Protection Impact Assessment for
  an FL deployment must be performed by the deployer. ConsentChain
  reduces certain risks (centralised consent storage, unverifiable
  erasure) and creates new ones (chain liveness dependency).
- **GDPR Art. 32 (security of processing).** The DS's private key
  custody, network transport security, and verifier-host security are
  out of scope (see `docs/threat-model.md` §4).
- **CCPA right-to-know (§ 1798.110, 1798.115).** ConsentChain does
  not enumerate what data is held about a consumer — only that consent
  exists and what its current state is.

These are well-defined boundaries; the protocol does not pretend to
solve them and reviewers should treat any claim that it does as a bug.

---

**End of `docs/regulatory-alignment.md`.**
