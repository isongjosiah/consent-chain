# ConsentChain (ICTC) — Camera-Ready Revision Brief

**Status:** 3× Accept — R1: 4/4/4/4 (Accept), R2: 5/5/4/4 (Accept, no comments), R3: 5/4/4/4 (Accept).
No rebuttal phase; all edits below are discretionary camera-ready improvements. Nothing is owed to R2.

The two substantive comments converge on things the paper already *names* but doesn't *resolve* — so the fix is sentence surgery, not new experiments.

---

## Priority 1 — Verifier committee: incentives / collusion (R1 + R3, the shared signal)

**What they said.** R1: threshold committee is "a trust floor susceptible to verifier collusion without formal zero-knowledge or theoretical erasure guarantees." R3: "detail economic incentives or penalty mechanisms for the verifier committee."

**What the paper already says.** III-E: "The committee is the trust floor: ≥ t colluding verifiers can certify an invalid attestation, so verifier-incentive design is future work." V: future work lists ZK eligibility proofs, PoA²-native verifiers, cross-chain portability.

**The edit.** Convert "future work" into a named mechanism sketch. Two variants depending on space:

**Variant A (recommended, ~3 lines, place in III-E replacing the current sentence):**

> The committee is the trust floor: ≥ t colluding verifiers can certify an invalid attestation. The remedy we defer to the journal version is incentive-layered: verifiers post stake slashed on a successful challenge — DS_i, who uniquely holds D_i, can dispute an accepted UA within a window by exhibiting a contradicting MIA evaluation against the attested (h_pre, h_post) — with per-attestation committees drawn by VRF from a larger registered pool to resist pre-collusion, and BLS aggregation keeping signature cost flat as n grows.

The DS_i-as-challenger point is the load-bearing one: the data subject is the *only* party who both holds D_i and has the incentive to dispute, so the dispute mechanism doesn't require trusting yet another auditor with the data.

**Variant B (minimal, ~1 line, extend the future-work sentence in V):**

> …zero-knowledge eligibility proofs, PoA²-native verifiers, cross-chain consent portability, and staked, VRF-rotated verifier committees with subject-initiated dispute slashing.

Since two of three reviewers raised this independently, Variant A is worth the space (offset below).

---

## Priority 2 — Federation scale (R3)

**What they said.** "Evaluating larger client pools beyond small toy testbeds (5–9 clients)."

**What the paper already has.** The live N ∈ [5, 500] validateRound sweep with linear gas (R² ≈ 1) and ≈3.2 ms latency — but Section IV-C never connects this to the client-count limitation, so the reviewer read 5–9 as the whole story.

**The edit (~2 lines, Section V, near the limitations discussion):**

> Our federations are dataset-bound (N-BaIoT ships nine device traces; our CASAS split yields five clients), while the on-chain path scales independently — validateRound is measured live to N = 500 with linear gas and ≈3.2 ms latency; scaling the learning and MIA-evaluation side to hundreds of clients is part of the journal evaluation.

**Optional, only if the camera-ready window allows:** a synthetic-partition run (split the nine N-BaIoT device traces into ~27 pseudo-clients) would give one cheap datapoint that the gate behaves at 3× scale, and would preempt the same comment from journal reviewers. Not required for camera-ready.

---

## Priority 3 (optional) — "no formal guarantees" framing (R1)

Section V already owns this ("deliberately conservative… 'reasonable measures' by predicate plus on-chain audit rather than formal erasure"). If you want one clause that reframes it as a design position rather than a gap:

> …rather than formal erasure — certified unlearning guarantees remain intractable for approximate methods at FL scale, so the gate is evidence-based by design —

---

## Space offsets (you're at the 4-page hard limit)

1. **Cut or compress the Section I roadmap paragraph** ("The rest of the paper proceeds as follows…") — the classic 4-page sacrifice, frees ~4 lines, enough for Variant A + the scale sentence.
2. If more is needed: the closing Related Work paragraph partially restates Table I and can lose a line.

---

## Polish checklist (pre-PDF-eXpress)

- [ ] **Index Terms trailing comma:** "…trustworthy AI, IoT," → drop the final comma.
- [ ] **E6/E7 labels** in Table II, Table III, and Fig. 2 captions are never defined in-text. Either strip them or anchor once, e.g. "(experiment IDs E6–E7 index the runs in docs/eval-protocol.md)."
- [ ] **PoA² spacing:** the extracted text shows "PoA2chain" run together in §II and §V — check the compiled PDF; likely a missing space/`\xspace` after the superscript macro.
- [ ] **Fig. 2:** the "revoke @ r=30" annotation appears to collide with the legend box ("Baseline (9 clients)revoke @ r=30" in extraction) — verify the render.
- [ ] **Ref [15]** "ebctc" → check intended capitalization (eBCTC?); protect with braces in BibTeX.
- [ ] **Ref [16]** Härterich — diacritic renders oddly in extraction; confirm the umlaut compiles correctly.
- [ ] **Optional, per your own calibration rule:** contribution (1) says "proved safety and liveness" while III-F says mechanised proofs are future work. "Established by construction" would be the strictly calibrated phrasing. Reviewers didn't object, so your call.

---

## Camera-ready mechanics

- [ ] Add the IEEE copyright notice per the acceptance instructions.
- [ ] Re-run IEEE PDF eXpress after edits; confirm still exactly 4 pages.
- [ ] Check the notification email for whether ICTC wants a summary-of-changes field in EDAS (usually not, but cheap to confirm).
- [ ] Confirm the acknowledgment grant numbers are final (IITP/NRF/ITRC percentages sum to 100 — good).
- [ ] Final upload via EDAS before the camera-ready deadline in the notification.
