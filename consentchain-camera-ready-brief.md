# ConsentChain (ICTC) — Camera-Ready Revision Brief

**Status:** 3× Accept — R1: 4/4/4/4 (Accept), R2: 5/5/4/4 (Accept, no comments), R3: 5/4/4/4 (Accept).
No rebuttal phase; all edits below are discretionary camera-ready improvements. Nothing is owed to R2.

The two substantive comments converge on things the paper already *names* but doesn't *resolve* — so the fix is sentence surgery, not new experiments.

---

## Priority 1 — Verifier committee: incentives / collusion (R1 + R3, the shared signal) — ✅ DONE

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

## Priority 2 — Federation scale (R3) — ✅ DONE (synthetic-partition run not attempted)

**What they said.** "Evaluating larger client pools beyond small toy testbeds (5–9 clients)."

**What the paper already has.** The live N ∈ [5, 500] validateRound sweep with linear gas (R² ≈ 1) and ≈3.2 ms latency — but Section IV-C never connects this to the client-count limitation, so the reviewer read 5–9 as the whole story.

**The edit (~2 lines, Section V, near the limitations discussion):**

> Our federations are dataset-bound (N-BaIoT ships nine device traces; our CASAS split yields five clients), while the on-chain path scales independently — validateRound is measured live to N = 500 with linear gas and ≈3.2 ms latency; scaling the learning and MIA-evaluation side to hundreds of clients is part of the journal evaluation.

**Optional, only if the camera-ready window allows:** a synthetic-partition run (split the nine N-BaIoT device traces into ~27 pseudo-clients) would give one cheap datapoint that the gate behaves at 3× scale, and would preempt the same comment from journal reviewers. Not required for camera-ready.

---

## Priority 3 (optional) — "no formal guarantees" framing (R1) — ✅ DONE

Section V already owns this ("deliberately conservative… 'reasonable measures' by predicate plus on-chain audit rather than formal erasure"). If you want one clause that reframes it as a design position rather than a gap:

> …rather than formal erasure — certified unlearning guarantees remain intractable for approximate methods at FL scale, so the gate is evidence-based by design —

---

## Space offsets (you're at the 4-page hard limit)

1. **Cut or compress the Section I roadmap paragraph** ("The rest of the paper proceeds as follows…") — the classic 4-page sacrifice, frees ~4 lines, enough for Variant A + the scale sentence.
2. If more is needed: the closing Related Work paragraph partially restates Table I and can lose a line.

---

## Polish checklist (pre-PDF-eXpress)

- [x] **Index Terms trailing comma** — no change needed; the final comma was already absent in the compiled PDF.
- [x] **E6/E7 labels** — stripped from the Table II, Table III and Fig. 2 captions rather than anchored, since anchoring costs space the page budget did not have.
- [x] **PoA² spacing** — false alarm. `pdftotext` drops the space after math mode; the PDF renders "PoA² chain" correctly in both §II and §V.
- [x] **Fig. 2 legend collision** — real. "revoke @ r=30" overprinted "Baseline (9 clients)". Fixed in `fl/experiments/collect_results.py:render_figure2` (annotation moved left of the rule, `ha="right"`, down to 30 % of axis height) and the figure regenerated.
- [x] **Ref [15] eBCTC** — brace-protected in BibTeX; now renders "eBCTC … K-ETS". Same fix applied to "XS-FedPRS" in [2] and "Monte Carlo" in [16], which had the same lowercasing problem.
- [x] **Ref [16] Härterich** — already correct. `H\"{a}rterich` compiles fine; the odd rendering was an extraction artefact.
- [x] **Calibration of contribution (1)** — applied. Now reads "nine transitions whose safety and liveness hold by construction", consistent with III-F leaving mechanised proofs to future work.

---

## Camera-ready mechanics

**Blocked on you:** the IEEE copyright notice string (comes from the acceptance email / EDAS), PDF eXpress, and the EDAS upload.


- [ ] Add the IEEE copyright notice per the acceptance instructions.
- [ ] Re-run IEEE PDF eXpress after edits; confirm still exactly 4 pages.
- [ ] Check the notification email for whether ICTC wants a summary-of-changes field in EDAS (usually not, but cheap to confirm).
- [x] Acknowledgment updated 2026-09-10 to four grants at 25 % each (IITP, NRF Priority Research Centers, ITRC, and NRF Basic Science Research RS-2025-25431637). Sums to 100.
- [ ] Final upload via EDAS before the camera-ready deadline in the notification.

---

## Applied edits (session of 2026-09-10)

Paper is back to **exactly 4 pages** after every edit below; `latexmk` clean, no new overfull boxes.

**Regressions fixed first (introduced in commit 3d5f5bd, not part of this brief):**
- §IV-C cost paragraph had lost its `\$` escapes, `\texttt{}` wrappers and `$\sim\!30$` — rendered as "cost is 0 on PureChain and 0.00120 on Amoy". Restored.
- §V limitations sentence had lost its verb ("formal-erasure and ECDSA-leakage guarantees (BLS-collapsed in the journal), and while…") — it was an ungrammatical fragment. Rewritten, and the detectability argument kept.

**Brief edits:** P1 Variant A (condensed to ~6 lines) in III-E; P2 scale limitation folded into the §V limitations paragraph; P3 "evidence-based by design" clause.

**Space offsets used** (needed ~7 lines; the paper had zero slack):
1. Cut the §I roadmap paragraph (brief-sanctioned).
2. Compressed the closing Related Work paragraph (brief-sanctioned).
3. Compressed the §IV opening sentence.
4. Tightened the new P1 and P2 text after first draft.

**Layout bugs found while re-rendering** (caused by the `\vspace{-0.5cm}` before section headings, which only misbehave when a heading lands at a column bottom):
- "III. SYSTEM MODEL AND FRAMEWORK" was overprinting the last line of §II. Reduced to `-0.2cm`.
- "IV. EVALUATION" shared a baseline with the preceding "work." Reduced to `-0.2cm`.

**Still open:**
- `ajakwe2025federated` was added to `references.bib` but is never `\cite`d, so it does not appear in the reference list. Citing it costs ~3 lines of references, which the 4-page budget does not currently have.

### Acknowledgment update (2026-09-10)

The fourth grant added 4 lines and pushed the paper to 5 pages (ref [16] spilling). Recovered without touching any claim:

- Fig. 2 regenerated shorter (`figsize` 2.6 → 2.35 in) with a shortened y-label, `"Test recon. error (MSE)"` — the original label was clipped at the reduced height, and the caption still names the centralized test set.
- Fig. 2 caption trimmed by one line.
- §V: "a permissioned-yet-public PoA² chain or permissionless Amoy" → "PoA² or permissionless Amoy"; conclusion opening tightened by a line.

Two space levers were tried and **rejected**: tightening the bibliography `itemsep` past `-0.65ex` overlapped adjacent reference entries at `-0.95ex`, and shrinking Fig. 2 to 2.25 in clipped the y-label. Both are dead ends if more space is ever needed.

Minor typography: added the missing spaces in "Korea(NRF)" and "Education(RS-2025-25431637" from the supplied text. Say the word if the funder requires it character-exact.
