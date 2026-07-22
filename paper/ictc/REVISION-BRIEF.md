# ConsentChain — ICTC 2026 Revision Brief

**Paste this into a Claude session that has access to the paper's LaTeX source.**

---

## Context

You are editing a 4-page IEEE two-column conference paper: *"ConsentChain: PureChain-Powered Verifiable Machine Unlearning and Dynamic Consent for Federated IoT Learning"*, targeting KICS ICTC 2026 (Jeju Island, 14–16 Oct 2026).

The paper has already been compressed from ~7 pages to exactly 4. **Do not undo that compression.** The edits below strengthen the paper's best-paper case; they are not a re-expansion.

### Hard constraints

| Constraint | Value |
|---|---|
| **Page limit** | **4 pages, absolute.** EDAS rejects anything longer. |
| Format | IEEE conference template, two-column, single-spaced, ≥10 pt, A4/Letter |
| Submission deadline | **July 31, 2026 (Final)** |
| Submission system | EDAS — https://edas.info/newPaper.php?c=35149 |
| PDF validation | IEEE PDF eXpress, Conference ID **71249X** |

**Recompile and check the page count after every edit.** If an edit pushes past 4.0 pages, recover space from the list in §"Space budget" below before dropping the edit.

### Two absolute rules

1. **Never invent a citation.** Where this brief asks for new references, search for real papers and verify author/title/venue/year against an actual source. A fabricated reference is worse than a missing one.
2. **Never alter a numerical result.** All MIA scores, gas counts, and USD figures must match the authors' experimental records exactly. Where this brief quotes a number, treat it as *to be verified against the source data*, not as authoritative.

---

## P0 — Do these first (highest impact)

### P0-1. Reframe the two datasets as a memorisation-regime pair

**Problem being fixed.** Table I currently reads badly to a hostile reviewer: the gate **accepts** the no-unlearning baseline and **rejects** naive retrain (the exact reference that never saw the data). As written, the paper never shows the gate doing useful positive work — there is no case where a genuinely leaking model is correctly rejected for the right reason.

**The fix is already in the data.** CASAS *is* the high-memorisation regime (shadow ≥ 0.84 pre-unlearning, most methods rejected, only SISA-Fed passes). Section V-E currently throws this away in one thin sentence. Promote it: N-BaIoT = weak-memorisation regime where the gate is correctly permissive; CASAS = strong-memorisation regime where it is correctly restrictive. Together they exercise **both branches** of the acceptance predicate.

**Action.** Replace the current §V-E with the following. Verify every number against the authors' CASAS results before committing; the bracketed figures come from earlier drafts and must be confirmed.

> **E. Cross-domain validation: the high-memorisation regime**
>
> N-BaIoT and CASAS are not a repetition but a pair spanning two memorisation regimes. The N-BaIoT federation memorises its revoked client only weakly — every baseline sits near random — so the gate's correct behaviour there is to admit. The 5-client CASAS BiLSTM is the opposite regime: pre-unlearning shadow MIA reaches [≥0.84], far above τ, and the gate must actively reject. It does. Naive retrain, gradient ascent, and FedEraser all remain at or above [0.84] and are refused; only SISA-Fed ([0.499/0.035/0.000] shadow/LiRA/label) drives all three attacks below threshold and is accepted. The two domains therefore exercise both branches of the acceptance predicate — correctly permissive where no membership signal survives, correctly restrictive where it does. Full cross-domain results appear in the journal version.

**Also adjust §V-B.** The existing sentence beginning "Even the no-unlearning baseline sits near random…" is good and honest — keep it, but add a forward pointer so the concession immediately resolves rather than hanging. Append to that sentence:

> "; Section V-E supplies the complementary high-memorisation regime in which the gate must, and does, reject."

**Rationale to preserve if you rewrite:** this converts the paper's single biggest logical vulnerability into evidence of soundness across regimes, using existing data and no new experiments.

---

### P0-2. Add recent references (2023–2025)

**Problem.** The newest *technical* citation is LiRA (2022). SISA and FedEraser are 2021. Meanwhile the regulatory framing cites the 2024 EU AI Act. A 2026 machine-unlearning paper whose unlearning literature stops at 2021 reads as under-engaged — federated unlearning has moved enormously since.

**Action.** Add **4–5 verified references from 2023–2025**, drawn from these areas:

- Federated unlearning methods or surveys (post-2022)
- Certified / verifiable unlearning with formal guarantees
- Blockchain-based FL addressing governance, incentives, or auditability (post-2022)
- **Critiques of MIA-based unlearning evaluation** — highest strategic value: if recent work questions whether MIA reliably measures forgetting, citing and engaging it turns the paper's known weakness into demonstrated awareness of the field

Then add **one positioning sentence** at the end of §II ("Positioning") acknowledging that federated unlearning has matured since 2021 and stating what ConsentChain still adds that these do not: the *chain-enforced binding* of consent state to verified forgetting.

**Verify each reference exists** before insertion.

---

### P0-3. Restore the Hilprecht citation

The cut removed the reference justifying the shadow-MIA adaptation to an **autoencoder** (Hilprecht et al., *Monte Carlo and Reconstruction Membership Inference Attacks Against Generative Models*, PoPETs 2019). This is not a standard construction — without it, a reviewer may reasonably ask why a shadow attack is valid on a reconstruction model at all. Restore it at the point where the shadow attack is introduced in §IV-B-2.

Also restore, if space permits: **Flower** (named in §V-A but now uncited) and **EIP-721** (soulbound ERC-721 is invoked but uncited). These are lower priority than Hilprecht.

---

## P1 — Strong value, do if space allows

### P1-1. Restore one displayed equation: the acceptance predicate

**Problem.** The intro promises "four formally-stated safety and liveness properties," but §III-E states them in prose and the paper now contains essentially no formalism. That is a claim/content mismatch a reviewer will notice. Separately, the four acceptance conditions in §IV-B-3 are currently one long run-on sentence that is hard to parse.

**Action.** Convert the four-condition prose in §IV-B-3 back into a single compact displayed equation — the conjunction of: (i) all three fixed-point MIA scores below τ^fp = 5500; (ii) at least *t* distinct recovered signers in *VC*; (iii) `h_pre` equal to `CR.modelHashAtRevocation(i)`; (iv) token state ∈ {REVOKED, EXPIRED}.

Typeset **compactly** — chain conjuncts with ∧ across as few lines as possible; push all "where" definitions into the running text. Target ≤4 lines total.

**If there is genuinely no space:** do not force it. Instead soften the intro claim from "four formally-stated safety and liveness properties" to "four stated safety and liveness properties."

### P1-2. Bridge to the conference's technical centre of gravity

**Why.** ICTC 2026's program is strongly wireless/6G-oriented (keynotes on transmission, AI-RAN, movable antennas, AI-native systems, 6G NTN). This paper is thematically adjacent, not central. Two sentences make the fit explicit before a reviewer asks.

**Action.** Insert into §I, after the opening sentence about FL in consumer IoT:

> As learning migrates to the network edge, these federations become infrastructure: consumer-IoT, smart-home, and vehicular deployments train continuously over data that remains subject to withdrawal at any time. Governance for such AI-native systems cannot remain a policy commitment audited after the fact — it must be machine-checkable at the point where data enters training.

### P1-3. Justify τ = 0.55

Currently τ appears without principled grounding; "(α, τ) co-calibrated per model class" in §VI is the only defence. Add **one sentence** at the first use of τ in §IV-B-3 stating the operating-point rationale — e.g. that τ sits just above the random-guess baseline of 0.5 with margin for finite-sample noise in |M|, and that the conjunction over three attacks (rather than any single one) is what makes a tight threshold tolerable.

### P1-4. Close the verifier-committee gap

The committee is now the root of trust, yet §III-D lists five adversaries and verifier collusion is not among them. Add **one clause** to §III-D acknowledging that ≥ *t* colluding verifiers can certify an invalid attestation, that this is the protocol's trust floor, and that verifier-incentive design is future work. Acknowledging it costs one line; leaving it unstated invites the reviewer to find it.

---

## P2 — Hygiene and logistics

- **§Abstract:** change "**proving** it faithful to MIA evidence" → "**showing**" or "**demonstrating**". Given the concession in §V-B that the no-unlearning baseline also passes, "proving" overclaims.
- **Author list:** a co-author and the second affiliation were dropped relative to the previous version. Confirm this is intentional and that all parties consented — **IEEE prohibits authorship changes after acceptance**, so it must be settled before submission.
- **Acknowledgment / funding:** check whether a funding acknowledgment (IITP/NRF or equivalent) is required. If so, budget ~3 lines. Also note IEEE requires disclosure in the acknowledgments if AI-generated text was used (routine grammar/editing assistance is generally outside this policy).
- **Consistency check:** the intro claims "six states, nine transitions" — verify against the §III-C prose after edits.
- **CFP version:** the authors hold v13; **v14 is now current**. Diff for any track or formatting changes.

---

## Space budget

Additions above total roughly 12–16 lines. Recover an equivalent amount from these, in order of preference:

1. **§VII Conclusion** — currently restates all four headline numbers nearly verbatim from the abstract. Compress to the thesis plus two numbers; save 3–5 lines.
2. **§IV-A ("On-chain access control and safety")** — nearly vestigial; one sentence of safety argument under its own subheading. Merge into §III-E or the head of §IV-B and delete the subsection; save 3–4 lines.
3. **§II "Positioning" paragraph** — tighten; save 2–3 lines.
4. **§V-A setup** — architecture and hyperparameters can shed a clause; save 1–2 lines.

**Do not** recover space from: Table I, Table II, the hero-result discussion in §V-B, or the conservative-by-design paragraph in §VI. Those carry the paper.

---

## Do not touch

- Any numerical result, gas count, or USD figure
- The two tables
- The core thesis framing ("missing edge, not a missing vertex")
- The conservative-by-design defence in §VI
- The hero result in §V-B (naive-retrain rejection)

---

## Final verification checklist

- [ ] Compiles to **≤ 4.0 pages**
- [ ] Every new reference verified as real (author, title, venue, year)
- [ ] No numerical result altered
- [ ] CASAS reframe present and numbers confirmed against source data
- [ ] Intro's formality claim matches what §III-E actually delivers
- [ ] "proving" → "showing" in abstract
- [ ] Author list settled and consented
- [ ] Funding acknowledgment added if required
- [ ] No leftover cross-references to deleted figures/tables
- [ ] Full proofread (subject–verb agreement, caption references)
- [ ] Validated through IEEE PDF eXpress (Conference ID 71249X) — **do this early, not on deadline day**
- [ ] EDAS track selected deliberately: Track 3, item 3 (*Distributed Learning, Federated Learning, and Edge Intelligence*); check whether the AICA special track is a better reviewer match
- [ ] EDAS keywords include "edge intelligence" and "trustworthy AI" alongside existing terms
- [ ] Oral presentation requested (not poster — poster placement effectively removes a paper from best-paper contention)

---

*Deadline: July 31, 2026 (Final). Separately: the ICTC 2026 Travel Grant offers up to $500 for presenting students and young researchers, applications by September 10 — worth applying for independently of the paper outcome.*
