# CLAUDE.md — ConsentChain

> **Smart contract-enforced dynamic consent and verifiable machine unlearning for federated IoT learning**
> IEEE BCCA 2026 submission — deadline **June 15, 2026** (extended from April 15) (8-page long paper)
> Venue: Barcelona, Spain, November 16–20, 2026. Track 8: Blockchain for Trustworthy & Ethical AI / Track 11: Blockchain for IoT and CPS.

This document is the authoritative brief for Claude Code sessions on this project. Read it end-to-end before making any architectural decisions. If you're about to write code that contradicts what's here, stop and flag it instead.

---

## 1. Who is building this and why

**Researcher:** Josiah Ayoola Isong — Lead Backend Engineer (Go, Python, TypeScript, AWS), M.S. candidate in IT Convergence Engineering at Kumoh National Institute of Technology, Graduate Researcher in the Network Systems Laboratory. Domain expertise: federated learning, Byzantine-robust aggregation, privacy-preserving ML, blockchain, ZK-proofs, genomics/healthcare applications.

**Why ConsentChain and not something else.** Josiah's existing portfolio covers federated PRS with Byzantine robustness (XS-FedPRS, RV-FedPRS), federated epidemic intelligence (FedEpi), privacy-preserving NFT metadata (Plonky3/STARKs), and verifiable storage substrates (LBVR-Med). None of this work addresses the **consent enforcement gap** in federated learning. The literature treats consent as a static opt-in binary: once a hospital, smart home, or device joins an FL deployment, their data contributes indefinitely. But GDPR Article 17 (right to erasure), the EU AI Act's high-risk AI provisions (effective August 2026), and CCPA §1798.105 all require that data subjects be able to *withdraw* — and that the trained model demonstrably *forget*. Three research threads attack this from separate angles: consent management in data systems (GConsent ontology, MedRec), machine unlearning (SISA, FedEraser), and blockchain for FL (BlockFL, BFLC). No published work binds all three into a single protocol. The consent-unlearning-verification triangle is the gap this paper fills.

**IoT framing.** The problem is most acute in IoT federated learning, where devices continuously collect intimate data — home occupancy patterns, body movement, network behavior — often under opaque blanket consent. IoT also stresses the economic feasibility question: if every consent state check costs gas, the model breaks at 10³+ devices. PureChain's gas-free architecture eliminates this. Healthcare and finance are secondary application domains; IoT is the primary.

---

## 2. The four contributions

> **We make four contributions:**
> 1. **Consent Lifecycle State Machine (CLSM):** A formally specified finite state machine (6 states, 9 transitions) with proved safety and liveness properties, modeling consent as a stateful process with purpose limitation, temporal bounds, and partial revocation. Implemented as soulbound ERC-721 tokens.
> 2. **Verifiable Unlearning Protocol (VUP):** An unlearning attestation scheme requiring three independent membership inference attack variants (shadow model, LiRA, label-only) plus a backdoor canary test to all fall below threshold τ before acceptance. Method-agnostic — works with SISA-Fed, gradient ascent, or FedEraser.
> 3. **Three-contract Solidity architecture on PureChain:** ConsentRegistry, UnlearningOracle, FLGatekeeper — deployed on PureChain (public, gas-free, PoA² consensus) and Polygon Amoy (public, gas-based, PoS). Dual deployment produces the first apples-to-apples cost comparison showing gas-free infrastructure is essential for IoT-scale consent management.
> 4. **Rigorous dual-domain IoT evaluation:** CASAS smart home (deep, primary) and N-BaIoT network security (validation) — two contrasting IoT domains demonstrating effective unlearning (MIA → ~50%), minimal utility loss (<3%), and zero on-chain cost on PureChain.

Everything in the paper serves these four contributions. If a feature doesn't support at least one, it's out of scope.

---

## 3. Scope decisions (locked)

### 3.1 Conference (April 15 submission)

| Decision | Value | Rationale |
|---|---|---|
| Primary dataset | **CASAS smart home (via FedAIoT)** — 5 apartments (Cairo, Milan, twor.2009, twor.2010, twor.summer.2009), 12 classes (Other + 11 ADL), sensor time-series (motion, temp, door) | Natural per-apartment FL split; intimate personal data makes consent revocation narratively compelling; FedAIoT's `datasets/casas/download.py` defines preprocessing (we reimplement it without TF) |
| Secondary dataset | **N-BaIoT** — 9 IoT devices, 7M instances, 115 network features, binary anomaly detection | Contrasts CASAS on every dimension (network vs. sensor, organizational vs. personal, cross-device vs. cross-silo); validates generalizability |
| FL framework | **Flower (flwr) 1.x + PyTorch** | Industry standard; clean client/server abstraction; Python ecosystem for MIA libraries |
| Primary blockchain | **PureChain** — public, gas-free, EVM-compatible, PoA² consensus, Merkle-proof-backed state, persistent storage | Zero gas eliminates IoT cost barrier; PoA² validators as natural Verifier Committee; Josiah's own infrastructure with full control |
| Comparison blockchain | **Polygon Amoy** — public L2 testnet, measurable gas costs | Identical Solidity contracts; produces cost comparison data; publicly verifiable by reviewers |
| Smart contract stack | **Solidity 0.8.20+, Hardhat, OpenZeppelin v5, ethers.js v6** | EVM-compatible with both PureChain and Polygon Amoy |
| Unlearning methods | **SISA-Fed, Gradient Ascent, FedEraser** (all three, compared) | Covers the three main paradigms; ConsentChain is method-agnostic, evaluating all three proves it |
| MIA attacks | **Shadow model (Shokri 2017), LiRA (Carlini 2022), Label-only (Choquette-Choo 2021)** + backdoor canary | Three independent attack methodologies eliminates "weak MIA" reviewer objection; canary is the strongest forgetting test |
| Verifier Committee | **ECDSA multi-sig (3-of-5)** on both chains | BLS threshold signatures are ideal but complex to implement by deadline; ECDSA multi-sig is an acceptable prototype simplification; state BLS as production upgrade in §VI |
| Baselines | **FedEraser** (closest existing), **Naive retraining** (gold standard), **No unlearning** (do-nothing) | Fair comparison; FedEraser is the most-cited federated unlearning method |

### 3.2 Post-conference extensions

| Decision | Value |
|---|---|
| ZK-consent proofs | Hide *who* consented while proving eligibility — extend with zk-SNARKs over CLSM state |
| PoA² native VC integration | Validators verify UAs during block production — consensus-consent fusion |
| Cross-chain consent portability | Consent granted on PureChain honored on Polygon / other L2s |
| Journal target | IEEE Internet of Things Journal or IEEE TDSC |

---

## 4. Architecture (conference scope)

### 4.1 System layers

```
┌──────────────────────────────────────────────────────────────┐
│  L4: Verifiable Unlearning Protocol (Python, off-chain)      │
│      • Three MIA attacks + backdoor canary test              │
│      • Unlearning Attestation generation                     │
│      • Verifier Committee signature collection               │
└──────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────┐
│  L3: Smart Contract Layer (Solidity, on-chain)               │
│      • ConsentRegistry.sol — soulbound Consent NFTs + CLSM  │
│      • UnlearningOracle.sol — UA validation + threshold sigs │
│      • FLGatekeeper.sol — pre-round consent gate + timeout   │
└──────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────┐
│  L2: Federated Learning Orchestrator (Python + Flower)       │
│      • FedAvg server with round logging + model hashing      │
│      • Consent-aware client selection via web3 → FLGatekeeper│
│      • Pluggable unlearning: SISA-Fed / GradAsc / FedEraser  │
└──────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────┐
│  L1: IoT Data Layer                                          │
│      • CASAS: smart home sensor streams (motion/temp/door)   │
│      • N-BaIoT: IoT network traffic features (115-dim)       │
│      • Per-client federated partitioning (per-apartment /    │
│        per-device)                                           │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Consent Lifecycle State Machine (CLSM)

Formally: CLSM = (S, Σ, δ, s₀, F) where:
- S = {INACTIVE, ACTIVE, SUSPENDED, REVOKED, EXPIRED, UNLEARNED}
- Σ = {grant(P,T), modify(P',T'), suspend(), resume(), revoke(P'), expire(), unlearn_verified()}
- δ: S × Σ → S (transition function)
- s₀ = INACTIVE
- F = {UNLEARNED} (terminal absorbing state)

**State transition table:**

| Current | Event | Next | Guard |
|---|---|---|---|
| INACTIVE | grant(P,T) | ACTIVE(P,T) | DS signature valid |
| ACTIVE | modify(P',T') | ACTIVE(P',T') | DS signature valid |
| ACTIVE | suspend() | SUSPENDED | DS or regulator signature |
| ACTIVE | tick(round_id) | ACTIVE | P includes round purpose, T > block.timestamp |
| ACTIVE | revoke(P') | REVOKED(P') | DS signature valid |
| ACTIVE | expire() | EXPIRED | block.timestamp > T |
| SUSPENDED | resume() | ACTIVE | DS signature valid |
| REVOKED | unlearn_verified() | UNLEARNED | Valid UA on-chain |
| EXPIRED | unlearn_verified() | UNLEARNED | Valid UA on-chain |

**On-chain representation:** Soulbound ERC-721 (transfer disabled). Metadata stores ConsentState enum, purposeSet (bytes32), ttl (uint256), dsOwner (address). Only the DS's private key can invoke transitions on their token. Only UnlearningOracle can call transitionToUnlearned().

**Formal properties (must be proved in paper and enforced in tests):**

1. **Consent Integrity (Safety):** No FL round may include DS_i's data unless ConsentRegistry.state(DS_i) = ACTIVE with matching purpose and valid TTL. No path from REVOKED → ACTIVE exists.
2. **Unlearning Enforceability (Liveness):** If state transitions to REVOKED or EXPIRED, then within Δ_max blocks, either a valid UA is committed or FLGatekeeper blocks all rounds. System degrades to "no training" rather than "unconsented training."
3. **Unlearning Verifiability:** UA accepted iff ≥ t of n VC members confirm all three MIA scores < τ.
4. **Audit Completeness:** Every consent transition, every UA, every model hash is logged on-chain.

### 4.3 Verifiable Unlearning Protocol (VUP)

Triggered by `UnlearningRequired(dsTokenId, modelHash)` event from ConsentRegistry.

**Step 1 — Unlearning execution (off-chain).** DC performs one of:
- **SISA-Fed:** Reload pre-shard checkpoint, retrain excluding DS_i's data. Requires pre-training with S=3 shards per client.
- **Gradient Ascent:** Apply revoked DS_i's cached gradients with negative learning rate. Steps: 10 (configurable).
- **FedEraser:** Replay aggregation history excluding DS_i's updates, apply calibration.

ConsentChain is method-agnostic. The protocol requires only `model_hash_pre` and `model_hash_post`.

**Step 2 — MIA evaluation (off-chain).** Three independent attacks on the post-unlearning model, targeting DS_i's data:
- Shadow model attack (Shokri et al. 2017): 5 shadow models, MLP attack classifier, reports accuracy.
- LiRA (Carlini et al. 2022): 4 IN + 4 OUT reference models, Gaussian fit, log-likelihood ratio.
- Label-only (Choquette-Choo et al. 2021): perturbation-based, robustness score.
- Backdoor canary: pre-injected pattern, trigger rate (target: 0% post-unlearning).

**Step 3 — UA generation.** Struct:
```
UA = {
    dsTokenId:          uint256,
    modelHashPre:       bytes32,
    modelHashPost:      bytes32,
    unlearningMethod:   uint8,        // 0=SISA, 1=GRAD_ASCENT, 2=FEDERASER
    shadowMiaScore:     uint256,      // fixed-point ×1e4 (5000 = 0.5000)
    liraScore:          uint256,
    labelOnlyMiaScore:  uint256,
    timestamp:          uint256,
    uaHash:             bytes32       // keccak256 of above
}
```

**Step 4 — VC validation.** ≥ 3 of 5 verifiers independently confirm all three scores < miaThreshold (default: 5500 = 0.55), sign the UA hash with ECDSA.

**Step 5 — On-chain commitment.** `UnlearningOracle.submitAttestation(ua, signatures[])` → verifies threshold → stores hash → calls `ConsentRegistry.transitionToUnlearned()` → calls `FLGatekeeper.unblock()`.

### 4.4 Smart contract specifications

**See `docs/contracts-spec.md` for full NatSpec interfaces.** Summary:

**ConsentRegistry.sol:** Manages soulbound Consent NFTs. Key functions: `grantConsent()`, `modifyConsent()`, `suspendConsent()`, `resumeConsent()`, `revokeConsent()`, `checkExpiry()`, `transitionToUnlearned()` (only callable by UnlearningOracle). Emits: `ConsentGranted`, `ConsentRevoked`, `ConsentExpired`, `UnlearningRequired`, `UnlearningCompleted`.

**UnlearningOracle.sol:** Validates UAs. Manages VC membership (add/remove verifiers, set threshold). Checks all three MIA scores < miaThreshold. Verifies ≥ threshold ECDSA signatures from registered verifiers. Stores UA hashes. Triggers ConsentRegistry state transition.

**FLGatekeeper.sol:** Pre-round gate. `validateRound(roundId, participantTokenIds[], roundPurpose)` → queries ConsentRegistry for each participant → returns eligible[] or reverts if pending unlearning. Δ_max timeout (default: 1000 blocks) triggers `ComplianceViolation` event. `updateModelHash()` after each round. `unblock()` called by UnlearningOracle.

### 4.5 Dual-chain deployment

**PureChain (primary):** Public, gas-free, PoA² consensus. All contract calls cost zero gas. `validateRound()` at any N costs $0. PoA² validators serve as natural VC members.

**Polygon Amoy (comparison baseline):** Public L2 testnet. Identical contracts. Measurable gas costs → USD conversion at current MATIC price. Produces the cost comparison data for Table 2 and the scalability plot.

**The killer argument:** At 1,000 IoT devices × 10 consent operations/day, PureChain costs $0/day. Polygon Amoy costs $X/day. This quantifies why gas-based chains are structurally incompatible with IoT-scale consent. No other blockchain-FL paper makes this comparison.

---

## 5. Threat model

### In scope

| Adversary | Capability | Defense |
|---|---|---|
| Lazy FL Orchestrator | Silently ignores revocation, continues training with revoked DS's data | FLGatekeeper blocks rounds when pending_unlearning > 0; Δ_max timeout triggers ComplianceViolation |
| Deceptive Data Controller | Claims unlearning was performed without actually doing it | VUP: three independent MIAs + canary must all pass; VC independently re-runs attacks |
| Colluding participants | Cache model updates to reconstruct revoked DS's data | Out of scope for consent enforcement; acknowledged as limitation; differential privacy on updates is future work |
| Consent forger | Attempts to grant/revoke consent on behalf of another DS | Soulbound NFT: only dsOwner private key can invoke transitions |
| Replay attacker | Submits previously valid UA for a different revocation | UA includes dsTokenId + modelHashPre; Oracle checks modelHashPre matches current model version |

### Out of scope (flag in paper limitations)

- Compromised DS private key (assume standard wallet security)
- Full BLS threshold signatures (ECDSA multi-sig is conference prototype; BLS is journal upgrade)
- Exact unlearning guarantees (approximate unlearning + MIA threshold τ is a pragmatic bound, not a mathematical proof of complete erasure — aligns with GDPR "reasonable measures" standard)
- Side-channel attacks on the MIA evaluation itself
- PoA² validator set compromise (permissioned validator set assumed honest-majority)
- Post-unlearning data reconstruction from public model checkpoints (orthogonal to consent enforcement)

---

## 6. Repo layout

```
consentchain/
├── CLAUDE.md                          ← this file
├── README.md                          ← lightweight overview, links to CLAUDE.md
├── .env.example                       ← RPC URLs, keys template
├── .gitignore
│
├── contracts/                         ← Solidity smart contracts
│   ├── hardhat.config.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── contracts/
│   │   ├── ConsentRegistry.sol
│   │   ├── UnlearningOracle.sol
│   │   ├── FLGatekeeper.sol
│   │   └── interfaces/
│   │       ├── IConsentRegistry.sol
│   │       ├── IUnlearningOracle.sol
│   │       └── IFLGatekeeper.sol
│   ├── test/
│   │   ├── ConsentRegistry.test.ts
│   │   ├── UnlearningOracle.test.ts
│   │   ├── FLGatekeeper.test.ts
│   │   └── Integration.test.ts        ← full lifecycle end-to-end
│   ├── scripts/
│   │   ├── deploy.ts                  ← deploy to PureChain + Polygon Amoy
│   │   ├── measure-gas.ts             ← gas measurement for Polygon
│   │   ├── measure-latency.ts         ← block finality timing
│   │   └── scalability-bench.ts       ← validateRound() at N=5..500
│   └── deployments/
│       ├── purechain.json             ← deployed addresses
│       └── polygon-amoy.json
│
├── fl/                                ← Federated learning pipeline (Python)
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── config/
│   │   ├── casas.yaml                 ← CASAS hyperparameters
│   │   └── nbaiot.yaml                ← N-BaIoT hyperparameters
│   ├── datasets/
│   │   ├── download_casas.py          ← via FedAIoT pipeline
│   │   ├── download_nbaiot.py         ← from UCI
│   │   ├── federated_split.py
│   │   └── canary_injection.py        ← backdoor canary for forgetting test
│   ├── models/
│   │   ├── bilstm_casas.py            ← BiLSTM for CASAS ADL recognition
│   │   └── autoencoder_nbaiot.py      ← Deep AE for N-BaIoT anomaly detection
│   ├── flower/
│   │   ├── server.py                  ← FedAvg server + round logging
│   │   ├── client.py                  ← dataset-agnostic FL client
│   │   ├── strategy.py                ← custom strategy with model hashing
│   │   └── consent_gate.py            ← pre-round on-chain check via web3
│   ├── unlearning/
│   │   ├── sisa_fed.py
│   │   ├── gradient_ascent.py
│   │   ├── federaser.py
│   │   └── model_hash.py              ← SHA-256 of state_dict
│   ├── mia/
│   │   ├── shadow_model.py            ← Shokri et al. 2017
│   │   ├── lira.py                    ← Carlini et al. 2022
│   │   ├── label_only.py              ← Choquette-Choo et al. 2021
│   │   └── canary_test.py             ← backdoor canary survival
│   ├── attestation/
│   │   ├── generate_ua.py             ← build UA struct
│   │   ├── submit_ua.py               ← submit to UnlearningOracle
│   │   └── verify_ua.py               ← off-chain VC simulation
│   └── experiments/
│       ├── run_casas.py               ← full CASAS pipeline (E1–E5)
│       ├── run_nbaiot.py              ← full N-BaIoT pipeline (E6)
│       ├── run_scalability.py         ← scalability experiments (E7–E8)
│       └── collect_results.py         ← aggregate → paper-ready tables
│
├── results/                           ← experiment outputs
│   ├── casas/
│   │   ├── metrics.json
│   │   └── model_checkpoints/
│   ├── nbaiot/
│   │   ├── metrics.json
│   │   └── model_checkpoints/
│   ├── blockchain/
│   │   ├── gas_comparison.json
│   │   ├── latency.json
│   │   └── scalability.json
│   └── tables/                        ← generated LaTeX
│       ├── table1_unlearning.tex
│       ├── table2_gas_comparison.tex
│       └── table3_feature_comparison.tex
│
├── docs/
│   ├── contracts-spec.md              ← full NatSpec interfaces, guard conditions
│   ├── mia-protocol.md               ← exact MIA implementation details
│   ├── eval-protocol.md              ← reproducibility commands for every figure
│   ├── threat-model.md               ← expanded from §5
│   └── regulatory-alignment.md       ← GDPR/EU AI Act/CCPA mapping
│
├── paper/
│   ├── main.tex                       ← IEEEtran, 8 pages, double-column
│   ├── sections/
│   │   ├── 00-abstract.tex
│   │   ├── 01-introduction.tex
│   │   ├── 02-related-work.tex
│   │   ├── 03-system-model.tex
│   │   ├── 04-framework.tex
│   │   ├── 05-evaluation.tex
│   │   ├── 06-discussion.tex
│   │   └── 07-conclusion.tex
│   ├── figures/
│   └── references.bib
│
└── scripts/
    ├── setup.sh                       ← full environment setup
    ├── run_all.sh                     ← complete experiment pipeline
    └── generate_paper_assets.sh       ← tables + figures from results/
```

---

## 7. Critical dependencies and versions

### Python (FL + MIA pipeline)
- **Python 3.12** (locked). 3.13/3.14 lack wheels for PyTorch 2.2, Flower 1.9, scikit-learn 1.4, and the FedAIoT preprocessing pipeline pins ≤3.11/3.12. Use `python3.12 -m venv fl/.venv` — do not bump to newer Python without re-verifying the entire ML stack ships matching wheels.
- `flwr` (Flower) 1.x — FL framework
- `torch` 2.x — model training
- `numpy`, `pandas`, `scikit-learn`
- `web3` 6.x — on-chain interaction from Python
- `pyyaml` — experiment configs
- `matplotlib`, `seaborn` — paper figures

### TypeScript / Solidity (smart contracts)
- **Node.js 18+**
- **Solidity 0.8.20+**
- **Hardhat** latest + plugins: hardhat-ethers, hardhat-toolbox, hardhat-gas-reporter
- **OpenZeppelin Contracts 5.x** — ERC721, Ownable, ReentrancyGuard, ECDSA
- **ethers.js v6**
- **TypeScript 5.x** (strict mode, no `any`)

### MIA-specific
- Shadow model: scikit-learn (RandomForest/MLP on confidence vectors)
- LiRA: port from Carlini et al.'s official repo (github.com/tensorflow/privacy/tree/master/research/mi_lira_2021), adapted TF → PyTorch. Use N_ref=8 (4 IN + 4 OUT). This is the hardest MIA to implement.
- Label-only: custom implementation, perturbation-based

### Data
- **CASAS:** via FedAIoT benchmark (github.com/AIoT-MLSys-Lab/FedAIoT). Subsets bundled in FedAIoT's CASAS Google-Drive zip: **Cairo, Milan, twor.2009, twor.2010, twor.summer.2009** (5 apartments → 5 FL clients). Preprocessing: categorical sensor-event encoding per their `datasets/casas/download.py` (`cookActivities`/`mappingActivities`), padded/truncated to 2000 events per activity instance. We reimplement their preprocessing in pure numpy/pandas in `fl/datasets/download_casas.py` to avoid pulling in TensorFlow (FedAIoT's only TF dep is `tf.keras.utils.pad_sequences`). Activity space: 12 classes (`Other` + 11 ADL). At training time the dataloader slices each sample to the **last 500 events** (`truncate_to_last_n: 500` in `fl/config/casas.yaml`) — most CASAS activity instances are far shorter than 500 events, so the tail captures the full activity at ≈4× cheaper LSTM cost than the 2000 baseline. Train/test split is **chronological per client** (last 20 % of each apartment's activities held out), stricter than a random permutation since the model never sees future routines during training.
- **N-BaIoT:** UCI ML Repository (archive.ics.uci.edu). 9 devices, 115 features, binary labels.

### Blockchain (keys in `.env.example`, never commit real keys)
- **PureChain** — public RPC endpoint (set as PURECHAIN_RPC_URL)
- **Polygon Amoy** — `https://rpc-amoy.polygon.technology` or Alchemy endpoint. Testnet MATIC from faucet.

---

## 8. Evaluation protocol — exact experiments

Every table/figure in the paper maps to one experiment. Name experiments E1–E8.

| ID | Experiment | Output | Expected runtime |
|---|---|---|---|
| E1 | CASAS FL baseline: FedAvg 100 rounds, 5 clients, log per-round accuracy + model hash | Checkpoint at round 60 (pre-revocation) + round 100 | ~2h |
| E2 | CASAS canary injection: inject backdoor in Client 3, verify trigger rate >90% | Trigger rate metric | ~30min |
| E3 | CASAS unlearning: revoke Client 3 at round 60, run SISA-Fed + GradAsc + FedEraser | 3 post-unlearning checkpoints + timing | ~3h |
| E4 | CASAS MIA evaluation: 3 attacks + canary on all unlearned models + baselines | **Table 1 (hero table)** — unlearning effectiveness | ~6h (LiRA dominates) |
| E5 | CASAS utility recovery: resume FL (rounds 61–70) with 4 remaining clients | **Figure 2a** — accuracy over rounds with dip+recovery | ~1h |
| E6 | N-BaIoT full pipeline: same as E1–E5 but 9 clients, 50 rounds, revoke Client 5 at round 30 | **Table 1 (N-BaIoT rows)** + **Figure 2b** | ~8h |
| E7 | On-chain cost comparison: deploy to PureChain + Polygon Amoy, measure gas for all operations, full lifecycle ×10 | **Table 2** — gas comparison | ~2h |
| E8 | Scalability: `validateRound()` with N∈{5,10,25,50,100,250,500} on both chains + revocation rate sweep | **Figure 1** — scalability plot | ~3h |

**Total estimated experiment time: ~26h** (not counting setup/debugging).

**Experiment reproducibility requirement.** Every experiment must be launched by a single `python experiments/run_{name}.py` command that writes raw JSON to `results/{dataset}/` with timestamp, commit hash, and random seed. Post-processing in `collect_results.py` reads JSON and emits LaTeX tables + PDF figures.

**Environment fingerprint** (saved to `results/{experiment}/env.json`):
```json
{
  "commit_hash": "abc123...",
  "python_version": "3.10.x",
  "torch_version": "2.x.x",
  "flower_version": "1.x.x",
  "os": "Ubuntu 22.04",
  "gpu": "NVIDIA ...",
  "random_seed": 42,
  "wall_start": "2026-04-06T07:00:00Z",
  "wall_end": "2026-04-06T13:12:45Z"
}
```

---

## 9. Paper structure (8-page IEEEtran)

| Section | Pages | Must-contain |
|---|---|---|
| I. Introduction | 0.75 | Triangle framing → IoT motivation → four contributions → Trautwein-equivalent gap statement citing Bourtoule 2021, Liu 2021, Azaria 2016 as separate threads |
| II. Related Work | 0.75 | Three converging threads (consent mgmt, machine unlearning, blockchain-FL), each ending with gap statement → positioning sentence |
| III. System Model & Threat Model | 0.75 | System entities, trust assumptions, four formal properties with safety/liveness sketches. Figure 3 = CLSM state diagram |
| IV. ConsentChain Framework | 2.0 | CLSM detail (0.5p), VUP with three MIA justification (0.75p), three-contract architecture with automation loop (0.75p) |
| V. Evaluation | 2.5 | **Largest section.** Table 1 (hero — unlearning effectiveness), Table 2 (gas comparison), Table 3 (feature comparison vs. FedEraser/MedRec/BlockFL), Figure 1 (scalability), Figure 2 (utility recovery) |
| VI. Discussion | 0.5 | Limitations (honest), regulatory mapping (GDPR Art. 17 → CLSM states, EU AI Act Art. 15), permissioned-vs-public tradeoff |
| VII. Conclusion | 0.25 | Restate triangle + key numbers (MIA→~50%, $0 gas, <3% utility loss) + future work (ZK-consent, cross-chain portability, PoA²-native VC) |

**Word budget:** ~8,000 words for 8 pages. Push detail to tables and figures.

**Figures/tables are the paper.** Reviewers will read Table 1 (unlearning effectiveness) and Table 2 (gas comparison) first. Invest disproportionate time in these.

---

## 10. Four-week execution plan

> **Schedule remapped 2026-05-13:** deadline extended from April 15 to **June 15, 2026**. D1 now starts May 13. Original spans preserved; D22 = Jun 10 with 5 buffer days (Jun 11–14) before the Jun 15 deadline.

### Week 1 — Smart contracts + FL baseline

**D1 (May 13)** — Repo skeleton. `npm init` for contracts, `pip install` for FL. Download CASAS via FedAIoT, download N-BaIoT from UCI. Verify both datasets load and preprocess correctly. Generate `docs/contracts-spec.md` — full NatSpec for all three contracts.

**D2 (May 14)** — Implement ConsentRegistry.sol. Soulbound ERC-721 with CLSM transitions. All guard conditions. Unit tests for every transition + revert cases. Soulbound transfer block.

**D3 (May 15)** — Implement UnlearningOracle.sol. UA validation, ECDSA multi-sig verification, MIA threshold check. Unit tests.

**D4 (May 16)** — Implement FLGatekeeper.sol. `validateRound()`, `pending_unlearning` queue, Δ_max timeout, `ComplianceViolation` event. Unit tests. **Integration.test.ts** — full lifecycle.

**D5 (May 17)** — Deploy to PureChain. Verify all functions execute gas-free. Record deployed addresses. Test full lifecycle on-chain.

**D6 (May 18)** — Deploy identical contracts to Polygon Amoy. Run `measure-gas.ts` for all operations. Record comparison. Test full lifecycle.

**D7 (May 19–20)** — FL baseline: CASAS FedAvg 100 rounds (E1). BiLSTM model. Save checkpoint at round 60. Log per-round accuracy + model hash.

**Week 1 exit criterion:** All three contracts deployed on both chains with passing tests. CASAS baseline trained with checkpoints saved. Gas comparison data collected.

### Week 2 — Unlearning + MIA + N-BaIoT

**D8 (May 21)** — Implement SISA-Fed, Gradient Ascent, FedEraser in `fl/unlearning/`. Unit tests on toy data first.

**D9 (May 22)** — Canary injection (E2). Inject backdoor in Client 3's CASAS data. Verify trigger rate >90%.

**🔴 GO/NO-GO CHECKPOINT at D9 EOD:**
- ✅ Canary triggers at >90% → proceed to unlearning experiments
- ❌ Canary doesn't trigger → increase poison ratio, strengthen pattern. Must resolve before D10.

**D10 (May 23–24)** — CASAS unlearning experiments (E3). Run all three methods. Save post-unlearning checkpoints. Record timing.

**D11 (May 25–26)** — MIA implementation. Shadow model first (simplest), then label-only, then LiRA (hardest). Run on CASAS (E4).

**🔴 GO/NO-GO CHECKPOINT at D11 EOD:**
- ✅ All three MIAs produce scores in [0,1] range and distinguish "no unlearning" (~65%) from "naive retrain" (~50%) → proceed
- ⚠️ LiRA flaky but shadow+label-only work → drop LiRA from the paper, report two MIA variants, acknowledge as limitation. Proceed.
- ❌ None of the MIAs work → **STOP.** Debug. This is the core measurement. Do not proceed to N-BaIoT until at least shadow model + label-only produce clean numbers.

**D12 (May 27)** — CASAS utility recovery (E5). Resume FL rounds 61–70. Measure recovery.

**D13 (May 28–29)** — N-BaIoT full pipeline (E6). Deep AE model, 50 rounds, revoke Client 5 at round 30. All three unlearning methods + all MIAs. If LiRA was dropped at D11, drop here too.

**Week 2 exit criterion:** Table 1 (hero table) has all CASAS and N-BaIoT rows filled. Figure 2 (utility recovery) renders for both datasets.

### Week 3 — Blockchain experiments + paper

**D14 (May 30)** — Scalability experiments (E7, E8). `validateRound()` at N=5..500 on both chains. Revocation rate sweep. Generate Figure 1 + Table 2.

**D15 (May 31)** — Full UA submission end-to-end on both chains. Verify ConsentRegistry transitions to UNLEARNED. Record latency.

**D16 (Jun 1–2)** — Generate `docs/eval-protocol.md` (reproducibility commands). Generate `docs/threat-model.md` and `docs/regulatory-alignment.md`. Run `collect_results.py` to aggregate all metrics → LaTeX tables + PDF figures.

**D17 (Jun 3–4)** — Draft Sections I–IV. Introduction (triangle + IoT motivation), Related Work (three threads), System Model (entities + properties), Framework (CLSM + VUP + contracts).

**🔴 GO/NO-GO CHECKPOINT at D17 EOD:**
- Is any section going to need more than a full day to finish? If yes, **cut the weakest experiment** rather than the section. Priority for cutting: E8 scalability sweep → E6 N-BaIoT (keep CASAS-only, note N-BaIoT as "confirmed in supplementary"). A tight paper with one deep case study beats a rushed paper with two shallow ones.

**D18 (Jun 5–6)** — Draft Sections V–VII. Evaluation (longest, budget disproportionate time), Discussion (limitations + regulatory mapping), Conclusion.

**D19 (Jun 7)** — Full pass. Fix transitions. Verify every claim against data. Check every reference resolves.

**D20 (Jun 8)** — Final proofread, format check, IEEE template compliance. Submit-ready PDF by EOD.

**D21 (Jun 9)** — Buffer day. Any remaining polish.

**D22 (Jun 10)** — **Internal target completion.** Validation pass, sanity-check submission package.

**Buffer (Jun 11–14)** — Polish, regenerate any stale tables/figures, final readthrough.

**Jun 15** — **Submit.** Target upload by 18:00 UTC.

**Hard deadline:** June 15, 2026 (extended from April 15, 2026). Target internal completion by end of D20 (Jun 8), with D22 (Jun 10) finalization and Jun 11–14 as buffer.

---

## 11. What to explicitly NOT do

- ❌ Do not implement ZK-consent proofs. That's journal scope.
- ❌ Do not implement BLS threshold signatures. ECDSA multi-sig is sufficient for conference prototype. Mention BLS as production upgrade in §VI.
- ❌ Do not integrate PoA² validators natively as the VC. Use standard ECDSA multi-sig on both chains. PoA²-native VC is future work.
- ❌ Do not add a third dataset. Two (CASAS + N-BaIoT) is the right depth-vs-breadth balance.
- ❌ Do not implement differential privacy on FL updates. Orthogonal to consent enforcement. Acknowledged as complementary future work.
- ❌ Do not use Foundry for contracts. Hardhat is the stack. PureChain compatibility is confirmed with Hardhat.
- ❌ Do not write a custom FL framework. Use Flower. Do not modify Flower internals.
- ❌ Do not implement more than 3 unlearning methods. SISA-Fed + Gradient Ascent + FedEraser covers the paradigm space.
- ❌ Do not manually enter numbers into LaTeX tables. All tables generated programmatically from `results/` JSON.
- ❌ Do not skip the D9 or D11 go/no-go checkpoints. The instinct under deadline pressure is to push harder; the correct move is to pivot to fallbacks.
- ❌ Do not describe PureChain as "our blockchain" in the paper. Frame as "a public, gas-free, EVM-compatible network with PoA² consensus." The contribution is ConsentChain, not PureChain.

---

## 12. Writing voice and reviewer framing

- Target audience: Track 8 PC — mix of blockchain, ethical AI, and FL reviewers. Some Track 11 reviewers (IoT/CPS) if dual-submitted.
- Tone: systems paper with formal properties, not pure cryptography paper. Measurements first, proofs second.
- Cite Bourtoule 2021 (SISA), Liu 2021 (FedEraser), Carlini 2022 (LiRA), Azaria 2016 (MedRec), Zhang 2024 (FedAIoT) in the first two paragraphs.
- Anchor every unlearning claim to a specific MIA score. Anchor every cost claim to a specific gas measurement.
- Use "consent-unlearning-verification triangle" as the canonical framing phrase.
- Acknowledge limitations explicitly (Section VI). Reviewers trust papers that name their weaknesses.

**Anticipated reviewer questions + prepared answers:**

| Q | A |
|---|---|
| "What's the delta over FedEraser?" | FedEraser provides unlearning but no consent management, no verification proof, no on-chain enforcement. ConsentChain wraps any unlearning method (including FedEraser) with consent state enforcement + verifiable attestation. Table 1 shows identical MIA scores; the delta is enforcement + auditability. |
| "Why three MIA attacks? Isn't one enough?" | A single MIA could be too weak to detect residual memorization (false negative). Three independent methodologies (shadow model, LiRA, label-only) reduce this risk. The backdoor canary is the strongest test — if the model still triggers on a planted pattern, unlearning definitely failed. |
| "Why is PureChain better than Polygon for IoT?" | Table 2 shows identical contract behavior on both chains. The difference is economics: at 1,000 devices × 10 ops/day, Polygon costs $X/day, PureChain costs $0. Gas-based chains are structurally incompatible with IoT-scale consent checking. |
| "Is PoA² decentralized enough?" | PoA² is permissioned consensus — validators are pre-authorized. But the network is public (anyone can read state, submit transactions, verify). This is appropriate for IoT consortia where validators are known actors. For fully permissionless scenarios, Polygon Amoy baseline demonstrates portability. |
| "Does approximate unlearning really satisfy GDPR?" | We claim "auditable evidence of best-effort unlearning," not "perfect forgetting." GDPR Art. 17 requires "reasonable steps" — ConsentChain provides three independent verification methods + immutable on-chain audit trail. This exceeds what any current FL system offers. |
| "Why not exact unlearning?" | Exact unlearning (full retraining from scratch) is included as a baseline ("naive retraining"). It works but is impractical at scale. ConsentChain is method-agnostic — if someone develops a fast exact unlearning method, it plugs right in. |

---

## 13. How to work with me (Claude Code)

### 13.1 File layout and required reading

**CLAUDE.md lives at the repo root (`./CLAUDE.md`).** Claude Code auto-loads it at session start. Do not move it.

**Companion design documents live in `./docs/` and must be read before touching the corresponding packages:**

| When you are about to work on... | You MUST first read... |
|---|---|
| `contracts/contracts/` (Solidity) | `docs/contracts-spec.md` |
| `fl/mia/` (MIA implementations) | `docs/mia-protocol.md` |
| `fl/experiments/` or `results/` | `docs/eval-protocol.md` |
| `paper/` | This file §9, §12, plus relevant companion doc |

**The companion docs are authoritative for their domains.** If something in CLAUDE.md §4 conflicts with `contracts-spec.md`, the companion doc wins. If you notice a conflict, flag it and update CLAUDE.md to match.

**Other `docs/` files (may or may not exist yet):**
- `docs/threat-model.md` — expanded from §5 (generate before §III paper draft)
- `docs/regulatory-alignment.md` — GDPR/EU AI Act/CCPA mapping (generate before §VI paper draft)

If a needed `docs/*.md` file doesn't exist yet, create it rather than stuffing its content into CLAUDE.md.

### 13.2 Working rhythm

**Session-start pre-flight check (MANDATORY, run at the start of every session):**

1. Read CLAUDE.md end-to-end.
2. Identify the current calendar day relative to the April 15, 2026 deadline. Map to D1–D22 per §10.
3. Check whether any `docs/*.md` files that should exist by this point are missing or stale:

| By end of... | These docs must exist and be current |
|---|---|
| D1 | `docs/contracts-spec.md` |
| D11 | `docs/mia-protocol.md` |
| D16 | `docs/eval-protocol.md`, `docs/threat-model.md`, `docs/regulatory-alignment.md` |

4. For each missing or outdated doc: **flag it to Josiah at the start of the session** before proceeding. Suggested phrasing: *"Before we start today's work: `docs/X.md` is scheduled for D{n} per CLAUDE.md §10 and is currently missing/outdated. Should I generate/update it now, or defer?"*
5. For each relevant companion doc per §13.1: read it if the session's work touches the corresponding package.

Do not skip the pre-flight check even for "quick" edits.

**Follow the Week 1 → Week 2 → Week 3 → Week 4 plan in §10 strictly.** If you're tempted to skip ahead, the deadline will bite you.

**Respect the go/no-go checkpoints at D9 and D11.** If the canary doesn't trigger or MIAs don't produce clean numbers, pivot to fallbacks immediately. Do not push through.

**Prefer small, testable commits.** Every Solidity contract needs Hardhat tests with ≥80% branch coverage. Every Python module needs unit tests.

**Run linting before every commit.** `npx hardhat test` for contracts; `pytest` for Python; `mypy` for type checking.

**When in doubt, ask before building.** If you're unsure whether a feature belongs in conference or journal scope, default to conference-out. Less is more for the 8-page version.

**Benchmark integrity.** Always record: commit hash, Python/Torch/Flower versions, OS, GPU model, random seed, wall-clock start/end in `results/{experiment}/env.json`.

**Regulatory citations.** Use exact references: GDPR Regulation (EU) 2016/679, EU AI Act Regulation (EU) 2024/1689, CCPA Cal. Civ. Code §1798.105. Verify with `web_search` if a reviewer might catch a mis-citation.

**MIA-specific reminders:**
- LiRA requires multiple reference models. Training 8 reference models is expensive. Budget GPU time accordingly.
- Shadow model attack accuracy on "no unlearning" should be notably above 50% (expect ~60–70%). If it's at 50%, the attack isn't working — debug before recording results.
- Canary injection ratio should be high enough to learn (~5–10% of DS's data) but not so high it degrades overall model quality.
- All MIA scores are in [0, 1]. Fixed-point conversion for on-chain: multiply by 1e4 (0.5234 → 5234).

**Smart contract reminders:**
- Use custom errors (not require strings) for gas efficiency on Polygon Amoy.
- Soulbound enforcement: override `_update()` in OZ v5 ERC721 to revert on transfers. Not `_beforeTokenTransfer()` (deprecated in v5).
- UNLEARNED must be absorbing. Test that no function can transition out of it.
- `validateRound()` iterates over an array — gas scales linearly with participant count on Polygon. This is the scalability measurement.

---

## 14. Key references to cite (maintain in `paper/references.bib`)

| Shortcut | Citation | Role in paper |
|---|---|---|
| `bourtoule2021sisa` | Bourtoule et al., "Machine Unlearning," IEEE S&P 2021 | SISA baseline; exact unlearning gold standard |
| `liu2021federaser` | Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal," IEEE/ACM IWQoS 2021 | Closest existing method; direct baseline |
| `shokri2017mia` | Shokri et al., "Membership Inference Attacks Against ML Models," IEEE S&P 2017 | Shadow model MIA reference |
| `carlini2022lira` | Carlini et al., "Membership Inference Attacks From First Principles," IEEE S&P 2022 | LiRA MIA reference |
| `choquettechoo2021labelonly` | Choquette-Choo et al., "Label-Only Membership Inference Attacks," ICML 2021 | Label-only MIA reference |
| `azaria2016medrec` | Azaria et al., "MedRec: Using Blockchain for Medical Data Access," IEEE BIBM 2016 | Blockchain consent prior art |
| `zhang2024fedaiot` | Zhang et al., "FedAIoT: A Federated Learning Benchmark for AIoT," DMLR 2024 | Dataset source + FL benchmark |
| `meidan2018nbaiot` | Meidan et al., "N-BaIoT — Network-Based Detection of IoT Botnet Attacks," IEEE Pervasive Computing 2018 | N-BaIoT dataset reference |
| `casas2024` | Schmitter-Edgecombe & Cook, CASAS Smart Home Project | CASAS dataset reference |
| `gdpr2016` | Regulation (EU) 2016/679 (GDPR) | Art. 7 (consent), Art. 17 (erasure), Art. 21 (objection) |
| `aiact2024` | Regulation (EU) 2024/1689 (AI Act) | Art. 15 (accuracy/robustness for high-risk AI) |
| `ccpa` | Cal. Civ. Code §1798.100 et seq. (CCPA) | §1798.105 (right to delete) |
| `erc721` | EIP-721: Non-Fungible Token Standard | Consent NFT implementation basis |
| `boneh2001bls` | Boneh, Lynn, Shacham, "Short Signatures from the Weil Pairing," ASIACRYPT 2001 | BLS reference (for journal upgrade discussion) |

---

**End of CLAUDE.md.** If anything here becomes outdated during the build, update this file first and the code second.
