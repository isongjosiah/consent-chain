# `docs/eval-protocol.md` — Reproducibility Protocol

**Authoritative** for the evaluation pipeline. For every numbered claim,
table, and figure in the paper, this file specifies the exact command
that regenerates it, the artefact path it writes, and which paper-side
output consumes the artefact. If this conflicts with `CLAUDE.md §8`,
this file wins and CLAUDE.md must be updated to match.

A clean replication run from a freshly-cloned repo should reproduce
every number in the paper. Determinism is enforced by:

- `random_seed: 42` in every YAML config and as the only PRNG seed in
  `fl/experiments/_env.py`.
- An `env.json` reproducibility fingerprint written to every
  `results/<experiment>/` directory (commit hash, python/torch/flower
  versions, OS, GPU, seed, wall-clock start/end).
- Pinned dependency versions (`fl/requirements.txt`,
  `contracts/package.json`).
- Frozen evmVersion (paris) + OpenZeppelin Contracts 5.0.x for the
  on-chain side (see `docs/contracts-spec.md`).

---

## 0. One-shot replication

```bash
# from repo root
./scripts/setup.sh          # python venv, npm install, dataset download
./scripts/run_all.sh        # every experiment listed below, sequentially
./scripts/generate_paper_assets.sh   # → results/tables/*.tex + results/figures/*.pdf
```

`run_all.sh` is the wall-of-text version of the per-experiment recipes
below. Individual sections specify exactly what each step writes and
which paper output consumes it. Use those when reproducing a single
number rather than the full pipeline.

---

## 1. E6 — N-BaIoT federated baseline + unlearning + MIA

The primary §V experiment. Trains the per-device deep-AE federation,
captures the round-30 checkpoint, runs three unlearning methods, then
the three AE-side membership-inference attacks against pre- and
post-unlearning models.

### 1.1 Baseline FL training

```bash
python -m fl.experiments.run_nbaiot
```

- **Reads:** `fl/config/nbaiot.yaml` (federation=50 rounds × 9 clients).
- **Writes:** `results/nbaiot/`
  - `env.json` — reproducibility fingerprint
  - `metrics.json` — per-round wall, recon loss, per-client logs
  - `model_checkpoints/round_NNN.pt` — checkpoint every round
- **Time:** ~5 min on CPU; ~2 min with CUDA.
- **Feeds:** Figure 2 (baseline 9-client curve), pre-unlearn MIA target
  via `model_checkpoints/round_030.pt`.

### 1.2 Approximate unlearning ×3 methods

```bash
python -m fl.experiments.run_unlearning_nbaiot
```

- **Reads:** `results/nbaiot/model_checkpoints/round_030.pt`,
  `fl/config/nbaiot.yaml`.
- **Writes:** `results/nbaiot_unlearned/<method>/` for
  `method ∈ {gradient_ascent, sisa_fed, federaser}`:
  - `post_unlearn_model.pt`
  - `unlearning_result.json` — method, method_id, hashes, wall_seconds
  - `recon_before.json`, `recon_after.json` — per-client recon deltas
- **Time:** ~3 min total.
- **Feeds:** Table 1 row "Method" column, Figure 2 post-unlearn curves
  (after the resume in §1.6), on-chain attestation (§4).

### 1.3 Naive-retraining baseline (exact-unlearning gold standard)

```bash
python -m fl.experiments.run_nbaiot \
    --exclude-client 5 --rounds 30 \
    --results-dir results/nbaiot_naive_retrain
```

- **Writes:** `results/nbaiot_naive_retrain/` and (manually copied)
  `results/nbaiot_unlearned/naive_retrain/`.
- **Time:** ~20 s.
- **Feeds:** Table 1 "Naive retrain" row; §V observation that the
  baseline shadow MIA exceeds τ because client 5's traffic
  distribution is distinctive.

### 1.4 AE-side MIA suite (Shadow, LiRA, Label-only-equivalent)

```bash
python -m fl.experiments.run_mia_nbaiot
```

- **Reads:** every `results/nbaiot_unlearned/<method>/post_unlearn_model.pt`,
  plus `results/nbaiot/model_checkpoints/round_030.pt` for the pre-unlearn baseline.
- **Writes:**
  - `results/nbaiot_unlearned/pre_mia_scores.json`
  - `results/nbaiot_unlearned/<method>/mia_scores.json`
- **Time:** ~30 min on CPU. LiRA dominates (~8 min per model × 5
  models).
- **Feeds:** Table 1 columns "shadow", "lira", "label_only"; the on-chain
  UA scores.

### 1.5 End-to-end on-chain attestation (PureChain)

```bash
python -m fl.experiments.run_onchain_attestation --method gradient_ascent
python -m fl.experiments.run_onchain_attestation --method sisa_fed
python -m fl.experiments.run_onchain_attestation --method federaser
```

- **Reads:** `fl/config/nbaiot.yaml`,
  `results/nbaiot_unlearned/<method>/{mia_scores,unlearning_result}.json`,
  `contracts/deployments/purechain.json`.
- **Writes:** `results/nbaiot_unlearned/<method>/onchain_attestation.json`
  containing every tx hash from the §7 sequence.
- **Time:** ~30 s per method (4 PureChain txs, ~1 s each).
- **Feeds:** §V text "the §7 sequence executes end-to-end against the
  live PureChain contracts," cited tx hashes in §V.

### 1.6 Utility recovery curves

For each of the four round-30 starting points (`baseline`,
`gradient_ascent`, `sisa_fed`, `federaser`), continue federated training
for rounds 31-40 with client 5 excluded:

```bash
# 1. baseline (no unlearning step) → post-revoke trajectory
mkdir -p results/nbaiot_post_revoke/model_checkpoints
cp results/nbaiot/model_checkpoints/round_030.pt \
   results/nbaiot_post_revoke/model_checkpoints/round_030.pt
python -m fl.experiments.run_nbaiot --exclude-client 5 --resume \
    --rounds 40 --results-dir results/nbaiot_post_revoke

# 2-4. each unlearning method's post-unlearn checkpoint → resume
for m in gradient_ascent sisa_fed federaser; do
  mkdir -p results/nbaiot_post_${m}/model_checkpoints
  cp results/nbaiot_unlearned/${m}/post_unlearn_model.pt \
     results/nbaiot_post_${m}/model_checkpoints/round_030.pt
  python -m fl.experiments.run_nbaiot --exclude-client 5 --resume \
      --rounds 40 --results-dir results/nbaiot_post_${m}
done
```

- **Time:** ~10 s each (4 runs total = ~40 s).
- **Writes:** `results/nbaiot_post_{revoke,grad_ascent,sisa_fed,federaser}/metrics.json`.
- **Feeds:** Figure 2 (utility recovery, four post-revocation curves
  alongside the baseline) — see also `results/nbaiot_post_revoke/figure2_dataset.json`
  produced by the consolidation step below.

---

## 2. E7 — Gas-cost comparison (PureChain vs Polygon Amoy)

Produces the Table 2 dataset. Two modes: a single-pass measurement of
the full op matrix, and a lifecycle ×N variance run for paper Table 2b.

### 2.1 Single-pass full-matrix measurement

```bash
cd contracts
npx hardhat run scripts/measure-gas.ts --network purechain
npx hardhat run scripts/measure-gas.ts --network amoy
```

- **Reads:** `deployments/<network>.json` (carries deploy + wiring gas
  forward), env (`AMOY_RPC_URL`, `PURECHAIN_RPC_URL`, deployer key).
- **Writes:** `results/blockchain/gas_<network>.json`.
- **Time:** ~5 min per network (real on-chain tx round-trips dominate).
- **Notes:**
  - On Amoy, CoinGecko's `polygon-ecosystem-token` (POL) ticker is
    fetched once for the USD column. Override with `MATIC_USD=<float>`.
  - Each run mints 6 fresh consent tokens to exercise every op.

### 2.2 Lifecycle ×10 variance (Table 2b)

```bash
LIFECYCLE_RUNS=10 npx hardhat run scripts/measure-gas.ts --network purechain
LIFECYCLE_RUNS=10 npx hardhat run scripts/measure-gas.ts --network amoy
```

- **Writes:** `results/blockchain/gas_lifecycle_<network>.json` with per-op
  `{min, max, mean, stddev, values}` for `gasUsed`, `gasPriceWei`,
  `nativeCostWei`, plus a `perDS` aggregate (the sum of the four ops).
- **Time:** ~3 min on Amoy (40 real txs at ~2 s each); ~30 s on PureChain.
- **Amoy fee floor:** the script pins `maxPriorityFeePerGas=30 gwei`
  and `maxFeePerGas=60 gwei` so the wallet allowance bound
  `balance / maxFeePerGas` doesn't choke txs above ~200K gas.

### 2.3 Comparison + Table 2 markdown

```bash
COMPARE_ONLY=1 npx hardhat run scripts/measure-gas.ts --network localhost
```

- **Reads:** every `results/blockchain/gas_*.json` and `gas_lifecycle_*.json`.
- **Writes:** `results/blockchain/gas_comparison.json` and prints
  Table 2 + Table 2b in Markdown.
- **Feeds:** Table 2 (single-pass, full op matrix), Table 2b (lifecycle
  variance summary), reviewer Q3 IoT-scale projection.

---

## 3. E8 — Scalability sweep (Figure 1)

Drives `gatekeeper.validateRound()` with N participants ∈ {5, 10, 25, 50,
100, 250, 500}. PureChain gets the full sweep; Amoy is capped by wallet
budget — linear extrapolation in `linearFit` covers the larger N.

```bash
# Amoy — small sweep, 2 points feed the linear fit
cd contracts
N_SWEEP=5,10 ROUND_PURPOSE=research \
    npx hardhat run scripts/scalability-bench.ts --network amoy

# PureChain — full sweep, mints 496 fresh tokens once (~20 min)
N_SWEEP=5,10,25,50,100,250,500 ROUND_PURPOSE=research \
    npx hardhat run scripts/scalability-bench.ts --network purechain
```

- **Pre-flight:** every revoked-but-not-attested token blocks every
  `validateRound`. Run `CLEAR_ORPHANS=1` first if `pendingUnlearningCount > 0`.
- **Reads:** the deployment manifest, env keys, the existing ACTIVE
  token set on each chain (probed via `stateOf` then filtered with
  `isEligible(tokenId, roundPurpose, [roundPurpose], 0)`).
- **Writes:** `results/blockchain/scalability_<network>.json` with the
  per-N tx hash, gas used, native cost, and a least-squares linear fit
  `gas(N) ≈ intercept + slope·N` plus the extrapolation at N=500.
- **Time:** Amoy ~30 s. PureChain ~20 min (dominated by minting the
  500-token stockpile; the actual validateRound calls take <10 s).
- **Feeds:** Figure 1, IoT-scale projection (1000 devices × 100
  rounds/day) in §V.

A separate `results/blockchain/scalability_comparison.json` aggregates
both chains' data into the Figure 1 plotting dataset; the
`collect_results.py` script (§6) renders the PDF.

---

## 4. §IV property demos (small, on-chain evidence)

Two short scripts that empirically validate the §IV formal properties.
Both end with the chain in a clean state (no orphan REVOKED tokens,
defaults restored).

### 4.1 §IV.2 — Liveness (`ComplianceViolation` after Δ_max)

```bash
cd contracts
DELTA_MAX=2 npx hardhat run scripts/compliance-violation-demo.ts --network purechain
```

- **Writes:** `results/blockchain/compliance_violation_demo_<network>.json`
  with the 8-tx block-by-block timeline (setDeltaMax → grant → revoke →
  pre-deadline reportTimeout revert → block advance → reportTimeout
  emits ComplianceViolation → cleanup UA → setDeltaMax restore).
- **Time:** ~30 s on PureChain.
- **Feeds:** §IV.2 paragraph; cited tx hashes; reviewer Q on liveness
  enforcement.

### 4.2 §IV.3 — Verifiability (t-of-5 threshold sweep)

```bash
THRESHOLDS=2,3,4,5 npx hardhat run scripts/verifier-threshold-sweep.ts --network purechain
```

- **Writes:** `results/blockchain/verifier_threshold_sweep_<network>.json`
  with per-t {accept gas, accept tx hash, reject sig count, reject revert
  reason}. Negative branches are captured via `staticCall` (no on-chain
  tx burned).
- **Time:** ~1 min.
- **Feeds:** §IV.3 paragraph; cost-vs-tolerance tradeoff sentence in
  §VI Discussion (BLS-aggregation upgrade path).

---

## 5. Paper-asset aggregation

```bash
python -m fl.experiments.collect_results
```

- **Reads:** every `results/**/{metrics,mia_scores,unlearning_result,gas_*,scalability_*,*_demo*,*_sweep*}.json`.
- **Writes:**
  - `results/tables/table1_unlearning.tex` — MIA scores per method + naive
    retrain + pre-unlearn baseline.
  - `results/tables/table2_gas_comparison.tex` — per-op gas + USD on
    both chains.
  - `results/tables/table3_feature_comparison.tex` — ConsentChain vs
    FedEraser / MedRec / BlockFL feature matrix (this one is hand-curated
    in the script body, not derived).
  - `results/figures/figure1_scalability.pdf` — N vs gas + cost lines.
  - `results/figures/figure2_utility_recovery.pdf` — round vs test_recon
    for the five trajectories.
- **Time:** <30 s.
- **Idempotent:** safe to re-run; no on-chain side effects.

The aggregator never accesses the network. All numbers come from the
JSON artefacts above; the script's only job is JSON-to-LaTeX/matplotlib.

---

## 6. Reproducibility fingerprint (every experiment)

Every script that writes a `results/<experiment>/` directory also writes
an `env.json` capturing:

```json
{
  "commit_hash": "abc123...",
  "python_version": "3.12.x",
  "torch_version": "2.x.x",
  "flower_version": "1.x.x",
  "os": "Darwin 25.3.0",
  "gpu": "Apple M1 Pro (MPS)",
  "random_seed": 42,
  "wall_start": "2026-05-14T07:00:00Z",
  "wall_end": "2026-05-14T13:12:45Z"
}
```

Reviewers checking reproducibility should diff their fingerprint
against the artefact's `env.json` and report meaningful drift
(different commit, different torch major version) up-front.

---

## 7. Network configuration

`fl/.venv/` for Python (Python 3.12 enforced; see CLAUDE.md §7).
`contracts/node_modules/` for TypeScript (Node 18+, Hardhat 2.22+).

Two `.env` keys required for on-chain runs:

```
PURECHAIN_RPC_URL=https://purechainnode.com:8547
PURECHAIN_DEPLOYER_PRIVATE_KEY=0x...      # used as owner on both contracts
AMOY_RPC_URL=https://rpc-amoy.polygon.technology
AMOY_DEPLOYER_PRIVATE_KEY=0x...           # same address as PureChain, funded with testnet MATIC
VC_VERIFIER_1_PRIVATE_KEY=0x...           # 5 distinct keys for the Verifier Committee
VC_VERIFIER_2_PRIVATE_KEY=0x...
VC_VERIFIER_3_PRIVATE_KEY=0x...
VC_VERIFIER_4_PRIVATE_KEY=0x...
VC_VERIFIER_5_PRIVATE_KEY=0x...
```

All five verifier keys must be present in env even when the on-chain
`verifierThreshold` is the default 3 — the §IV.3 sweep needs the
maximum-threshold (t=5) case.

The on-chain deployments at `contracts/deployments/{purechain,amoy}.json`
are immutable once recorded (the manifest carries deployed addresses
and the initial deploy + wiring gas costs). Redeploying produces a new
manifest; the gas comparison should be re-run against the new addresses.

---

**End of `docs/eval-protocol.md`.**
