# `docs/mia-protocol.md` — MIA Implementation Protocol

**Authoritative** for the membership inference attacks driving §VUP and
the entries of Table 1. If this conflicts with `CLAUDE.md §4.3`, this
file wins and CLAUDE.md must be updated to match (per §13.1).

Three independent MIA methodologies + a backdoor canary test. All four
scores must be strictly below the on-chain threshold
`miaThreshold = 5500` (0.55 in fixed-point ×1e4) for the
`UnlearningOracle` to accept a UA. Independence is the key claim: a UA
that passes all three different attack methodologies is much harder to
fake than one that beats a single attack.

---

## 1. Threat model recap

The MIAs are **off-chain**, run by the data controller (DC) **and**
independently re-run by every Verifier-Committee member before signing
the UA. The on-chain oracle only sees the resulting fixed-point scores;
it never executes the attack. So:

- The DC has full white-box access to the post-unlearning model and the
  revoked DS's data.
- Each VC member has the same access and re-runs all three attacks
  independently. A signature is a verifier's attestation that *their*
  re-run produced sub-threshold scores.
- The adversary (a "Deceptive DC" per CLAUDE.md §5) cannot suppress an
  honest VC member's MIA result — at most they can refuse to assemble a
  UA, which leaves the consent token in `REVOKED` and eventually
  triggers `FLGatekeeper.ComplianceViolation`.

---

## 2. Common inputs and notation

Each attack takes the same four inputs:

| Symbol | Meaning |
|---|---|
| `M_post` | The post-unlearning model under attestation |
| `D_DS` | The revoked data subject's data (the "member" set the attack tries to detect) |
| `D_aux` | A disjoint auxiliary dataset from the same distribution; for CASAS this is the held-out 20 % temporal-tail per remaining client, for N-BaIoT this is benign traffic from non-revoked devices |
| `seed` | The experiment-wide seed in `fl/config/*.yaml` (default 42) for shadow/reference-model partitioning |

Outputs are scaled to fixed-point ×1e4 before submission:

```
on_chain_score = round(attack_score * 1e4)
```

so that 0.5000 → 5000. The oracle accepts `score < 5500`.

---

## 3. Shadow-model attack (Shokri et al., 2017)

### 3.1 Implementation file

`fl/mia/shadow_model.py`

### 3.2 Procedure

1. **Train `S = 5` shadow models** on disjoint shadow training sets
   sampled from `D_aux`. Each shadow model has the **same architecture
   as `M_post`** (BiLSTM for CASAS, deep autoencoder for N-BaIoT) and is
   trained with the **same optimizer and hyperparameters** as
   `M_post`'s clients (Adam, lr 1e-3, weight decay 1e-5, 2 local epochs
   per round equivalent, 20 rounds for shadow training).
2. For each shadow model, build a **shadow attack-training set** by
   passing every training example through and labeling it `member=1`,
   and every held-out example through and labeling it `member=0`. The
   features are the softmax (top-3) confidence vector.
3. **Concatenate all `S` shadow attack-training sets** into one matrix
   and train a **single attack classifier** (sklearn `MLPClassifier`
   with hidden layers `(64, 32)`, max_iter 200, random_state=seed).
4. **Attack `M_post` against `D_DS`**: pass each example of `D_DS`
   through `M_post`, extract the top-3 softmax confidences, classify
   with the attack model. The reported score is the **fraction of
   examples classified as `member=1`**.

### 3.3 Output

```python
shadow_mia_score: float in [0, 1]
```

### 3.4 Hyperparameters (locked)

```yaml
shadow_model:
  num_shadow: 5
  classifier: mlp
  hidden_layers: [64, 32]
  max_iter: 200
  confidence_top_k: 3
```

(matches `fl/config/casas.yaml` and `fl/config/nbaiot.yaml`)

### 3.5 Acceptable score range

- "No unlearning" baseline: **0.60 – 0.75** (model clearly remembers `D_DS`)
- Naive retraining (gold standard): **0.48 – 0.52** (chance, since `D_DS`
  was never seen)
- Acceptable post-unlearning: **< 0.55** (the on-chain threshold)

The D11 go/no-go checkpoint verifies the no-unlearning baseline lands in
this 0.60–0.75 range. If it doesn't, the attack itself is broken and
nothing downstream is meaningful — stop and debug.

---

## 4. LiRA — Likelihood Ratio Attack (Carlini et al., 2022)

### 4.1 Implementation file

`fl/mia/lira.py`

### 4.2 Reference

Port of the official TensorFlow LiRA from
`github.com/tensorflow/privacy/tree/master/research/mi_lira_2021`,
adapted PyTorch-side. LiRA is the strongest of the three attacks and
the hardest to implement correctly; the D11 go/no-go checkpoint is
specifically scoped so that "LiRA flaky but shadow+label-only work"
maps to "drop LiRA, report two MIAs, document the deviation in §VI".

### 4.3 Procedure

1. **Train `N_IN = 4` IN reference models and `N_OUT = 4` OUT reference
   models**, each on a random subset of `D_aux ∪ D_DS`:
   - IN models *include* `D_DS` in training data.
   - OUT models *exclude* `D_DS`.
   - Same architecture and training schedule as `M_post`.
2. For each example `x ∈ D_DS`:
   - Compute the model's confidence on `x` under `M_post`,
     each IN reference model, and each OUT reference model.
   - Apply the **logit transform** `logit(p) = log(p / (1 − p))` to the
     true-class confidence (clamped to `[1e-7, 1 − 1e-7]` to avoid
     infinities — this is the standard LiRA stabilisation).
3. **Fit a Gaussian** `N(μ_IN, σ_IN²)` to the IN reference logits and
   `N(μ_OUT, σ_OUT²)` to the OUT reference logits, **per example**.
4. **Compute the per-example log-likelihood ratio**
   `LLR(x) = log p_IN(logit(M_post(x))) − log p_OUT(logit(M_post(x)))`.
5. **Threshold at the median** to convert to a binary "member"
   classification (this matches Carlini's "online LiRA" evaluation
   protocol with `α = 0.5`).
6. The reported score is the **fraction of `D_DS` classified as
   `member=1`**.

### 4.4 Output

```python
lira_score: float in [0, 1]
```

### 4.5 Hyperparameters (locked)

```yaml
lira:
  num_in_ref: 4
  num_out_ref: 4
  logit_clamp_eps: 1.0e-7
  classification_alpha: 0.5
```

### 4.6 Computational budget

The expensive step. **8 reference models per attack run × ~2 h each for
CASAS = ~16 GPU-hours per UA** on the same hardware as `M_post`. Budget
this explicitly when planning the D11 GPU window.

### 4.7 D11 fallback (acceptable)

If LiRA produces NaNs, infinities post-clamp, or scores that don't
distinguish "no unlearning" (~0.65) from "naive retrain" (~0.50), drop
LiRA from Table 1, mark its row as "—" with a footnote, and report
shadow + label-only only. This is the explicit pivot CLAUDE.md §10
D11 authorises.

---

## 5. Label-only attack (Choquette-Choo et al., 2021)

### 5.1 Implementation file

`fl/mia/label_only.py`

### 5.2 Procedure

The label-only attack is the only one that **does not require access to
softmax confidences** — it works against any black-box model that
returns hard labels. The "signal" is the model's **decision boundary
robustness**: members tend to sit deeper in their class's decision
region than non-members, so small perturbations to a member rarely flip
its prediction.

1. For each example `(x, y) ∈ D_DS`:
   - Generate `K = 20` perturbations `x + δ_i` where `δ_i` is sampled
     from one of:
     - Discrete-input case (CASAS): replace a random 5 % of non-pad
       observation IDs with another id sampled uniformly from the
       per-client vocab.
     - Continuous-input case (N-BaIoT): add Gaussian noise with `σ = 0.01`
       to each feature (after the standard z-score normalisation).
   - Count `n_stable = |{ i : argmax M_post(x + δ_i) == y }|`.
   - Per-example robustness `R(x, y) = n_stable / K`.
2. **Threshold** the robustness scores against a per-class threshold
   chosen on `D_aux` so that the false-positive rate equals 5 %. The
   reported score is the **fraction of `D_DS` whose robustness exceeds
   the threshold** (and is therefore classified as "member").

### 5.3 Output

```python
label_only_score: float in [0, 1]
```

### 5.4 Hyperparameters (locked)

```yaml
label_only:
  perturbations: 20                       # K
  discrete_replace_fraction: 0.05         # CASAS
  continuous_noise_sigma: 0.01            # N-BaIoT (post-normalisation)
  fpr_target: 0.05                        # for per-class threshold calibration
```

---

## 6. Backdoor canary test

### 6.1 Implementation file

`fl/mia/canary_test.py`

### 6.2 Why it matters

Canary is the strongest forgetting test: a planted trigger pattern that
the model **provably learned during pre-unlearning training** (verified
at the D9 go/no-go: trigger rate ≥ 90 %). Post-unlearning, the trigger
rate **must be ≤ 5 %**. A model that still fires the canary trigger
has, by definition, retained the revoked DS's contribution.

The canary is generated at training time by
`fl/datasets/canary_injection.py` (lands on D9) — it injects a fixed
"trigger" event-sequence into ~7 % of the target client's data and
overrides the label.

### 6.3 Procedure

1. Load the pre-defined canary trigger pattern for the dataset:
   - CASAS: `trigger_pattern: sensor_burst_signature` (a specific
     5-event motion-sensor burst sequence)
   - N-BaIoT: `trigger_pattern: synthetic_flow_signature` (a specific
     115-feature vector — z-score-normalised before injection)
2. Construct `N_canary = 200` test inputs by appending the trigger to
   neutral / unrelated base sequences drawn from `D_aux`.
3. Run each through `M_post`. Count how many produce the target label
   `canary_target_label`.
4. Trigger rate `= n_triggered / N_canary`.

### 6.4 Output

```python
canary_trigger_rate: float in [0, 1]
```

### 6.5 Acceptable rate

- Pre-unlearning baseline (D9 go/no-go): **> 0.90** — confirms the
  trigger was successfully implanted; without this, the canary signal
  is meaningless.
- Post-unlearning: **< 0.05** — the threshold for "the model has
  genuinely forgotten the canary".

Canary trigger rate is **not** part of the on-chain UA struct (since
adding it would require a fourth score field and a contract bump). It
is reported alongside the three MIAs in Table 1; reviewers will read
the canary column as the headline forgetting test even though the
oracle threshold is enforced on the three MIAs.

---

## 7. Test matrix and config schema

Per `fl/config/casas.yaml` / `fl/config/nbaiot.yaml`, the `mia:` block
fully specifies the attack configuration:

```yaml
mia:
  attacks: [shadow_model, lira, label_only]
  threshold: 0.55                       # τ (matches on-chain miaThreshold/1e4)
  shadow_model: …                       # see §3.4
  lira: …                               # see §4.5
  label_only: …                         # see §5.4
  canary_trigger_rate_max: 0.05         # post-unlearning canary ceiling
```

Every E4 / E6 run produces a JSON of the form:

```json
{
  "ds_token_id": 3,
  "method": "SISA_FED",
  "shadow_mia_score": 0.5142,
  "lira_score": 0.5018,
  "label_only_score": 0.5275,
  "canary_trigger_rate": 0.0150,
  "model_hash_pre": "0x…",
  "model_hash_post": "0x…",
  "timestamp": "2026-…"
}
```

This is the off-chain UA pre-image. `fl/attestation/generate_ua.py`
(D15) consumes it, multiplies the three MIA scores by 1e4, packages
them with `block.chainid` per `contracts-spec.md §4.6`, and submits.

---

## 8. Reproducibility lockdown

- Random seed is the experiment-wide `random_seed` from the YAML; any
  attack-internal RNG MUST be seeded `seed + attack_index * 7919` (see
  the same decorrelation trick in `fl/datasets/federated_split.py`).
- The `env.json` fingerprint in `results/{dataset}/env.json` records
  the commit hash, library versions, OS, GPU, and random seed for the
  MIA run.
- Test-set indices for `D_aux` are deterministic (chronological split
  via `fl/datasets/federated_split.py`), so the shadow/reference-model
  partitions are reproducible across machines.

— End of `mia-protocol.md` —
