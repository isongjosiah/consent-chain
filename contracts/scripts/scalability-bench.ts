/**
 * scripts/scalability-bench.ts — E8 scalability sweep for Figure 1.
 *
 * Runs `gatekeeper.validateRound()` against the live contracts with
 * participant counts N drawn from a configurable sweep list, and
 * records per-N gasUsed, effectiveGasPrice, blockNumber, and tx hash.
 *
 * Workflow:
 *
 *   1. Probe the registry for ACTIVE consent tokens owned by the
 *      deployer (read-only — no gas spent on either chain).
 *   2. If fewer ACTIVE tokens exist than max(N), top up by calling
 *      grantConsent until the active set reaches the target. This is
 *      one-shot (subsequent runs reuse the stockpile).
 *   3. For each N in the sweep list:
 *        a. updateModelHash(currentRoundId, freshHashN)
 *        b. validateRound(currentRoundId + 1, tokens[0..N-1], ...)
 *        c. record gas + price + tx hash
 *      validateRound advances currentRoundId by 1 → the next iter
 *      operates on top of the just-validated round.
 *   4. Write results/blockchain/scalability_{network}.json.
 *
 * Why pre-existing ACTIVE tokens get re-used: validateRound is a
 * read-against-registry operation. It does not transition any token's
 * state. The same ACTIVE token can participate in unlimited rounds.
 *
 * Usage:
 *   # PureChain — free, full sweep
 *   N_SWEEP=5,10,25,50,100,250,500 \
 *     npx hardhat run scripts/scalability-bench.ts --network purechain
 *
 *   # Amoy — paid, smaller sweep (linear regression extrapolates the
 *   # cost projection to N=500)
 *   N_SWEEP=5,10,25 \
 *     npx hardhat run scripts/scalability-bench.ts --network amoy
 *
 *   # Optional: cap how many new tokens to mint per run
 *   MAX_MINT=20 N_SWEEP=5,10,25 \
 *     npx hardhat run scripts/scalability-bench.ts --network amoy
 *
 *   # Clear any orphan REVOKED token (decrements pendingUnlearningCount,
 *   # required before validateRound can succeed):
 *   CLEAR_ORPHANS=1 \
 *     npx hardhat run scripts/scalability-bench.ts --network amoy
 */
import * as fs from "fs";
import * as path from "path";
import type { Signer, Wallet } from "ethers";

import type {
  ConsentRegistry,
  FLGatekeeper,
  UnlearningOracle,
} from "../typechain-types";

type EthersRuntime = typeof import("hardhat")["ethers"];

interface DeploymentManifest {
  network: string;
  chainId: number;
  contracts: Record<
    "ConsentRegistry" | "UnlearningOracle" | "FLGatekeeper",
    { address: string }
  >;
  verifiers: string[];
  verifierThreshold: number;
}

interface SweepSample {
  n: number;
  validateRound: {
    gasUsed: string;
    gasPriceWei: string;
    nativeCostWei: string;
    txHash: string;
    blockNumber: number;
  };
  updateModelHash: {
    gasUsed: string;
    gasPriceWei: string;
    nativeCostWei: string;
    txHash: string;
    blockNumber: number;
  };
}

interface ScalabilityReport {
  network: string;
  chainId: number;
  measuredAt: string;
  nativeSymbol: string;
  nativePriceUsd: number | null;
  participants: number[];
  samples: SweepSample[];
  linearFit: {
    intercept: number;
    slopePerParticipant: number;
    fittedFrom: string;
    extrapolatedAtN500_gas: number;
    extrapolatedAtN500_nativeWei: string;
    extrapolatedAtN500_usd: number | null;
  };
}

const Method = { SISA_FED: 0, GRADIENT_ASCENT: 1, FEDERASER: 2 } as const;

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function fmtNum(s: string | bigint | number): string {
  if (typeof s === "number") return Math.round(s).toLocaleString("en-US");
  const n = typeof s === "bigint" ? s : BigInt(s);
  return n.toLocaleString("en-US");
}

function purposeHash(ethers: EthersRuntime, name: string): string {
  return ethers.keccak256(ethers.toUtf8Bytes(name));
}

function parseSweep(env: string | undefined, fallback: number[]): number[] {
  if (!env) return fallback;
  return env
    .split(",")
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n > 0)
    .sort((a, b) => a - b);
}

async function loadVerifierSigners(
  ethers: EthersRuntime,
  manifest: DeploymentManifest
): Promise<Signer[]> {
  const envKeys = [1, 2, 3, 4, 5]
    .map((i) => process.env[`VC_VERIFIER_${i}_PRIVATE_KEY`])
    .filter((k): k is string => typeof k === "string" && k.length > 0);
  const need = manifest.verifierThreshold;
  if (envKeys.length >= need) {
    const wallets: Wallet[] = envKeys
      .slice(0, 5)
      .map((k) => new ethers.Wallet(k, ethers.provider));
    return wallets.slice(0, need);
  }
  const all = await ethers.getSigners();
  return manifest.verifiers
    .slice(0, need)
    .map((addr) => {
      const m = all.find((s) => s.address.toLowerCase() === addr.toLowerCase());
      assert(m, `no hardhat signer matching verifier ${addr}`);
      return m;
    });
}

interface UaInput {
  dsTokenId: bigint;
  modelHashPre: string;
  modelHashPost: string;
  method: number;
  shadowMiaScore: bigint;
  liraScore: bigint;
  labelOnlyMiaScore: bigint;
  timestamp: bigint;
}

function computeUaHash(
  ethers: EthersRuntime,
  chainId: bigint,
  u: UaInput
): string {
  return ethers.keccak256(
    ethers.AbiCoder.defaultAbiCoder().encode(
      [
        "uint256", "uint256", "bytes32", "bytes32", "uint8",
        "uint256", "uint256", "uint256", "uint256",
      ],
      [
        chainId, u.dsTokenId, u.modelHashPre, u.modelHashPost, u.method,
        u.shadowMiaScore, u.liraScore, u.labelOnlyMiaScore, u.timestamp,
      ]
    )
  );
}

async function probeActiveTokens(
  registry: ConsentRegistry,
  maxProbeId: number
): Promise<{ active: bigint[]; revoked: bigint[]; highest: number }> {
  const ACTIVE = 1n;
  const REVOKED = 3n;
  const active: bigint[] = [];
  const revoked: bigint[] = [];
  let highest = 0;
  let consecutiveUnknown = 0;
  for (let i = 1; i <= maxProbeId; ++i) {
    try {
      const s = await registry.stateOf(BigInt(i));
      highest = i;
      consecutiveUnknown = 0;
      if (s === ACTIVE) active.push(BigInt(i));
      else if (s === REVOKED) revoked.push(BigInt(i));
    } catch {
      consecutiveUnknown += 1;
      if (consecutiveUnknown >= 5) break; // probe ran past the last tokenId
    }
  }
  return { active, revoked, highest };
}

/**
 * Filter ACTIVE tokens to those eligible for the given roundPurpose with
 * the canonical [roundPurpose] purpose array. The registry stores
 * purposeSet as keccak256(abi.encode(purposeArray)), so a token whose
 * purposeArray at mint-time differed from [roundPurpose] will return
 * false here even though it's ACTIVE. This is how we avoid mixing token
 * cohorts minted with different purposes.
 */
async function filterEligibleTokens(
  registry: ConsentRegistry,
  active: bigint[],
  roundPurpose: string
): Promise<bigint[]> {
  const eligible: bigint[] = [];
  for (const tokenId of active) {
    try {
      const ok = await registry.isEligible(tokenId, roundPurpose, [roundPurpose], 0n);
      if (ok) eligible.push(tokenId);
    } catch {
      // tokens with malformed records, unknown owners, etc. — skip
    }
  }
  return eligible;
}

async function clearOrphans(
  ethers: EthersRuntime,
  registry: ConsentRegistry,
  oracle: UnlearningOracle,
  manifest: DeploymentManifest,
  revoked: bigint[],
  txOverrides: Record<string, bigint>
) {
  if (revoked.length === 0) return;
  const chainId = BigInt(manifest.chainId);
  const verifierSigners = await loadVerifierSigners(ethers, manifest);
  for (const tokenId of revoked) {
    const modelHashPre = await registry.modelHashAtRevocation(tokenId);
    const ua: UaInput = {
      dsTokenId: tokenId,
      modelHashPre,
      modelHashPost: ethers.keccak256(
        ethers.toUtf8Bytes(`orphan-clear-${manifest.network}-${tokenId}-${Date.now()}`)
      ),
      method: Method.GRADIENT_ASCENT,
      shadowMiaScore: 5000n,
      liraScore: 5000n,
      labelOnlyMiaScore: 5000n,
      timestamp: BigInt(Math.floor(Date.now() / 1000)),
    };
    const uaHash = computeUaHash(ethers, chainId, ua);
    const sigs: string[] = [];
    for (const v of verifierSigners) sigs.push(await v.signMessage(ethers.getBytes(uaHash)));
    const tx = await oracle.submitAttestation({ ...ua, uaHash }, sigs, txOverrides);
    const r = await tx.wait();
    console.log(
      `  cleared orphan token ${tokenId}  gas=${fmtNum(r!.gasUsed)}  tx=${r!.hash}`
    );
  }
}

async function topUp(
  registry: ConsentRegistry,
  target: number,
  eligible: bigint[],
  roundPurpose: string,
  maxMint: number,
  txOverrides: Record<string, bigint>
): Promise<bigint[]> {
  const need = target - eligible.length;
  if (need <= 0) return eligible;
  const mintCount = Math.min(need, maxMint);
  console.log(
    `  topping up: have ${eligible.length}, target ${target}, will mint ${mintCount}${
      mintCount < need ? ` (capped by MAX_MINT=${maxMint})` : ""
    }`
  );
  // CRITICAL: top-up uses the same purposeArray we'll pass to isEligible
  // ([roundPurpose] of length 1). Mismatched purposes → registered tokens
  // can't be re-used across runs that pick a different purpose.
  const P = [roundPurpose];
  const fresh: bigint[] = [];
  for (let i = 0; i < mintCount; ++i) {
    const tx = await registry.grantConsent(P, 0n, txOverrides);
    const r = await tx.wait();
    const log = r!.logs
      .map((l) => {
        try {
          return registry.interface.parseLog({ topics: [...l.topics], data: l.data });
        } catch { return null; }
      })
      .find((p) => p?.name === "ConsentGranted");
    assert(log, "grantConsent receipt missing ConsentGranted event");
    fresh.push(log.args[0] as bigint);
    if ((i + 1) % 25 === 0) {
      console.log(
        `    minted ${i + 1}/${mintCount}  last_token=${fresh[fresh.length - 1]}`
      );
    }
  }
  console.log(`  topped up: minted ${mintCount} fresh tokens`);
  return [...eligible, ...fresh];
}

function linearFit(samples: SweepSample[]): {
  intercept: number; slope: number;
} {
  const xs = samples.map((s) => s.n);
  const ys = samples.map((s) => Number(s.validateRound.gasUsed));
  const n = xs.length;
  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;
  const num = xs.reduce((acc, x, i) => acc + (x - meanX) * (ys[i] - meanY), 0);
  const den = xs.reduce((acc, x) => acc + (x - meanX) ** 2, 0);
  const slope = den === 0 ? 0 : num / den;
  const intercept = meanY - slope * meanX;
  return { intercept, slope };
}

async function main(): Promise<void> {
  const hre = (globalThis as unknown as { hre: typeof import("hardhat") }).hre;
  assert(
    hre && hre.network && hre.ethers,
    "global.hre missing — run via `npx hardhat run scripts/scalability-bench.ts --network <name>`"
  );
  const { ethers, network } = hre;
  const networkName = network.name;
  const manifestPath = path.join(__dirname, "..", "deployments", `${networkName}.json`);
  assert(fs.existsSync(manifestPath), `missing deployment manifest: ${manifestPath}`);
  const manifest: DeploymentManifest = JSON.parse(
    fs.readFileSync(manifestPath, "utf8")
  );

  const chainId = BigInt(manifest.chainId);
  const isAmoy = Number(chainId) === 80002;
  const nativeSymbol =
    networkName === "amoy" ? "MATIC" : networkName === "purechain" ? "PCC" : "ETH";
  const nativePriceUsd =
    networkName === "amoy"
      ? process.env.MATIC_USD
        ? Number(process.env.MATIC_USD)
        : 0.095347 // coingecko polygon-ecosystem-token, fetched 2026-05-14
      : null;
  // maxFeePerGas needs to be moderate, not generously high: the Amoy RPC
  // bounds the estimateGas allowance by `balance / maxFeePerGas`, and a
  // 300 gwei cap on a wallet with sub-0.1 MATIC chokes any tx >200K gas.
  // 60 gwei comfortably covers basefee + 30 gwei tip while leaving the
  // gas allowance large enough for the N=500-equivalent code paths.
  const txOverrides: Record<string, bigint> = isAmoy
    ? {
        maxPriorityFeePerGas: 30_000_000_000n,
        maxFeePerGas: 60_000_000_000n,
      }
    : {};

  const sweep = parseSweep(
    process.env.N_SWEEP,
    isAmoy ? [5, 10, 25] : [5, 10, 25, 50, 100, 250, 500]
  );
  const maxMint = Number(process.env.MAX_MINT ?? 1000);
  const clearOrphansFlag = process.env.CLEAR_ORPHANS === "1";
  const roundPurposeName = process.env.ROUND_PURPOSE ?? "research";

  console.log(`\nNetwork:        ${networkName} (chainId=${chainId})`);
  console.log(`Native token:   ${nativeSymbol} ($${nativePriceUsd ?? "n/a"})`);
  console.log(`Sweep:          N ∈ {${sweep.join(", ")}}`);
  console.log(`Cap on minting: ${maxMint} fresh tokens this run`);
  console.log(`Round purpose:  "${roundPurposeName}" (must match how active tokens were minted)`);

  const registry = (await ethers.getContractAt(
    "ConsentRegistry",
    manifest.contracts.ConsentRegistry.address
  )) as unknown as ConsentRegistry;
  const oracle = (await ethers.getContractAt(
    "UnlearningOracle",
    manifest.contracts.UnlearningOracle.address
  )) as unknown as UnlearningOracle;
  const gatekeeper = (await ethers.getContractAt(
    "FLGatekeeper",
    manifest.contracts.FLGatekeeper.address
  )) as unknown as FLGatekeeper;

  // 1) probe state
  const maxN = sweep[sweep.length - 1];
  const probeLimit = Math.max(2 * maxN + 32, 80);
  console.log(`\n[probe] scanning tokens 1..${probeLimit}`);
  const { active, revoked, highest } = await probeActiveTokens(registry, probeLimit);
  console.log(
    `  highest tokenId: ${highest}  ACTIVE: ${active.length}  REVOKED: ${revoked.length}`
  );

  // 2) clear orphans if requested
  if (clearOrphansFlag && revoked.length > 0) {
    console.log(`\n[clear-orphans] clearing ${revoked.length} REVOKED token(s)...`);
    await clearOrphans(ethers, registry, oracle, manifest, revoked, txOverrides);
  }

  const pending = await gatekeeper.pendingUnlearningCount();
  if (pending !== 0n) {
    throw new Error(
      `gatekeeper.pendingUnlearningCount=${pending} > 0 — every validateRound will revert. ` +
        `Run with CLEAR_ORPHANS=1 first to submit attestations and decrement the count.`
    );
  }

  // Filter ACTIVE → eligible (matching purposeSet). Tokens minted with a
  // different purpose array can't be reused here even though they're
  // ACTIVE.
  const ROUND_PURPOSE = purposeHash(ethers, roundPurposeName);
  const eligibleActive = await filterEligibleTokens(registry, active, ROUND_PURPOSE);
  console.log(
    `  ACTIVE eligible for purpose "${roundPurposeName}": ${eligibleActive.length} of ${active.length}`
  );

  // 3) top up eligible set (mint with the same purpose array we'll
  // pass to isEligible / validateRound)
  let activeTokens = eligibleActive;
  if (activeTokens.length < maxN) {
    activeTokens = await topUp(
      registry, maxN, activeTokens, ROUND_PURPOSE, maxMint, txOverrides
    );
    if (activeTokens.length < maxN) {
      console.log(
        `\n[warn] only have ${activeTokens.length} eligible tokens, sweep will be truncated`
      );
    }
  }

  // 4) sweep
  const samples: SweepSample[] = [];

  console.log(`\n[sweep] N in [${sweep.join(", ")}]`);
  for (const n of sweep) {
    if (activeTokens.length < n) {
      console.log(`  N=${n} skipped — only ${activeTokens.length} ACTIVE tokens available`);
      continue;
    }
    const curRound = await gatekeeper.currentRoundId();
    const freshHash = ethers.keccak256(
      ethers.toUtf8Bytes(`scal-${networkName}-N${n}-${Date.now()}`)
    );

    // a) commit hash for the current round
    const txU = await gatekeeper.updateModelHash(curRound, freshHash, txOverrides);
    const rU = await txU.wait();
    const u = {
      gasUsed: BigInt(rU!.gasUsed).toString(),
      gasPriceWei: BigInt(rU!.gasPrice ?? 0).toString(),
      nativeCostWei: (BigInt(rU!.gasUsed) * BigInt(rU!.gasPrice ?? 0)).toString(),
      txHash: rU!.hash,
      blockNumber: Number(rU!.blockNumber),
    };

    // b) validate next round with first N active tokens
    const participants = activeTokens.slice(0, n);
    const purposes = participants.map(() => [ROUND_PURPOSE]);
    const indices = participants.map(() => 0n);
    const ctx = {
      roundId: curRound + 1n,
      roundPurpose: ROUND_PURPOSE,
      expectedModelHashPre: freshHash,
    };
    const txV = await gatekeeper.validateRound(
      ctx, participants, purposes, indices, txOverrides
    );
    const rV = await txV.wait();
    const v = {
      gasUsed: BigInt(rV!.gasUsed).toString(),
      gasPriceWei: BigInt(rV!.gasPrice ?? 0).toString(),
      nativeCostWei: (BigInt(rV!.gasUsed) * BigInt(rV!.gasPrice ?? 0)).toString(),
      txHash: rV!.hash,
      blockNumber: Number(rV!.blockNumber),
    };

    samples.push({ n, validateRound: v, updateModelHash: u });
    console.log(
      `  N=${String(n).padStart(3)}  validateRound gas=${fmtNum(v.gasUsed).padStart(10)}` +
        `  price=${(Number(v.gasPriceWei) / 1e9).toFixed(2)} gwei` +
        `  block=${v.blockNumber}` +
        (isAmoy && nativePriceUsd
          ? `  cost=$${((Number(v.nativeCostWei) / 1e18) * nativePriceUsd).toFixed(6)}`
          : "")
    );
  }

  // 5) linear regression for the projection
  const fit = linearFit(samples);
  const gas500 = fit.intercept + fit.slope * 500;
  const samplePrice = samples.length > 0 ? BigInt(samples[0].validateRound.gasPriceWei) : 0n;
  const nativeWei500 = BigInt(Math.round(gas500)) * samplePrice;
  const usd500 =
    nativePriceUsd != null ? (Number(nativeWei500) / 1e18) * nativePriceUsd : null;

  const report: ScalabilityReport = {
    network: networkName,
    chainId: Number(chainId),
    measuredAt: new Date().toISOString(),
    nativeSymbol,
    nativePriceUsd,
    participants: samples.map((s) => s.n),
    samples,
    linearFit: {
      intercept: fit.intercept,
      slopePerParticipant: fit.slope,
      fittedFrom: `N ∈ {${samples.map((s) => s.n).join(", ")}}`,
      extrapolatedAtN500_gas: gas500,
      extrapolatedAtN500_nativeWei: nativeWei500.toString(),
      extrapolatedAtN500_usd: usd500,
    },
  };

  const resultsDir = path.join(__dirname, "..", "..", "results", "blockchain");
  fs.mkdirSync(resultsDir, { recursive: true });
  const outPath = path.join(resultsDir, `scalability_${networkName}.json`);
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2) + "\n");
  console.log(`\nReport written: ${outPath}`);
  console.log(
    `Linear fit: gas(N) ≈ ${fmtNum(fit.intercept)} + ${fmtNum(fit.slope)} * N` +
      `   → at N=500: ${fmtNum(gas500)} gas` +
      (usd500 != null ? ` ≈ $${usd500.toFixed(6)}` : "")
  );
}

main()
  .then(() => process.exit(0))
  .catch((err: unknown) => {
    console.error(err);
    process.exit(1);
  });
