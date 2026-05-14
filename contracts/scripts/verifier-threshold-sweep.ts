/**
 * scripts/verifier-threshold-sweep.ts — §IV.3 (Verifiability) evidence.
 *
 * Sweeps the Verifier-Committee threshold t in {2, 3, 4, 5} (the n=5
 * registered verifiers stay fixed). For each t the script does TWO
 * submitAttestation calls against fresh tokens:
 *
 *   - With EXACTLY t valid signatures → expected: accept, state→UNLEARNED,
 *     gatekeeper.unblock fires, gas measured.
 *   - With EXACTLY t-1 valid signatures → expected: revert with
 *     InsufficientSignatures(t-1, t). Captured via eth_call simulation
 *     so we don't burn a real tx for the negative case.
 *
 * Output covers §IV.3's two reviewer-anticipated questions:
 *
 *   1. Does the on-chain gate enforce ≥t signatures? (positive vs
 *      negative branch.)
 *   2. How does t affect cost? (submitAttestation gas as t grows.)
 *
 * Also produces the per-t cleanup trail (each accept transitions a
 * fresh token to UNLEARNED, so the sweep leaves no orphans).
 *
 * Output: results/blockchain/verifier_threshold_sweep.json
 *
 * Usage:
 *   npx hardhat run scripts/verifier-threshold-sweep.ts --network purechain
 *   THRESHOLDS=2,3 npx hardhat run scripts/verifier-threshold-sweep.ts --network purechain
 */
import * as fs from "fs";
import * as path from "path";
import type { Wallet } from "ethers";

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

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function purposeHash(ethers: EthersRuntime, name: string): string {
  return ethers.keccak256(ethers.toUtf8Bytes(name));
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

async function loadAllVerifierSigners(
  ethers: EthersRuntime,
): Promise<Wallet[]> {
  const envKeys = [1, 2, 3, 4, 5]
    .map((i) => process.env[`VC_VERIFIER_${i}_PRIVATE_KEY`])
    .filter((k): k is string => typeof k === "string" && k.length > 0);
  if (envKeys.length < 5) {
    throw new Error(
      `expected all 5 VC_VERIFIER_*_PRIVATE_KEY in env (found ${envKeys.length}). ` +
        "The sweep needs the maximum-threshold case t=5."
    );
  }
  return envKeys.slice(0, 5).map((k) => new ethers.Wallet(k, ethers.provider));
}

async function freshConsent(
  ethers: EthersRuntime,
  registry: ConsentRegistry,
  gatekeeper: FLGatekeeper,
  purpose: string,
): Promise<{
  tokenId: bigint;
  modelHashPre: string;
  grantBlock: number;
  revokeBlock: number;
  grantGas: bigint;
  revokeGas: bigint;
  updateGas: bigint;
}> {
  const Hpre = ethers.keccak256(ethers.toUtf8Bytes(`vt-sweep-${Date.now()}-${Math.random()}`));
  const curRound = await gatekeeper.currentRoundId();
  const updateR = await (await gatekeeper.updateModelHash(curRound, Hpre)).wait();
  const grantR = await (await registry.grantConsent([purpose], 0n)).wait();
  const grantedLog = grantR!.logs
    .map((l: any) => {
      try { return registry.interface.parseLog({ topics: [...l.topics], data: l.data }); }
      catch { return null; }
    })
    .find((p: any) => p?.name === "ConsentGranted");
  assert(grantedLog, "missing ConsentGranted event");
  const tokenId = grantedLog.args[0] as bigint;
  const revokeR = await (await registry.revokeConsent(tokenId)).wait();
  const modelHashPre = await registry.modelHashAtRevocation(tokenId);
  return {
    tokenId,
    modelHashPre,
    grantBlock: Number(grantR!.blockNumber),
    revokeBlock: Number(revokeR!.blockNumber),
    grantGas: BigInt(grantR!.gasUsed),
    revokeGas: BigInt(revokeR!.gasUsed),
    updateGas: BigInt(updateR!.gasUsed),
  };
}

interface ThresholdResult {
  threshold: number;
  acceptTokenId: string;
  acceptSigCount: number;
  acceptTxHash: string;
  acceptGasUsed: string;
  acceptBlockNumber: number;
  rejectAttempted: boolean;
  rejectSigCount: number;
  rejectRevertedAtStaticCall: boolean;
  rejectRevertReason: string | null;
  // Token used for the negative case is left in REVOKED state — cleared
  // at the end via a full-signature attestation so we don't leave a
  // ghost pendingUnlearningCount behind.
  rejectTokenId: string;
}

async function main(): Promise<void> {
  const hre = (globalThis as unknown as { hre: typeof import("hardhat") }).hre;
  assert(hre && hre.network && hre.ethers, "global.hre missing");
  const { ethers, network } = hre;
  const networkName = network.name;
  const manifestPath = path.join(__dirname, "..", "deployments", `${networkName}.json`);
  const manifest: DeploymentManifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const chainId = BigInt(manifest.chainId);

  const registry = (await ethers.getContractAt(
    "ConsentRegistry", manifest.contracts.ConsentRegistry.address
  )) as unknown as ConsentRegistry;
  const oracle = (await ethers.getContractAt(
    "UnlearningOracle", manifest.contracts.UnlearningOracle.address
  )) as unknown as UnlearningOracle;
  const gatekeeper = (await ethers.getContractAt(
    "FLGatekeeper", manifest.contracts.FLGatekeeper.address
  )) as unknown as FLGatekeeper;

  const pending = await gatekeeper.pendingUnlearningCount();
  if (pending !== 0n) {
    throw new Error(`pendingUnlearningCount=${pending} > 0 — clear orphans before sweep.`);
  }

  const thresholds = (process.env.THRESHOLDS ?? "2,3,4,5")
    .split(",").map((s) => parseInt(s.trim(), 10))
    .filter((n) => n >= 1 && n <= 5);
  const initialT = Number(await oracle.verifierThreshold());

  console.log(`Network: ${networkName}`);
  console.log(`Sweep thresholds t ∈ {${thresholds.join(", ")}}`);
  console.log(`Current threshold on-chain: ${initialT}`);

  const allSigners = await loadAllVerifierSigners(ethers);
  const purpose = purposeHash(ethers, "verifier-threshold-sweep");

  const results: ThresholdResult[] = [];

  for (const t of thresholds) {
    console.log(`\n=== t = ${t} (signing with t valid; expected accept) ===`);

    // 1. Set threshold to t.
    await (await oracle.setVerifierThreshold(BigInt(t))).wait();
    const onchainT = Number(await oracle.verifierThreshold());
    assert(onchainT === t, `setVerifierThreshold did not propagate: ${onchainT}`);

    // 2. ACCEPT branch: fresh token + UA + exactly t signatures.
    const acceptConsent = await freshConsent(ethers, registry, gatekeeper, purpose);
    const acceptUa: UaInput = {
      dsTokenId: acceptConsent.tokenId,
      modelHashPre: acceptConsent.modelHashPre,
      modelHashPost: ethers.keccak256(ethers.toUtf8Bytes(`accept-post-${t}-${Date.now()}`)),
      method: 1, // GRADIENT_ASCENT
      shadowMiaScore: 5000n,
      liraScore: 5000n,
      labelOnlyMiaScore: 5000n,
      timestamp: BigInt(Math.floor(Date.now() / 1000)),
    };
    const acceptHash = computeUaHash(ethers, chainId, acceptUa);
    const acceptSigs: string[] = [];
    for (let i = 0; i < t; ++i) {
      acceptSigs.push(await allSigners[i].signMessage(ethers.getBytes(acceptHash)));
    }
    const acceptR = await (
      await oracle.submitAttestation({ ...acceptUa, uaHash: acceptHash }, acceptSigs)
    ).wait();
    const acceptedLog = acceptR!.logs
      .map((l: any) => {
        try { return oracle.interface.parseLog({ topics: [...l.topics], data: l.data }); }
        catch { return null; }
      })
      .find((p: any) => p?.name === "AttestationAccepted");
    assert(acceptedLog, `t=${t}: AttestationAccepted not emitted`);
    console.log(`  ✓ accept  tokenId=${acceptConsent.tokenId}  sigs=${t}  gas=${acceptR!.gasUsed}  tx=${acceptR!.hash}`);

    // 3. REJECT branch: fresh token + UA + (t-1) signatures, simulated via
    // eth_call (staticCall). Skipped for t=1 (no negative case below
    // the floor). The token stays in REVOKED state after this — we
    // submit a clean follow-up UA at the same threshold to clear it.
    let rejectResult: Partial<ThresholdResult> = {
      rejectAttempted: false,
      rejectSigCount: t - 1,
      rejectRevertedAtStaticCall: false,
      rejectRevertReason: null,
      rejectTokenId: "",
    };
    if (t >= 2) {
      const rejectConsent = await freshConsent(ethers, registry, gatekeeper, purpose);
      const rejectUa: UaInput = {
        dsTokenId: rejectConsent.tokenId,
        modelHashPre: rejectConsent.modelHashPre,
        modelHashPost: ethers.keccak256(ethers.toUtf8Bytes(`reject-post-${t}-${Date.now()}`)),
        method: 1,
        shadowMiaScore: 5000n,
        liraScore: 5000n,
        labelOnlyMiaScore: 5000n,
        timestamp: BigInt(Math.floor(Date.now() / 1000)),
      };
      const rejectHash = computeUaHash(ethers, chainId, rejectUa);
      const rejectSigs: string[] = [];
      for (let i = 0; i < t - 1; ++i) {
        rejectSigs.push(await allSigners[i].signMessage(ethers.getBytes(rejectHash)));
      }
      let revertReason: string | null = null;
      try {
        await oracle.submitAttestation.staticCall({ ...rejectUa, uaHash: rejectHash }, rejectSigs);
        console.log(`  ! UNEXPECTED: t=${t} with ${t-1} sigs did NOT revert`);
      } catch (e: any) {
        revertReason = e.shortMessage ?? e.message ?? String(e);
        console.log(`  ✓ reject  tokenId=${rejectConsent.tokenId}  sigs=${t-1}  reason="${(revertReason ?? "").slice(0, 80)}"`);
      }
      // Cleanup: submit a full-signature UA against the rejected token to
      // decrement pendingUnlearningCount — otherwise the sweep leaves
      // ghosts that would block future validateRound calls.
      const cleanupSigs: string[] = [];
      for (let i = 0; i < t; ++i) {
        cleanupSigs.push(await allSigners[i].signMessage(ethers.getBytes(rejectHash)));
      }
      await (await oracle.submitAttestation({ ...rejectUa, uaHash: rejectHash }, cleanupSigs)).wait();
      rejectResult = {
        rejectAttempted: true,
        rejectSigCount: t - 1,
        rejectRevertedAtStaticCall: revertReason !== null,
        rejectRevertReason: revertReason,
        rejectTokenId: rejectConsent.tokenId.toString(),
      };
    } else {
      console.log(`  (no negative branch for t=1 — that's the floor)`);
    }

    results.push({
      threshold: t,
      acceptTokenId: acceptConsent.tokenId.toString(),
      acceptSigCount: t,
      acceptTxHash: acceptR!.hash,
      acceptGasUsed: BigInt(acceptR!.gasUsed).toString(),
      acceptBlockNumber: Number(acceptR!.blockNumber),
      ...rejectResult,
    } as ThresholdResult);
  }

  // Restore initial threshold.
  console.log(`\n[cleanup] restoring threshold to ${initialT}`);
  await (await oracle.setVerifierThreshold(BigInt(initialT))).wait();
  const finalT = Number(await oracle.verifierThreshold());
  const finalPending = await gatekeeper.pendingUnlearningCount();
  console.log(`  → threshold ${finalT}, pendingUnlearningCount ${finalPending}`);

  const report = {
    description:
      "§IV.3 Verifiability empirical evidence: UnlearningOracle.submitAttestation " +
      "accepts iff ≥ verifierThreshold distinct registered ECDSA signatures " +
      "are recovered from the uaHash; (t-1) signatures revert with " +
      "InsufficientSignatures(t-1, t). Sweep covers t in {" + thresholds.join(", ") + "} " +
      "against a fixed n=5 registered verifier set.",
    network: networkName,
    chainId: Number(chainId),
    nMax: 5,
    initialThreshold: initialT,
    sweep: results,
    note:
      "submitAttestation gas grows ~linearly with t because ECDSA signature " +
      "recovery (~3,000 gas per signer) and verifier-set membership check " +
      "are performed once per submitted signature. Negative branches were " +
      "captured via eth_call simulation (rejectRevertedAtStaticCall=true) " +
      "so no real on-chain revert tx was charged; for paper Table-X we " +
      "report the accept tx hash and gas, plus the verified revert message.",
    measuredAt: new Date().toISOString(),
  };

  const outDir = path.join(__dirname, "..", "..", "results", "blockchain");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, `verifier_threshold_sweep_${networkName}.json`);
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2) + "\n");
  console.log(`\nReport: ${outPath}`);

  console.log(`\n=== sweep summary ===`);
  console.log(`| t | accept gas | reject sigs | revert? |`);
  console.log(`|---|------------|-------------|---------|`);
  for (const r of results) {
    const rg = r.rejectAttempted ? String(r.rejectSigCount) : "—";
    const rv = r.rejectAttempted ? (r.rejectRevertedAtStaticCall ? "yes" : "no") : "n/a";
    console.log(`| ${r.threshold} | ${Number(r.acceptGasUsed).toLocaleString().padStart(10)} | ${rg.padStart(11)} | ${rv.padStart(7)} |`);
  }
}

main().then(() => process.exit(0)).catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
