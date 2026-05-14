/**
 * scripts/compliance-violation-demo.ts — §IV.2 (Liveness) evidence.
 *
 * Demonstrates that FLGatekeeper.reportTimeout() permissionlessly
 * emits ComplianceViolation after Δ_max blocks have elapsed since a
 * revocation without an Unlearning Attestation being submitted. This
 * is the on-chain proof for the §IV.2 liveness property: the system
 * degrades to "no training" rather than "unconsented training" if the
 * data controller fails to perform unlearning within Δ_max blocks.
 *
 * Sequence (all on PureChain, free):
 *
 *   1. setDeltaMax(small_value)        // shrink the window for the demo
 *   2. grantConsent → tokenId          // fresh DS for the demo
 *   3. updateModelHash(...)            // commit a pre-hash for the captured snapshot
 *   4. revokeConsent(tokenId)          // gatekeeper records pendingUnlearningSince[token] = block.number
 *   5. advance ≥ deltaMax+1 blocks     // by submitting dummy gatekeeper.updateModelHash() txs
 *   6. reportTimeout(tokenId)          // expected: succeeds, emits ComplianceViolation(tokenId, deadline)
 *   7. (cleanup) submit valid UA       // decrement pendingUnlearningCount so subsequent rounds can run
 *   8. setDeltaMax(DEFAULT)            // restore normal operation
 *
 * Negative checkpoint: before block deadline, reportTimeout reverts
 * with NotYetTimedOut(deadline, blockNumber) — captured separately so
 * the paper can cite both branches (timeout-too-early reverts;
 * timeout-after-deadline succeeds).
 *
 * Output: results/blockchain/compliance_violation_demo.json
 *
 * Usage:
 *   npx hardhat run scripts/compliance-violation-demo.ts --network purechain
 *   DELTA_MAX=2 npx hardhat run scripts/compliance-violation-demo.ts --network purechain
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

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function purposeHash(ethers: EthersRuntime, name: string): string {
  return ethers.keccak256(ethers.toUtf8Bytes(name));
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

async function loadVerifierSigners(
  ethers: EthersRuntime,
  manifest: DeploymentManifest,
  need: number,
): Promise<Signer[]> {
  const envKeys = [1, 2, 3, 4, 5]
    .map((i) => process.env[`VC_VERIFIER_${i}_PRIVATE_KEY`])
    .filter((k): k is string => typeof k === "string" && k.length > 0);
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

async function main(): Promise<void> {
  const hre = (globalThis as unknown as { hre: typeof import("hardhat") }).hre;
  assert(hre && hre.network && hre.ethers, "global.hre missing");
  const { ethers, network } = hre;
  const networkName = network.name;
  const manifestPath = path.join(__dirname, "..", "deployments", `${networkName}.json`);
  assert(fs.existsSync(manifestPath), `missing deployment manifest: ${manifestPath}`);
  const manifest: DeploymentManifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const chainId = BigInt(manifest.chainId);

  const targetDeltaMax = BigInt(process.env.DELTA_MAX ?? 2);
  const DEFAULT_DELTA_MAX = 1000n;

  console.log(`Network: ${networkName} (chainId=${chainId})`);
  console.log(`Target Δ_max for the demo: ${targetDeltaMax} blocks`);

  const registry = (await ethers.getContractAt(
    "ConsentRegistry", manifest.contracts.ConsentRegistry.address
  )) as unknown as ConsentRegistry;
  const oracle = (await ethers.getContractAt(
    "UnlearningOracle", manifest.contracts.UnlearningOracle.address
  )) as unknown as UnlearningOracle;
  const gatekeeper = (await ethers.getContractAt(
    "FLGatekeeper", manifest.contracts.FLGatekeeper.address
  )) as unknown as FLGatekeeper;

  const initialDeltaMax = await gatekeeper.deltaMaxBlocks();
  console.log(`Current Δ_max on-chain: ${initialDeltaMax} blocks`);

  // Sanity: pendingUnlearningCount must be 0 before the demo so validateRound
  // wouldn't be blocked by an unrelated orphan during cleanup.
  const pending = await gatekeeper.pendingUnlearningCount();
  if (pending !== 0n) {
    throw new Error(
      `pendingUnlearningCount=${pending} > 0 — clear orphans before running this demo.`
    );
  }

  const txs: { step: string; txHash: string; blockNumber: number; gasUsed?: string }[] = [];
  const record = async (step: string, p: Promise<unknown>) => {
    const tx: any = await p;
    const r: any = await tx.wait();
    txs.push({ step, txHash: r.hash, blockNumber: Number(r.blockNumber), gasUsed: r.gasUsed.toString() });
    console.log(`  ${step.padEnd(38)} block=${r.blockNumber} gas=${r.gasUsed} tx=${r.hash}`);
    return r;
  };

  // 1. Shrink Δ_max so the demo doesn't have to wait 1000 blocks.
  console.log(`\n[1] setDeltaMax(${targetDeltaMax})`);
  await record(`setDeltaMax(${targetDeltaMax})`, gatekeeper.setDeltaMax(targetDeltaMax));

  // 2. Fresh consent for the demo (purpose "compliance-demo" so it doesn't
  // accidentally satisfy a future validateRound with "research").
  console.log(`\n[2] grant fresh consent`);
  const P = [purposeHash(ethers, "compliance-demo")];
  const grantR = await record("grantConsent", registry.grantConsent(P, 0n));
  const grantedLog = grantR.logs
    .map((l: any) => {
      try { return registry.interface.parseLog({ topics: [...l.topics], data: l.data }); }
      catch { return null; }
    })
    .find((p: any) => p?.name === "ConsentGranted");
  assert(grantedLog, "missing ConsentGranted event");
  const tokenId = grantedLog.args[0] as bigint;
  console.log(`  → tokenId = ${tokenId}`);

  // 3. Commit a model hash before revocation so revokeConsent has a snapshot.
  console.log(`\n[3] updateModelHash`);
  const Hpre = ethers.keccak256(ethers.toUtf8Bytes(`compliance-demo-pre-${Date.now()}`));
  const curRound = await gatekeeper.currentRoundId();
  await record("updateModelHash", gatekeeper.updateModelHash(curRound, Hpre));

  // 4. Revoke → records pendingUnlearningSince[tokenId] = block.number.
  console.log(`\n[4] revokeConsent (pendingUnlearningSince ← block.number)`);
  await record("revokeConsent", registry.revokeConsent(tokenId));
  const since = await gatekeeper.pendingUnlearningSince(tokenId);
  const deadline = since + targetDeltaMax;
  console.log(`  → pendingUnlearningSince = ${since}, deadline = since + Δ_max = ${deadline}`);

  // 4b. Negative checkpoint: reportTimeout before deadline should revert.
  console.log(`\n[4b] reportTimeout BEFORE deadline (expecting revert)`);
  let earlyRevertReason: string | null = null;
  try {
    await gatekeeper.reportTimeout.staticCall(tokenId);
    console.log(`  unexpected: pre-deadline reportTimeout did not revert`);
  } catch (e: any) {
    const reason = e.shortMessage ?? e.message ?? String(e);
    earlyRevertReason = reason;
    console.log(`  ✓ reverted as expected: ${reason.slice(0, 120)}`);
  }

  // 5. Advance ≥ Δ_max+1 blocks by submitting dummy updateModelHash calls.
  // updateModelHash on the current round doesn't advance roundId — it just
  // mines a block on PureChain with a cheap state write. We need
  // block.number > deadline, i.e., block.number ≥ deadline + 1.
  console.log(`\n[5] advancing chain past deadline`);
  let currentBlock = BigInt((await ethers.provider.getBlock("latest"))!.number);
  console.log(`  current block: ${currentBlock}`);
  let bumps = 0;
  while (currentBlock <= deadline) {
    const dummyHash = ethers.keccak256(ethers.toUtf8Bytes(`bump-${Date.now()}-${bumps}`));
    await gatekeeper.updateModelHash(curRound, dummyHash).then((tx: any) => tx.wait());
    bumps += 1;
    currentBlock = BigInt((await ethers.provider.getBlock("latest"))!.number);
  }
  console.log(`  advanced ${bumps} block(s); current=${currentBlock} > deadline=${deadline}`);

  // 6. reportTimeout — succeeds, emits ComplianceViolation.
  console.log(`\n[6] reportTimeout AFTER deadline (expecting ComplianceViolation event)`);
  const timeoutR = await record("reportTimeout", gatekeeper.reportTimeout(tokenId));
  const violationLog = timeoutR.logs
    .map((l: any) => {
      try { return gatekeeper.interface.parseLog({ topics: [...l.topics], data: l.data }); }
      catch { return null; }
    })
    .find((p: any) => p?.name === "ComplianceViolation");
  if (!violationLog) {
    throw new Error("ComplianceViolation event was not emitted — §IV.2 evidence failed");
  }
  const violationData = {
    tokenId: (violationLog.args[0] as bigint).toString(),
    deadline: (violationLog.args[1] as bigint).toString(),
    reportedAtBlock: timeoutR.blockNumber,
  };
  console.log(`  ✓ ComplianceViolation(tokenId=${violationData.tokenId}, deadline=${violationData.deadline}) at block ${violationData.reportedAtBlock}`);

  // 7. Cleanup: submit a valid UA to clear the pending state.
  console.log(`\n[7] submit valid UA to clear pending state`);
  const need = Number(await oracle.verifierThreshold());
  const verifierSigners = await loadVerifierSigners(ethers, manifest, need);
  const Hpost = ethers.keccak256(ethers.toUtf8Bytes(`compliance-demo-post-${Date.now()}`));
  const ua: UaInput = {
    dsTokenId: tokenId,
    modelHashPre: await registry.modelHashAtRevocation(tokenId),
    modelHashPost: Hpost,
    method: 1, // GRADIENT_ASCENT
    shadowMiaScore: 5000n,
    liraScore: 5000n,
    labelOnlyMiaScore: 5000n,
    timestamp: BigInt(Math.floor(Date.now() / 1000)),
  };
  const uaHash = computeUaHash(ethers, chainId, ua);
  const sigs: string[] = [];
  for (const v of verifierSigners) sigs.push(await v.signMessage(ethers.getBytes(uaHash)));
  await record("submitAttestation", oracle.submitAttestation({ ...ua, uaHash }, sigs));

  // 8. Restore Δ_max.
  console.log(`\n[8] restore Δ_max to default (${DEFAULT_DELTA_MAX})`);
  await record(`setDeltaMax(${DEFAULT_DELTA_MAX})`, gatekeeper.setDeltaMax(DEFAULT_DELTA_MAX));

  // Verify final state.
  const finalState = Number(await registry.stateOf(tokenId));
  const finalPending = await gatekeeper.pendingUnlearningCount();
  console.log(`\nFinal state: tokenId=${tokenId} state=${finalState} (5=UNLEARNED), pendingUnlearningCount=${finalPending}`);

  const report = {
    description:
      "§IV.2 Liveness empirical evidence: FLGatekeeper.reportTimeout emits " +
      "ComplianceViolation iff (block.number > pendingUnlearningSince + Δ_max), " +
      "permissionlessly, with no signature required. Demonstrated end-to-end " +
      "on PureChain at Δ_max = " + targetDeltaMax + ".",
    network: networkName,
    chainId: Number(chainId),
    deltaMaxUsed: Number(targetDeltaMax),
    deltaMaxDefault: Number(DEFAULT_DELTA_MAX),
    tokenId: tokenId.toString(),
    timeline: txs,
    earlyTimeoutRevert: earlyRevertReason,
    complianceViolationEvent: violationData,
    finalConsentState: finalState,
    finalPendingUnlearningCount: finalPending.toString(),
    measuredAt: new Date().toISOString(),
  };

  const outDir = path.join(__dirname, "..", "..", "results", "blockchain");
  fs.mkdirSync(outDir, { recursive: true });
  const outPath = path.join(outDir, `compliance_violation_demo_${networkName}.json`);
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2) + "\n");
  console.log(`\nReport: ${outPath}`);
}

main().then(() => process.exit(0)).catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
