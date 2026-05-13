/**
 * scripts/smoke-test.ts — run the full §7 lifecycle on-chain.
 *
 * Reads deployments/{network}.json (produced by deploy.ts) and exercises:
 *   1. grantConsent(PURPOSES, 0)
 *   2. updateModelHash(currentRoundId, H_pre)
 *   3. validateRound(currentRoundId + 1, [tokenId], ...)
 *   4. updateModelHash(newRoundId, H_post)
 *   5. revokeConsent(tokenId)   → gatekeeper.pendingUnlearningCount += 1
 *   6. build UA with modelHashPre = H_post
 *   7. verifier signatures (3-of-5)
 *   8. oracle.submitAttestation(ua, sigs)
 *      → registry.transitionToUnlearned
 *      → gatekeeper.unblock(tokenId)
 *   9. assert state == UNLEARNED, pendingUnlearningCount == 0
 *
 * Idempotent: works against either a freshly deployed manifest or one
 * where prior smoke tests have advanced currentRoundId.
 *
 * Prints gasUsed + effectiveGasPrice per tx; tallies non-zero-gas-price
 * tx count (must be 0 on PureChain).
 *
 * Usage:
 *   npx hardhat run scripts/smoke-test.ts --network <localhost|purechain|amoy>
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

const Method = { SISA_FED: 0, GRADIENT_ASCENT: 1, FEDERASER: 2 } as const;

const PURPOSES: string[] = [];

function buildPurposes(ethers: EthersRuntime): string[] {
  return ["adl-recognition", "anomaly-detection", "research"]
    .map((p) => ethers.keccak256(ethers.toUtf8Bytes(p)))
    .sort();
}

interface DeploymentManifest {
  network: string;
  chainId: number;
  contracts: Record<"ConsentRegistry" | "UnlearningOracle" | "FLGatekeeper", { address: string }>;
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

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

async function gasReport(
  ethers: EthersRuntime,
  label: string,
  txHash: string
): Promise<{ gasUsed: bigint; gasPrice: bigint }> {
  const r = await ethers.provider.getTransactionReceipt(txHash);
  assert(r, `receipt missing for ${label} (${txHash})`);
  const gasUsed = r.gasUsed;
  const gasPrice = r.gasPrice ?? 0n;
  console.log(
    `  ${label.padEnd(36)} gas=${gasUsed.toString().padStart(8)}` +
      ` price=${gasPrice.toString()} wei`
  );
  return { gasUsed, gasPrice };
}

async function loadVerifierSigners(
  ethers: EthersRuntime,
  manifest: DeploymentManifest
): Promise<Signer[]> {
  const need = manifest.verifierThreshold;
  const envKeys = [1, 2, 3, 4, 5]
    .map((i) => process.env[`VC_VERIFIER_${i}_PRIVATE_KEY`])
    .filter((k): k is string => typeof k === "string" && k.length > 0);

  if (envKeys.length >= need) {
    const wallets: Wallet[] = envKeys
      .slice(0, 5)
      .map((k) => new ethers.Wallet(k, ethers.provider));
    const addresses = await Promise.all(wallets.map((w) => w.getAddress()));
    for (let i = 0; i < need; ++i) {
      assert(
        addresses[i].toLowerCase() === manifest.verifiers[i].toLowerCase(),
        `verifier #${i + 1}: env key derives ${addresses[i]} but manifest expects ${manifest.verifiers[i]}`
      );
    }
    return wallets.slice(0, need);
  }

  const all = await ethers.getSigners();
  const signers: Signer[] = [];
  for (const addr of manifest.verifiers.slice(0, need)) {
    const match = all.find((s) => s.address.toLowerCase() === addr.toLowerCase());
    assert(match, `no hardhat signer matching verifier ${addr} — set VC_VERIFIER_*_PRIVATE_KEY`);
    signers.push(match);
  }
  return signers;
}

async function main(): Promise<void> {
  const hre = (globalThis as unknown as { hre: typeof import("hardhat") }).hre;
  assert(
    hre && hre.network && hre.ethers,
    "global.hre not populated — run via `npx hardhat run scripts/smoke-test.ts --network <name>`"
  );
  const { ethers, network } = hre;

  const networkName = network.name;
  const chainId = BigInt((await ethers.provider.getNetwork()).chainId);
  const manifestPath = path.join(__dirname, "..", "deployments", `${networkName}.json`);
  assert(fs.existsSync(manifestPath), `missing deployment manifest: ${manifestPath}`);
  const manifest: DeploymentManifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));

  const purposes = buildPurposes(ethers);
  PURPOSES.push(...purposes);
  const ROUND_PURPOSE = purposes[0];

  const [deployer] = await ethers.getSigners();
  console.log(`\nNetwork:   ${networkName} (chainId=${chainId})`);
  console.log(`Deployer:  ${deployer.address}`);
  console.log(`Manifest:  ${manifestPath}\n`);

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

  const verifierSigners = await loadVerifierSigners(ethers, manifest);
  console.log(`Verifiers loaded: ${verifierSigners.length}/${manifest.verifierThreshold}\n`);

  const receipts: { gasUsed: bigint; gasPrice: bigint }[] = [];
  const track = async (label: string, txHash: string) => {
    receipts.push(await gasReport(ethers, label, txHash));
  };

  // 1. grantConsent
  console.log("[1/8] grantConsent");
  const grantTx = await registry.grantConsent(purposes, 0n);
  const grantReceipt = await grantTx.wait();
  assert(grantReceipt, "grant receipt missing");
  await track("registry.grantConsent", grantReceipt.hash);

  const grantedLog = grantReceipt.logs
    .map((l) => {
      try {
        return registry.interface.parseLog({ topics: [...l.topics], data: l.data });
      } catch {
        return null;
      }
    })
    .find((p) => p?.name === "ConsentGranted");
  assert(grantedLog, "ConsentGranted not in receipt");
  const tokenId: bigint = grantedLog.args[0];
  console.log(`  → tokenId=${tokenId}\n`);

  // 2. updateModelHash(pre)
  console.log("[2/8] updateModelHash for current round");
  const curRound = await gatekeeper.currentRoundId();
  const Hpre = ethers.keccak256(
    ethers.toUtf8Bytes(`smoke-${networkName}-pre-${Date.now()}`)
  );
  const preTx = await gatekeeper.updateModelHash(curRound, Hpre);
  const preReceipt = await preTx.wait();
  assert(preReceipt, "updateModelHash(pre) receipt missing");
  await track(`gatekeeper.updateModelHash(${curRound})`, preReceipt.hash);
  console.log();

  // 3. validateRound
  console.log("[3/8] validateRound for round N+1");
  const nextRound = curRound + 1n;
  const valTx = await gatekeeper.validateRound(
    { roundId: nextRound, roundPurpose: ROUND_PURPOSE, expectedModelHashPre: Hpre },
    [tokenId],
    [purposes],
    [0n]
  );
  const valReceipt = await valTx.wait();
  assert(valReceipt, "validateRound receipt missing");
  await track(`gatekeeper.validateRound(${nextRound})`, valReceipt.hash);
  console.log();

  // 4. updateModelHash(post)
  console.log("[4/8] updateModelHash for newly-validated round");
  const Hpost = ethers.keccak256(
    ethers.toUtf8Bytes(`smoke-${networkName}-post-${Date.now()}`)
  );
  const postTx = await gatekeeper.updateModelHash(nextRound, Hpost);
  const postReceipt = await postTx.wait();
  assert(postReceipt, "updateModelHash(post) receipt missing");
  await track(`gatekeeper.updateModelHash(${nextRound})`, postReceipt.hash);
  console.log();

  // 5. revoke
  console.log("[5/8] revokeConsent");
  const revokeTx = await registry.revokeConsent(tokenId);
  const revokeReceipt = await revokeTx.wait();
  assert(revokeReceipt, "revoke receipt missing");
  await track("registry.revokeConsent", revokeReceipt.hash);
  const pendingAfterRevoke = await gatekeeper.pendingUnlearningCount();
  assert(
    pendingAfterRevoke >= 1n,
    `pendingUnlearningCount expected ≥ 1 after revoke, got ${pendingAfterRevoke}`
  );
  console.log(`  pendingUnlearningCount=${pendingAfterRevoke}\n`);

  // 6+7. build + sign + submit UA
  console.log("[6/8] build + sign UA");
  const ua: UaInput = {
    dsTokenId: tokenId,
    modelHashPre: Hpost,
    modelHashPost: ethers.keccak256(
      ethers.toUtf8Bytes(`smoke-${networkName}-unlearned-${Date.now()}`)
    ),
    method: Method.SISA_FED,
    shadowMiaScore: 5100n,
    liraScore: 4900n,
    labelOnlyMiaScore: 5000n,
    timestamp: BigInt(Math.floor(Date.now() / 1000)),
  };
  const uaHash = computeUaHash(ethers, chainId, ua);
  const sigs: string[] = [];
  for (const v of verifierSigners) {
    sigs.push(await v.signMessage(ethers.getBytes(uaHash)));
  }
  console.log(`  uaHash=${uaHash}`);
  console.log(`  ${sigs.length} signatures collected\n`);

  console.log("[7/8] submitAttestation");
  const submitTx = await oracle.submitAttestation({ ...ua, uaHash }, sigs);
  const submitReceipt = await submitTx.wait();
  assert(submitReceipt, "submit receipt missing");
  await track("oracle.submitAttestation", submitReceipt.hash);
  console.log();

  // 8. assertions
  console.log("[8/8] post-state checks");
  const finalState = await registry.stateOf(tokenId);
  assert(finalState === 5n, `expected state UNLEARNED (5), got ${finalState}`);
  const pendingAfterUa = await gatekeeper.pendingUnlearningCount();
  assert(
    pendingAfterUa === pendingAfterRevoke - 1n,
    `pendingUnlearningCount should decrement to ${pendingAfterRevoke - 1n}, got ${pendingAfterUa}`
  );
  console.log(`  registry.stateOf(${tokenId}) = UNLEARNED ✓`);
  console.log(`  gatekeeper.pendingUnlearningCount = ${pendingAfterUa} ✓\n`);

  const totalGas = receipts.reduce((a, r) => a + r.gasUsed, 0n);
  const nonZeroGasPrice = receipts.filter((r) => r.gasPrice > 0n).length;
  console.log(
    `Totals: ${receipts.length} txs, ${totalGas} total gas, ` +
      `${nonZeroGasPrice} non-zero-gas-price txs.`
  );
  if (nonZeroGasPrice === 0) {
    console.log("✓ Gas-free smoke-test verified (every tx had effectiveGasPrice == 0).");
  } else {
    console.log("ℹ  At least one tx paid gas (expected on Polygon Amoy).");
  }
}

main()
  .then(() => process.exit(0))
  .catch((err: unknown) => {
    console.error(err);
    process.exit(1);
  });
