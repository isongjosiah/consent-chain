/**
 * scripts/deploy.ts — Deploy ConsentChain to the active Hardhat network.
 *
 * Order (per docs/contracts-spec.md §1):
 *   1. ConsentRegistry(owner)
 *   2. UnlearningOracle(registry, owner)
 *   3. FLGatekeeper(registry, owner)
 *
 * Wiring:
 *   - registry.setOracle(oracle)
 *   - registry.setGatekeeper(gatekeeper)
 *   - oracle.addVerifier(...) × 5  (3-of-5 threshold by default)
 *
 * Verifier addresses come from:
 *   - VC_VERIFIER_{1..5}_PRIVATE_KEY in .env (production path), OR
 *   - hardhat default signers[1..5] if those env vars are unset (local-dev).
 *
 * Writes a manifest to contracts/deployments/{network}.json with addresses,
 * tx hashes, gas used, and effective gas price (which must be 0 on
 * PureChain). Refuses to overwrite an existing manifest unless --force is
 * passed via DEPLOY_FORCE=1.
 *
 * Usage:
 *   npx hardhat run scripts/deploy.ts --network <localhost|purechain|amoy>
 *   DEPLOY_FORCE=1 npx hardhat run scripts/deploy.ts --network purechain
 */
import * as fs from "fs";
import * as path from "path";
import type { TransactionReceipt } from "ethers";

interface ContractDeployment {
  address: string;
  txHash: string;
  blockNumber: number;
  gasUsed: string;
  effectiveGasPrice: string;
}

interface WiringStep {
  description: string;
  txHash: string;
  gasUsed: string;
  effectiveGasPrice: string;
}

interface DeploymentManifest {
  network: string;
  chainId: number;
  deployedAt: string;
  deployer: string;
  contracts: {
    ConsentRegistry: ContractDeployment;
    UnlearningOracle: ContractDeployment;
    FLGatekeeper: ContractDeployment;
  };
  verifiers: string[];
  verifierThreshold: number;
  wiring: WiringStep[];
  totals: {
    txCount: number;
    totalGasUsed: string;
    nonZeroGasPriceTxs: number;
  };
}

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) {
    throw new Error(msg);
  }
}

async function gatherVerifierAddresses(
  ethers: typeof import("hardhat")["ethers"]
): Promise<string[]> {
  const envKeys = [1, 2, 3, 4, 5]
    .map((i) => process.env[`VC_VERIFIER_${i}_PRIVATE_KEY`])
    .filter((k): k is string => typeof k === "string" && k.length > 0);

  if (envKeys.length >= 5) {
    console.log(`  → verifiers: 5 keys from VC_VERIFIER_*_PRIVATE_KEY`);
    return envKeys.slice(0, 5).map((k) => new ethers.Wallet(k).address);
  }

  if (envKeys.length > 0) {
    console.warn(
      `  ⚠  found only ${envKeys.length}/5 VC_VERIFIER_*_PRIVATE_KEY — falling back to hardhat signers`
    );
  }
  const signers = await ethers.getSigners();
  assert(
    signers.length >= 6,
    `need ≥6 signers for a fallback verifier set; only ${signers.length} available`
  );
  console.log(`  → verifiers: 5 hardhat default signers (local-dev fallback)`);
  return signers.slice(1, 6).map((s) => s.address);
}

function summarizeReceipt(receipt: TransactionReceipt): {
  gasUsed: string;
  effectiveGasPrice: string;
  blockNumber: number;
} {
  return {
    gasUsed: receipt.gasUsed.toString(),
    effectiveGasPrice: receipt.gasPrice?.toString() ?? "0",
    blockNumber: receipt.blockNumber,
  };
}

async function main(): Promise<void> {
  // hardhat run injects the augmented HRE as a global; static and dynamic
  // imports of "hardhat" both return stubs in this script's context.
  const hre = (globalThis as unknown as { hre: typeof import("hardhat") }).hre;
  assert(
    hre && hre.network && hre.ethers,
    "global.hre not populated — run via `npx hardhat run scripts/deploy.ts --network <name>`"
  );
  const { ethers, network } = hre;

  const force = process.env.DEPLOY_FORCE === "1";
  const networkName = network.name;
  const chainId = Number((await ethers.provider.getNetwork()).chainId);

  const deploymentsDir = path.join(__dirname, "..", "deployments");
  fs.mkdirSync(deploymentsDir, { recursive: true });
  const manifestPath = path.join(deploymentsDir, `${networkName}.json`);

  if (fs.existsSync(manifestPath) && !force) {
    throw new Error(
      `deployment manifest already exists at ${manifestPath}.\n` +
        `Pass DEPLOY_FORCE=1 to overwrite (this is irreversible).`
    );
  }

  const [deployer] = await ethers.getSigners();
  console.log(`\nNetwork:  ${networkName} (chainId=${chainId})`);
  console.log(`Deployer: ${deployer.address}`);
  console.log(
    `Balance:  ${ethers.formatEther(await ethers.provider.getBalance(deployer.address))} ETH/native\n`
  );

  const verifierAddresses = await gatherVerifierAddresses(ethers);
  console.log();

  // 1. ConsentRegistry
  console.log("[1/3] Deploying ConsentRegistry…");
  const Registry = await ethers.getContractFactory("ConsentRegistry");
  const registry = await Registry.deploy(deployer.address);
  const registryDeployTx = registry.deploymentTransaction();
  assert(registryDeployTx, "registry deployment transaction missing");
  const registryReceipt = await registryDeployTx.wait();
  assert(registryReceipt, "registry receipt missing");
  const registryAddress = await registry.getAddress();
  const registrySummary = summarizeReceipt(registryReceipt);
  console.log(`  address:   ${registryAddress}`);
  console.log(
    `  gas used:  ${registrySummary.gasUsed} @ ${registrySummary.effectiveGasPrice} wei`
  );

  // 2. UnlearningOracle
  console.log("\n[2/3] Deploying UnlearningOracle…");
  const Oracle = await ethers.getContractFactory("UnlearningOracle");
  const oracle = await Oracle.deploy(registryAddress, deployer.address);
  const oracleDeployTx = oracle.deploymentTransaction();
  assert(oracleDeployTx, "oracle deployment transaction missing");
  const oracleReceipt = await oracleDeployTx.wait();
  assert(oracleReceipt, "oracle receipt missing");
  const oracleAddress = await oracle.getAddress();
  const oracleSummary = summarizeReceipt(oracleReceipt);
  console.log(`  address:   ${oracleAddress}`);
  console.log(
    `  gas used:  ${oracleSummary.gasUsed} @ ${oracleSummary.effectiveGasPrice} wei`
  );

  // 3. FLGatekeeper
  console.log("\n[3/3] Deploying FLGatekeeper…");
  const Gatekeeper = await ethers.getContractFactory("FLGatekeeper");
  const gatekeeper = await Gatekeeper.deploy(registryAddress, deployer.address);
  const gkDeployTx = gatekeeper.deploymentTransaction();
  assert(gkDeployTx, "gatekeeper deployment transaction missing");
  const gkReceipt = await gkDeployTx.wait();
  assert(gkReceipt, "gatekeeper receipt missing");
  const gatekeeperAddress = await gatekeeper.getAddress();
  const gkSummary = summarizeReceipt(gkReceipt);
  console.log(`  address:   ${gatekeeperAddress}`);
  console.log(
    `  gas used:  ${gkSummary.gasUsed} @ ${gkSummary.effectiveGasPrice} wei`
  );

  // Wiring
  console.log("\n[wiring] registry.setOracle / setGatekeeper, oracle.addVerifier × 5…");
  const wiring: WiringStep[] = [];

  const setOracleTx = await registry.setOracle(oracleAddress);
  const setOracleReceipt = await setOracleTx.wait();
  assert(setOracleReceipt, "setOracle receipt missing");
  wiring.push({
    description: "registry.setOracle",
    txHash: setOracleReceipt.hash,
    gasUsed: setOracleReceipt.gasUsed.toString(),
    effectiveGasPrice: setOracleReceipt.gasPrice?.toString() ?? "0",
  });
  console.log(`  setOracle:     ${setOracleReceipt.hash}`);

  const setGkTx = await registry.setGatekeeper(gatekeeperAddress);
  const setGkReceipt = await setGkTx.wait();
  assert(setGkReceipt, "setGatekeeper receipt missing");
  wiring.push({
    description: "registry.setGatekeeper",
    txHash: setGkReceipt.hash,
    gasUsed: setGkReceipt.gasUsed.toString(),
    effectiveGasPrice: setGkReceipt.gasPrice?.toString() ?? "0",
  });
  console.log(`  setGatekeeper: ${setGkReceipt.hash}`);

  for (let i = 0; i < verifierAddresses.length; ++i) {
    const v = verifierAddresses[i];
    const tx = await oracle.addVerifier(v);
    const r = await tx.wait();
    assert(r, `addVerifier(${v}) receipt missing`);
    wiring.push({
      description: `oracle.addVerifier #${i + 1}`,
      txHash: r.hash,
      gasUsed: r.gasUsed.toString(),
      effectiveGasPrice: r.gasPrice?.toString() ?? "0",
    });
    console.log(`  addVerifier #${i + 1} (${v}): ${r.hash}`);
  }

  // Manifest
  const allReceipts: { gasUsed: bigint; gasPrice: bigint }[] = [
    { gasUsed: registryReceipt.gasUsed, gasPrice: registryReceipt.gasPrice ?? 0n },
    { gasUsed: oracleReceipt.gasUsed, gasPrice: oracleReceipt.gasPrice ?? 0n },
    { gasUsed: gkReceipt.gasUsed, gasPrice: gkReceipt.gasPrice ?? 0n },
    ...wiring.map((w) => ({
      gasUsed: BigInt(w.gasUsed),
      gasPrice: BigInt(w.effectiveGasPrice),
    })),
  ];
  const totalGasUsed = allReceipts.reduce((a, r) => a + r.gasUsed, 0n);
  const nonZeroGasPriceTxs = allReceipts.filter((r) => r.gasPrice > 0n).length;

  const manifest: DeploymentManifest = {
    network: networkName,
    chainId,
    deployedAt: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      ConsentRegistry: {
        address: registryAddress,
        txHash: registryDeployTx.hash,
        blockNumber: registrySummary.blockNumber,
        gasUsed: registrySummary.gasUsed,
        effectiveGasPrice: registrySummary.effectiveGasPrice,
      },
      UnlearningOracle: {
        address: oracleAddress,
        txHash: oracleDeployTx.hash,
        blockNumber: oracleSummary.blockNumber,
        gasUsed: oracleSummary.gasUsed,
        effectiveGasPrice: oracleSummary.effectiveGasPrice,
      },
      FLGatekeeper: {
        address: gatekeeperAddress,
        txHash: gkDeployTx.hash,
        blockNumber: gkSummary.blockNumber,
        gasUsed: gkSummary.gasUsed,
        effectiveGasPrice: gkSummary.effectiveGasPrice,
      },
    },
    verifiers: verifierAddresses,
    verifierThreshold: Number(await oracle.verifierThreshold()),
    wiring,
    totals: {
      txCount: allReceipts.length,
      totalGasUsed: totalGasUsed.toString(),
      nonZeroGasPriceTxs,
    },
  };

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  console.log(`\nManifest written: ${manifestPath}`);
  console.log(
    `Totals: ${manifest.totals.txCount} txs, ${manifest.totals.totalGasUsed} total gas, ` +
      `${manifest.totals.nonZeroGasPriceTxs} non-zero-gas-price txs.`
  );
  if (nonZeroGasPriceTxs === 0) {
    console.log("✓ Gas-free deployment verified (all effectiveGasPrice == 0).");
  } else {
    console.log("ℹ  Deployment paid gas (expected on Polygon Amoy; would indicate a problem on PureChain).");
  }
}

main()
  .then(() => process.exit(0))
  .catch((err: unknown) => {
    console.error(err);
    process.exit(1);
  });
