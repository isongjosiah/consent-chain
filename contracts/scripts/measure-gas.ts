/**
 * scripts/measure-gas.ts — produce the Table 2 dataset for the BCCA paper.
 *
 * Two modes:
 *
 *   1. Default: run against the active hardhat network, exercising the
 *      full operation matrix once with fresh tokens. Writes
 *      results/blockchain/gas_{network}.json.
 *
 *   2. With env COMPARE_ONLY=1: read every results/blockchain/gas_*.json
 *      file on disk, merge into results/blockchain/gas_comparison.json,
 *      and print a markdown table sized for the paper's Table 2.
 *
 * Operations measured (all live, once each, in a single deterministic order):
 *
 *   grantConsent (1 purpose), grantConsent (4 purposes), modifyConsent,
 *   suspendConsent, resumeConsent, revokeConsent, updateModelHash,
 *   validateRound (N=1), validateRound (N=5), submitAttestation (3-of-5).
 *
 * Deployment + wiring gas come from deployments/{network}.json so we don't
 * pay to re-deploy on every measurement run. The script reads the manifest
 * for addresses and verifier keys.
 *
 * MATIC/USD price fetched from CoinGecko (best-effort; missing price leaves
 * usdCost = null). Override with MATIC_USD env var (decimal string).
 *
 * Usage:
 *   npx hardhat run scripts/measure-gas.ts --network purechain
 *   npx hardhat run scripts/measure-gas.ts --network amoy
 *   COMPARE_ONLY=1 npx hardhat run scripts/measure-gas.ts --network localhost
 */
import * as fs from "fs";
import * as path from "path";
import * as https from "https";
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
    { address: string; gasUsed: string; effectiveGasPrice: string }
  >;
  wiring: { description: string; gasUsed: string; effectiveGasPrice: string }[];
  verifiers: string[];
  verifierThreshold: number;
}

interface OperationMeasurement {
  category: "deploy" | "wiring" | "consent" | "round" | "attestation";
  gasUsed: string;
  gasPriceWei: string;
  nativeCostWei: string;
  txHash?: string;
}

interface NetworkGasReport {
  network: string;
  chainId: number;
  measuredAt: string;
  nativeSymbol: string;
  nativePriceUsd: number | null;
  operations: Record<string, OperationMeasurement>;
}

const Method = { SISA_FED: 0, GRADIENT_ASCENT: 1, FEDERASER: 2 } as const;

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function fmtNum(s: string | bigint): string {
  const n = typeof s === "bigint" ? s : BigInt(s);
  return n.toLocaleString("en-US");
}

function purposeHash(ethers: EthersRuntime, name: string): string {
  return ethers.keccak256(ethers.toUtf8Bytes(name));
}

// Fetch Polygon gas-token price from CoinGecko. Polygon migrated the
// MATIC ticker to POL in late 2024; the new CoinGecko ID is
// "polygon-ecosystem-token". Fall back to MATIC_USD env override.
async function fetchMaticUsd(): Promise<number | null> {
  if (process.env.MATIC_USD) {
    const v = Number(process.env.MATIC_USD);
    if (!Number.isNaN(v)) return v;
  }
  return new Promise<number | null>((resolve) => {
    const req = https.get(
      "https://api.coingecko.com/api/v3/simple/price?ids=polygon-ecosystem-token&vs_currencies=usd",
      { timeout: 6000 },
      (res) => {
        let data = "";
        res.on("data", (c) => (data += c));
        res.on("end", () => {
          try {
            const j = JSON.parse(data);
            const v = j?.["polygon-ecosystem-token"]?.usd;
            resolve(typeof v === "number" ? v : null);
          } catch {
            resolve(null);
          }
        });
      }
    );
    req.on("error", () => resolve(null));
    req.on("timeout", () => {
      req.destroy();
      resolve(null);
    });
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

// =======================================================================
// Operation matrix — run against the active network
// =======================================================================

async function measureLive(
  ethers: EthersRuntime,
  manifest: DeploymentManifest,
  nativeSymbol: string,
  nativePriceUsd: number | null
): Promise<NetworkGasReport> {
  const chainId = BigInt(manifest.chainId);
  const ops: Record<string, OperationMeasurement> = {};

  // -- Carry-over from the deployment manifest --
  ops["deploy_ConsentRegistry"] = {
    category: "deploy",
    gasUsed: manifest.contracts.ConsentRegistry.gasUsed,
    gasPriceWei: manifest.contracts.ConsentRegistry.effectiveGasPrice,
    nativeCostWei: (
      BigInt(manifest.contracts.ConsentRegistry.gasUsed) *
      BigInt(manifest.contracts.ConsentRegistry.effectiveGasPrice)
    ).toString(),
  };
  ops["deploy_UnlearningOracle"] = {
    category: "deploy",
    gasUsed: manifest.contracts.UnlearningOracle.gasUsed,
    gasPriceWei: manifest.contracts.UnlearningOracle.effectiveGasPrice,
    nativeCostWei: (
      BigInt(manifest.contracts.UnlearningOracle.gasUsed) *
      BigInt(manifest.contracts.UnlearningOracle.effectiveGasPrice)
    ).toString(),
  };
  ops["deploy_FLGatekeeper"] = {
    category: "deploy",
    gasUsed: manifest.contracts.FLGatekeeper.gasUsed,
    gasPriceWei: manifest.contracts.FLGatekeeper.effectiveGasPrice,
    nativeCostWei: (
      BigInt(manifest.contracts.FLGatekeeper.gasUsed) *
      BigInt(manifest.contracts.FLGatekeeper.effectiveGasPrice)
    ).toString(),
  };
  for (const w of manifest.wiring) {
    const key = w.description.replace(/[^a-zA-Z0-9]+/g, "_");
    ops[`wire_${key}`] = {
      category: "wiring",
      gasUsed: w.gasUsed,
      gasPriceWei: w.effectiveGasPrice,
      nativeCostWei: (BigInt(w.gasUsed) * BigInt(w.effectiveGasPrice)).toString(),
    };
  }

  // -- Get live handles --
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

  const measure = async (
    label: string,
    category: OperationMeasurement["category"],
    txPromise: Promise<unknown>
  ): Promise<bigint /* returns gasUsed for sequencing */> => {
    const tx: any = await txPromise;
    const receipt = await tx.wait();
    const gasUsed = BigInt(receipt.gasUsed);
    const gasPrice = BigInt(receipt.gasPrice ?? 0);
    ops[label] = {
      category,
      gasUsed: gasUsed.toString(),
      gasPriceWei: gasPrice.toString(),
      nativeCostWei: (gasUsed * gasPrice).toString(),
      txHash: receipt.hash,
    };
    console.log(`  ${label.padEnd(34)} gas=${fmtNum(gasUsed).padStart(10)} price=${gasPrice.toString()}`);
    return gasUsed;
  };

  // ------ grantConsent variants ------
  const P1 = [purposeHash(ethers, "research")];
  const P4 = ["a", "b", "c", "d"].map((s) => purposeHash(ethers, `p-${s}`)).sort();

  console.log("\n[live measurements]");
  await measure("grantConsent_1purpose", "consent", registry.grantConsent(P1, 0n));
  // tokenIds advance from whatever the manifest already deployed; we read receipts
  // for the IDs via events below.

  // Need to know which tokenIds we just minted. Read the latest events.
  const block = await ethers.provider.getBlockNumber();
  const grantedLogs = await registry.queryFilter(
    registry.filters.ConsentGranted(),
    block - 1,
    block
  );
  assert(grantedLogs.length >= 1, "expected ConsentGranted event for tokenA");
  const tokenA = grantedLogs[grantedLogs.length - 1].args[0];

  await measure("grantConsent_4purposes", "consent", registry.grantConsent(P4, 0n));
  const grantedLogs2 = await registry.queryFilter(
    registry.filters.ConsentGranted(),
    await ethers.provider.getBlockNumber() - 1,
    "latest"
  );
  const tokenB = grantedLogs2[grantedLogs2.length - 1].args[0];

  // Tokens C–F for validateRound(N=5). All use P1 so the round purpose matches.
  const extraTokens: bigint[] = [];
  for (let i = 0; i < 4; ++i) {
    const tx = await registry.grantConsent(P1, 0n);
    const r = await tx.wait();
    assert(r, "extra-grant receipt missing");
    const logs = r.logs
      .map((l) => {
        try {
          return registry.interface.parseLog({ topics: [...l.topics], data: l.data });
        } catch {
          return null;
        }
      })
      .filter((p) => p?.name === "ConsentGranted");
    extraTokens.push(logs[0]!.args[0] as bigint);
  }

  // ------ round-validation measurements ------
  // Commit current model hash for the current round, then validate.
  const curRound = await gatekeeper.currentRoundId();
  const Hpre = ethers.keccak256(ethers.toUtf8Bytes(`gas-${manifest.network}-pre-${Date.now()}`));
  await measure(
    "updateModelHash",
    "round",
    gatekeeper.updateModelHash(curRound, Hpre)
  );

  const ROUND_PURPOSE = P1[0];
  await measure(
    "validateRound_N1",
    "round",
    gatekeeper.validateRound(
      { roundId: curRound + 1n, roundPurpose: ROUND_PURPOSE, expectedModelHashPre: Hpre },
      [tokenA],
      [P1],
      [0n]
    )
  );

  // Bump the model hash so we can validate the next round.
  const Hmid = ethers.keccak256(ethers.toUtf8Bytes(`gas-${manifest.network}-mid-${Date.now()}`));
  await gatekeeper.updateModelHash(curRound + 1n, Hmid).then((tx) => tx.wait());

  await measure(
    "validateRound_N5",
    "round",
    gatekeeper.validateRound(
      { roundId: curRound + 2n, roundPurpose: ROUND_PURPOSE, expectedModelHashPre: Hmid },
      [tokenA, ...extraTokens],
      [P1, P1, P1, P1, P1],
      [0n, 0n, 0n, 0n, 0n]
    )
  );

  // ------ consent lifecycle on tokenA ------
  await measure("modifyConsent", "consent", registry.modifyConsent(tokenA, P4, 0n));
  await measure("suspendConsent", "consent", registry.suspendConsent(tokenA));
  await measure("resumeConsent", "consent", registry.resumeConsent(tokenA));
  await measure("revokeConsent", "consent", registry.revokeConsent(tokenA));

  // ------ submitAttestation ------
  const modelHashAtRevocation = await registry.modelHashAtRevocation(tokenA);
  const ua: UaInput = {
    dsTokenId: tokenA,
    modelHashPre: modelHashAtRevocation,
    modelHashPost: ethers.keccak256(ethers.toUtf8Bytes(`gas-${manifest.network}-post-${Date.now()}`)),
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
  await measure(
    "submitAttestation_3of5",
    "attestation",
    oracle.submitAttestation({ ...ua, uaHash }, sigs)
  );

  // tokenB is intentionally left ACTIVE to avoid further pending state.
  void tokenB;

  return {
    network: manifest.network,
    chainId: Number(chainId),
    measuredAt: new Date().toISOString(),
    nativeSymbol,
    nativePriceUsd,
    operations: ops,
  };
}

// =======================================================================
// Compare mode — merge per-network reports → comparison + markdown table
// =======================================================================

function comparisonRows(reports: NetworkGasReport[]) {
  // Union of operation keys, preserving the order from the first report.
  const order: string[] = [];
  const seen = new Set<string>();
  for (const r of reports) {
    for (const k of Object.keys(r.operations)) {
      if (!seen.has(k)) {
        order.push(k);
        seen.add(k);
      }
    }
  }
  return order.map((op) => {
    const row: any = { operation: op };
    for (const r of reports) {
      const m = r.operations[op];
      if (!m) {
        row[`${r.network}_gas`] = null;
        row[`${r.network}_native`] = null;
        row[`${r.network}_usd`] = null;
        continue;
      }
      const native = BigInt(m.nativeCostWei);
      row[`${r.network}_gas`] = m.gasUsed;
      row[`${r.network}_native`] = (Number(native) / 1e18).toFixed(8);
      row[`${r.network}_usd`] =
        r.nativePriceUsd != null
          ? ((Number(native) / 1e18) * r.nativePriceUsd).toFixed(6)
          : null;
    }
    return row;
  });
}

function renderMarkdownTable(reports: NetworkGasReport[]): string {
  // PureChain first to lead with the killer argument; Amoy after.
  const ordered = [...reports].sort((a, b) =>
    a.network === "purechain" ? -1 : b.network === "purechain" ? 1 : a.network.localeCompare(b.network)
  );
  const rows = comparisonRows(ordered);
  const headers: string[] = ["Operation", "Gas"];
  for (const r of ordered) {
    headers.push(`${r.network} (${r.nativeSymbol})`);
    headers.push(`${r.network} (USD)`);
  }
  const align =
    "|---|---:|" + ordered.map(() => "---:|---:|").join("");
  const body = rows
    .map((r) => {
      const gas = ordered
        .map((rp) => r[`${rp.network}_gas`])
        .find((v) => v != null);
      const cols = [`\`${r.operation}\``, gas != null ? fmtNum(gas) : "—"];
      for (const rp of ordered) {
        const native = r[`${rp.network}_native`];
        const usd = r[`${rp.network}_usd`];
        // PureChain: gasPrice always 0 → cost always $0 regardless of PCC USD price.
        if (rp.network === "purechain") {
          cols.push(native != null ? native : "—");
          cols.push("$0.000000");
        } else {
          cols.push(native != null ? native : "—");
          cols.push(usd != null ? `$${usd}` : "—");
        }
      }
      return "| " + cols.join(" | ") + " |";
    })
    .join("\n");
  return ["| " + headers.join(" | ") + " |", align, body].join("\n");
}

function lifecycleSummary(reports: NetworkGasReport[]) {
  const operations = ["grantConsent_1purpose", "revokeConsent", "submitAttestation_3of5"];
  const per: any = {};
  for (const r of reports) {
    const totalGas = operations
      .map((o) => (r.operations[o] ? BigInt(r.operations[o].gasUsed) : 0n))
      .reduce((a, b) => a + b, 0n);
    const totalNative = operations
      .map((o) =>
        r.operations[o] ? BigInt(r.operations[o].nativeCostWei) : 0n
      )
      .reduce((a, b) => a + b, 0n);
    const totalUsd =
      r.nativePriceUsd != null
        ? (Number(totalNative) / 1e18) * r.nativePriceUsd
        : null;
    per[r.network] = {
      operations,
      totalGas: totalGas.toString(),
      totalNativeWei: totalNative.toString(),
      totalUsd,
    };
  }
  return per;
}

async function compareMode() {
  const resultsDir = path.join(__dirname, "..", "..", "results", "blockchain");
  const files = fs
    .readdirSync(resultsDir)
    .filter((f) => f.startsWith("gas_") && f.endsWith(".json") && f !== "gas_comparison.json");
  if (files.length === 0) {
    throw new Error(`no per-network gas reports found under ${resultsDir}`);
  }
  const reports = files.map(
    (f) => JSON.parse(fs.readFileSync(path.join(resultsDir, f), "utf8")) as NetworkGasReport
  );

  const comparison = {
    generatedAt: new Date().toISOString(),
    networks: reports.map((r) => ({
      network: r.network,
      chainId: r.chainId,
      nativeSymbol: r.nativeSymbol,
      nativePriceUsd: r.nativePriceUsd,
      measuredAt: r.measuredAt,
    })),
    perOperation: comparisonRows(reports),
    lifecyclePerDS: lifecycleSummary(reports),
  };
  const outPath = path.join(resultsDir, "gas_comparison.json");
  fs.writeFileSync(outPath, JSON.stringify(comparison, null, 2) + "\n");
  console.log(`Comparison written: ${outPath}\n`);
  console.log("=== Table 2 (markdown) ===");
  console.log(renderMarkdownTable(reports));
  console.log("\n=== Per-DS lifecycle (grant + revoke + unlearn) ===");
  for (const [net, s] of Object.entries(comparison.lifecyclePerDS)) {
    const u = (s as any).totalUsd;
    console.log(
      `  ${net}: ${fmtNum((s as any).totalGas)} gas` +
        (u != null ? `, $${u.toFixed(6)}` : "")
    );
  }
}

// =======================================================================
// main
// =======================================================================

async function main(): Promise<void> {
  const hre = (globalThis as unknown as { hre: typeof import("hardhat") }).hre;
  assert(
    hre && hre.network && hre.ethers,
    "global.hre not populated — run via `npx hardhat run scripts/measure-gas.ts --network <name>`"
  );
  const { ethers, network } = hre;

  if (process.env.COMPARE_ONLY === "1") {
    await compareMode();
    return;
  }

  const networkName = network.name;
  const manifestPath = path.join(__dirname, "..", "deployments", `${networkName}.json`);
  assert(fs.existsSync(manifestPath), `missing deployment manifest: ${manifestPath}`);
  const manifest: DeploymentManifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));

  const nativeSymbol = networkName === "amoy" ? "MATIC" : networkName === "purechain" ? "PCC" : "ETH";
  const nativePriceUsd =
    networkName === "amoy" ? await fetchMaticUsd() : null;

  console.log(`\nNetwork:        ${networkName} (chainId=${manifest.chainId})`);
  console.log(`Native token:   ${nativeSymbol}`);
  console.log(`Token / USD:    ${nativePriceUsd ?? "n/a"}`);

  const report = await measureLive(ethers, manifest, nativeSymbol, nativePriceUsd);

  const resultsDir = path.join(__dirname, "..", "..", "results", "blockchain");
  fs.mkdirSync(resultsDir, { recursive: true });
  const outPath = path.join(resultsDir, `gas_${networkName}.json`);
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2) + "\n");
  console.log(`\nReport written: ${outPath}`);
  console.log(
    `Operations recorded: ${Object.keys(report.operations).length}`
  );
}

main()
  .then(() => process.exit(0))
  .catch((err: unknown) => {
    console.error(err);
    process.exit(1);
  });
