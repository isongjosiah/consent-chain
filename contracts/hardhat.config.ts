import "@nomicfoundation/hardhat-toolbox";
import "hardhat-gas-reporter";
import * as dotenv from "dotenv";
import { HardhatUserConfig } from "hardhat/config";
import path from "path";

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });

const {
  PURECHAIN_RPC_URL,
  PURECHAIN_CHAIN_ID,
  PURECHAIN_DEPLOYER_PRIVATE_KEY,
  AMOY_RPC_URL,
  AMOY_CHAIN_ID,
  AMOY_DEPLOYER_PRIVATE_KEY,
  POLYGONSCAN_API_KEY,
  COINMARKETCAP_API_KEY,
  REPORT_GAS,
} = process.env;

const purechainAccounts =
  PURECHAIN_DEPLOYER_PRIVATE_KEY && PURECHAIN_DEPLOYER_PRIVATE_KEY.length > 0
    ? [PURECHAIN_DEPLOYER_PRIVATE_KEY]
    : [];

const amoyAccounts =
  AMOY_DEPLOYER_PRIVATE_KEY && AMOY_DEPLOYER_PRIVATE_KEY.length > 0
    ? [AMOY_DEPLOYER_PRIVATE_KEY]
    : [];

const config: HardhatUserConfig = {
  solidity: {
    version: "0.8.24",
    settings: {
      optimizer: { enabled: true, runs: 200 },
      // viaIR avoids "stack too deep" in UnlearningOracle.submitAttestation
      // (many UA struct fields + event args). Slightly slower compile;
      // produces tighter bytecode.
      viaIR: true,
      // Cancun is required by @openzeppelin/contracts ≥ 5.1 (uses mcopy).
      // Re-verify PureChain supports Cancun during D5 deploy; if it does not,
      // pin OZ to ~5.0.x and drop back to "paris".
      evmVersion: "cancun",
    },
  },
  networks: {
    hardhat: {
      chainId: 31337,
      allowUnlimitedContractSize: false,
    },
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
    },
    purechain: {
      url: PURECHAIN_RPC_URL ?? "",
      chainId: PURECHAIN_CHAIN_ID ? Number(PURECHAIN_CHAIN_ID) : undefined,
      accounts: purechainAccounts,
      gasPrice: 0,
    },
    amoy: {
      url: AMOY_RPC_URL ?? "https://rpc-amoy.polygon.technology",
      chainId: AMOY_CHAIN_ID ? Number(AMOY_CHAIN_ID) : 80002,
      accounts: amoyAccounts,
    },
  },
  etherscan: {
    apiKey: {
      polygonAmoy: POLYGONSCAN_API_KEY ?? "",
    },
    customChains: [
      {
        network: "polygonAmoy",
        chainId: 80002,
        urls: {
          apiURL: "https://api-amoy.polygonscan.com/api",
          browserURL: "https://amoy.polygonscan.com",
        },
      },
    ],
  },
  // hardhat-gas-reporter v1 schema; Polygon-on-chain numbers come from
  // scripts/measure-gas.ts against Amoy, not from the in-test reporter.
  gasReporter: {
    enabled: REPORT_GAS === "true",
    currency: "USD",
    coinmarketcap: COINMARKETCAP_API_KEY,
    outputFile: "gas-report.txt",
    noColors: true,
    token: "MATIC",
    gasPriceApi:
      "https://api.polygonscan.com/api?module=proxy&action=eth_gasPrice",
  },
  typechain: {
    outDir: "typechain-types",
    target: "ethers-v6",
  },
  mocha: {
    timeout: 120_000,
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts",
  },
};

export default config;
