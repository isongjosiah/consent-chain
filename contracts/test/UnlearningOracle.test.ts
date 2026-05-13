/**
 * UnlearningOracle tests — covers docs/contracts-spec.md §4 (UA validation,
 * ECDSA multi-sig verification, replay & threshold checks, registry hand-off).
 */
import { expect } from "chai";
import { ethers, network } from "hardhat";
import { loadFixture } from "@nomicfoundation/hardhat-network-helpers";
import type { HardhatEthersSigner } from "@nomicfoundation/hardhat-ethers/signers";

import type {
  ConsentRegistry,
  MockFLGatekeeper,
  UnlearningOracle,
} from "../typechain-types";

// IUnlearningOracle.UnlearningMethod
const Method = { SISA_FED: 0, GRADIENT_ASCENT: 1, FEDERASER: 2 } as const;

// IConsentRegistry.ConsentState
const State = {
  INACTIVE: 0,
  ACTIVE: 1,
  SUSPENDED: 2,
  REVOKED: 3,
  EXPIRED: 4,
  UNLEARNED: 5,
} as const;

const MIA_SCALE = 10_000n;
const DEFAULT_MIA_THRESHOLD = 5500n;
const DEFAULT_VERIFIER_THRESHOLD = 3n;

const PURPOSES: string[] = ["adl-recognition", "anomaly-detection", "research"]
  .map((p) => ethers.keccak256(ethers.toUtf8Bytes(p)))
  .sort();

const MODEL_HASH_PRE = ethers.keccak256(ethers.toUtf8Bytes("model-round-60"));
const MODEL_HASH_POST = ethers.keccak256(ethers.toUtf8Bytes("model-round-60-post-unlearn"));

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

interface Ua extends UaInput {
  uaHash: string;
}

function computeUaHash(chainId: bigint, u: UaInput): string {
  const encoded = ethers.AbiCoder.defaultAbiCoder().encode(
    [
      "uint256", // chainid
      "uint256", // dsTokenId
      "bytes32", // modelHashPre
      "bytes32", // modelHashPost
      "uint8", // method
      "uint256", // shadowMia
      "uint256", // lira
      "uint256", // labelOnly
      "uint256", // timestamp
    ],
    [
      chainId,
      u.dsTokenId,
      u.modelHashPre,
      u.modelHashPost,
      u.method,
      u.shadowMiaScore,
      u.liraScore,
      u.labelOnlyMiaScore,
      u.timestamp,
    ]
  );
  return ethers.keccak256(encoded);
}

async function buildUa(u: UaInput): Promise<Ua> {
  const chainId = BigInt((await ethers.provider.getNetwork()).chainId);
  return { ...u, uaHash: computeUaHash(chainId, u) };
}

async function signUa(signers: HardhatEthersSigner[], uaHash: string): Promise<string[]> {
  const bytes = ethers.getBytes(uaHash);
  return Promise.all(signers.map((s) => s.signMessage(bytes)));
}

describe("UnlearningOracle", () => {
  async function deployFixture() {
    const [owner, alice, bob, stranger, v1, v2, v3, v4, v5] = await ethers.getSigners();

    const Registry = await ethers.getContractFactory("ConsentRegistry");
    const registry = (await Registry.deploy(owner.address)) as unknown as ConsentRegistry;

    const Gatekeeper = await ethers.getContractFactory("MockFLGatekeeper");
    const gatekeeper = (await Gatekeeper.deploy()) as unknown as MockFLGatekeeper;

    const Oracle = await ethers.getContractFactory("UnlearningOracle");
    const oracle = (await Oracle.deploy(
      await registry.getAddress(),
      owner.address
    )) as unknown as UnlearningOracle;

    return {
      owner, alice, bob, stranger,
      verifiers: [v1, v2, v3, v4, v5],
      registry, gatekeeper, oracle,
    };
  }

  async function wiredFixture() {
    const f = await deployFixture();
    await f.registry.connect(f.owner).setOracle(await f.oracle.getAddress());
    await f.registry
      .connect(f.owner)
      .setGatekeeper(await f.gatekeeper.getAddress());
    await f.gatekeeper.setCurrentModelHash(MODEL_HASH_PRE);
    return f;
  }

  /** Five verifiers added, threshold = 3 (the default). */
  async function verifiersSetFixture() {
    const f = await wiredFixture();
    for (const v of f.verifiers) {
      await f.oracle.connect(f.owner).addVerifier(v.address);
    }
    return f;
  }

  /** Alice owns tokenId=1 in REVOKED state, modelHashAtRevocation == MODEL_HASH_PRE. */
  async function revokedFixture() {
    const f = await verifiersSetFixture();
    await f.registry.connect(f.alice).grantConsent(PURPOSES, 0n);
    await f.registry.connect(f.alice).revokeConsent(1n);
    return { ...f, tokenId: 1n };
  }

  async function uaFixture() {
    const f = await revokedFixture();
    const ua = await buildUa({
      dsTokenId: f.tokenId,
      modelHashPre: MODEL_HASH_PRE,
      modelHashPost: MODEL_HASH_POST,
      method: Method.SISA_FED,
      shadowMiaScore: 5000n,    // 0.50
      liraScore: 4800n,         // 0.48
      labelOnlyMiaScore: 4900n, // 0.49
      timestamp: BigInt(Math.floor(Date.now() / 1000)),
    });
    // 3-of-5 — pick first three verifiers.
    const sigs = await signUa(f.verifiers.slice(0, 3), ua.uaHash);
    return { ...f, ua, sigs };
  }

  // -------------------------------------------------------------------
  // Construction & immutable wiring
  // -------------------------------------------------------------------

  describe("deployment", () => {
    it("sets registry, owner, and default thresholds", async () => {
      const { oracle, owner, registry } = await loadFixture(deployFixture);
      expect(await oracle.registry()).to.equal(await registry.getAddress());
      expect(await oracle.owner()).to.equal(owner.address);
      expect(await oracle.miaThreshold()).to.equal(DEFAULT_MIA_THRESHOLD);
      expect(await oracle.verifierThreshold()).to.equal(DEFAULT_VERIFIER_THRESHOLD);
      expect(await oracle.verifierCount()).to.equal(0n);
    });

    it("emits ThresholdsUpdated at construction", async () => {
      const [owner] = await ethers.getSigners();
      const Registry = await ethers.getContractFactory("ConsentRegistry");
      const r = (await Registry.deploy(owner.address)) as unknown as ConsentRegistry;
      const Oracle = await ethers.getContractFactory("UnlearningOracle");
      const oracle = (await Oracle.deploy(
        await r.getAddress(),
        owner.address
      )) as unknown as UnlearningOracle;
      const receipt = await oracle.deploymentTransaction()!.wait();
      const parsed = receipt!.logs
        .map((l) => {
          try {
            return oracle.interface.parseLog({
              topics: [...l.topics],
              data: l.data,
            });
          } catch {
            return null;
          }
        })
        .find((p) => p?.name === "ThresholdsUpdated");
      expect(parsed, "ThresholdsUpdated event was not emitted").to.not.be.null;
      expect(parsed!.args[0]).to.equal(DEFAULT_MIA_THRESHOLD);
      expect(parsed!.args[1]).to.equal(DEFAULT_VERIFIER_THRESHOLD);
    });
  });

  // -------------------------------------------------------------------
  // Verifier management
  // -------------------------------------------------------------------

  describe("addVerifier / removeVerifier", () => {
    it("only owner can add", async () => {
      const { oracle, alice, verifiers } = await loadFixture(deployFixture);
      await expect(
        oracle.connect(alice).addVerifier(verifiers[0].address)
      ).to.be.revertedWithCustomError(oracle, "OwnableUnauthorizedAccount");
    });

    it("adds a verifier and emits VerifierAdded", async () => {
      const { oracle, owner, verifiers } = await loadFixture(deployFixture);
      await expect(oracle.connect(owner).addVerifier(verifiers[0].address))
        .to.emit(oracle, "VerifierAdded")
        .withArgs(verifiers[0].address);
      expect(await oracle.isVerifier(verifiers[0].address)).to.be.true;
      expect(await oracle.verifierCount()).to.equal(1n);
      expect(await oracle.verifierAt(0)).to.equal(verifiers[0].address);
    });

    it("reverts on duplicate add", async () => {
      const { oracle, owner, verifiers } = await loadFixture(deployFixture);
      await oracle.connect(owner).addVerifier(verifiers[0].address);
      await expect(
        oracle.connect(owner).addVerifier(verifiers[0].address)
      ).to.be.revertedWithCustomError(oracle, "VerifierAlreadyRegistered");
    });

    it("removes a verifier and emits VerifierRemoved", async () => {
      const { oracle, owner, verifiers } = await loadFixture(verifiersSetFixture);
      await expect(oracle.connect(owner).removeVerifier(verifiers[2].address))
        .to.emit(oracle, "VerifierRemoved")
        .withArgs(verifiers[2].address);
      expect(await oracle.isVerifier(verifiers[2].address)).to.be.false;
      expect(await oracle.verifierCount()).to.equal(4n);
    });

    it("reverts when removing a non-registered address", async () => {
      const { oracle, owner, stranger } = await loadFixture(deployFixture);
      await expect(
        oracle.connect(owner).removeVerifier(stranger.address)
      ).to.be.revertedWithCustomError(oracle, "VerifierNotRegistered");
    });
  });

  // -------------------------------------------------------------------
  // Threshold management
  // -------------------------------------------------------------------

  describe("setMiaThreshold / setVerifierThreshold", () => {
    it("setMiaThreshold rejects zero and >MIA_SCALE", async () => {
      const { oracle, owner } = await loadFixture(deployFixture);
      await expect(
        oracle.connect(owner).setMiaThreshold(0n)
      ).to.be.revertedWithCustomError(oracle, "InvalidMiaThreshold");
      await expect(
        oracle.connect(owner).setMiaThreshold(MIA_SCALE + 1n)
      ).to.be.revertedWithCustomError(oracle, "InvalidMiaThreshold");
    });

    it("setMiaThreshold updates and emits", async () => {
      const { oracle, owner } = await loadFixture(deployFixture);
      await expect(oracle.connect(owner).setMiaThreshold(6000n))
        .to.emit(oracle, "ThresholdsUpdated")
        .withArgs(6000n, DEFAULT_VERIFIER_THRESHOLD);
      expect(await oracle.miaThreshold()).to.equal(6000n);
    });

    it("setVerifierThreshold rejects zero or > verifier count", async () => {
      const { oracle, owner } = await loadFixture(deployFixture);
      // No verifiers added: any positive t > 0 fails.
      await expect(
        oracle.connect(owner).setVerifierThreshold(1n)
      ).to.be.revertedWithCustomError(oracle, "InvalidVerifierThreshold");
    });

    it("setVerifierThreshold accepts t ∈ [1, |verifiers|]", async () => {
      const { oracle, owner } = await loadFixture(verifiersSetFixture);
      await expect(oracle.connect(owner).setVerifierThreshold(2n))
        .to.emit(oracle, "ThresholdsUpdated")
        .withArgs(DEFAULT_MIA_THRESHOLD, 2n);
      expect(await oracle.verifierThreshold()).to.equal(2n);
    });
  });

  // -------------------------------------------------------------------
  // submitAttestation — happy path
  // -------------------------------------------------------------------

  describe("submitAttestation — happy path", () => {
    it("retires the consent token via the registry and emits", async () => {
      const { oracle, registry, ua, sigs, tokenId, stranger } = await loadFixture(uaFixture);
      await expect(oracle.connect(stranger).submitAttestation(ua, sigs))
        .to.emit(oracle, "AttestationAccepted")
        .withArgs(
          tokenId,
          ua.uaHash,
          ua.method,
          ua.shadowMiaScore,
          ua.liraScore,
          ua.labelOnlyMiaScore
        )
        .and.to.emit(registry, "UnlearningCompleted")
        .withArgs(tokenId, ua.uaHash);

      expect(await registry.stateOf(tokenId)).to.equal(State.UNLEARNED);
      expect(await oracle.seenUaHash(ua.uaHash)).to.be.true;
      expect(await oracle.lastUaHash(tokenId)).to.equal(ua.uaHash);
    });

    it("works equally from the EXPIRED state", async () => {
      const { oracle, registry, gatekeeper, owner, alice, stranger, verifiers } =
        await loadFixture(verifiersSetFixture);
      // grant with TTL, then time-warp + checkExpiry
      const block = await ethers.provider.getBlock("latest");
      const ttl = BigInt(block!.timestamp + 60);
      await registry.connect(alice).grantConsent(PURPOSES, ttl);
      await network.provider.send("evm_setNextBlockTimestamp", [Number(ttl) + 1]);
      await network.provider.send("evm_mine");
      await registry.connect(stranger).checkExpiry(1n);

      const ua = await buildUa({
        dsTokenId: 1n,
        modelHashPre: MODEL_HASH_PRE,
        modelHashPost: MODEL_HASH_POST,
        method: Method.FEDERASER,
        shadowMiaScore: 4900n,
        liraScore: 5000n,
        labelOnlyMiaScore: 4500n,
        timestamp: BigInt(block!.timestamp),
      });
      const sigs = await signUa(verifiers.slice(0, 3), ua.uaHash);
      await expect(oracle.connect(owner).submitAttestation(ua, sigs)).to.not.be.reverted;
      expect(await registry.stateOf(1n)).to.equal(State.UNLEARNED);
    });

    it("works with > threshold signatures (4-of-5)", async () => {
      const { oracle, ua, verifiers } = await loadFixture(uaFixture);
      const fourSigs = await signUa(verifiers.slice(0, 4), ua.uaHash);
      await expect(oracle.submitAttestation(ua, fourSigs)).to.not.be.reverted;
    });
  });

  // -------------------------------------------------------------------
  // submitAttestation — UA-hash mismatch & replay
  // -------------------------------------------------------------------

  describe("submitAttestation — replay protection", () => {
    it("reverts UaReplay when declared uaHash does not match recomputed", async () => {
      const { oracle, ua, sigs } = await loadFixture(uaFixture);
      const tampered = { ...ua, uaHash: ethers.keccak256(ethers.toUtf8Bytes("bogus")) };
      await expect(
        oracle.submitAttestation(tampered, sigs)
      ).to.be.revertedWithCustomError(oracle, "UaReplay");
    });

    it("reverts UaReplay on second submit of an already-accepted UA", async () => {
      const { oracle, registry, ua, sigs, tokenId } = await loadFixture(uaFixture);
      await oracle.submitAttestation(ua, sigs);
      expect(await registry.stateOf(tokenId)).to.equal(State.UNLEARNED);
      await expect(
        oracle.submitAttestation(ua, sigs)
      ).to.be.revertedWithCustomError(oracle, "UaReplay");
    });

    it("uaHash differs across chains (chainid is bound into the preimage)", async () => {
      const { ua } = await loadFixture(uaFixture);
      const currentChainId = BigInt((await ethers.provider.getNetwork()).chainId);
      const otherChainHash = computeUaHash(currentChainId + 1n, {
        dsTokenId: ua.dsTokenId,
        modelHashPre: ua.modelHashPre,
        modelHashPost: ua.modelHashPost,
        method: ua.method,
        shadowMiaScore: ua.shadowMiaScore,
        liraScore: ua.liraScore,
        labelOnlyMiaScore: ua.labelOnlyMiaScore,
        timestamp: ua.timestamp,
      });
      expect(otherChainHash).to.not.equal(ua.uaHash);
    });
  });

  // -------------------------------------------------------------------
  // submitAttestation — MIA threshold
  // -------------------------------------------------------------------

  describe("submitAttestation — MIA threshold", () => {
    async function badScore(field: "shadowMiaScore" | "liraScore" | "labelOnlyMiaScore") {
      const f = await loadFixture(revokedFixture);
      const base = {
        dsTokenId: f.tokenId,
        modelHashPre: MODEL_HASH_PRE,
        modelHashPost: MODEL_HASH_POST,
        method: Method.SISA_FED,
        shadowMiaScore: 4900n,
        liraScore: 4900n,
        labelOnlyMiaScore: 4900n,
        timestamp: BigInt(Math.floor(Date.now() / 1000)),
      };
      base[field] = DEFAULT_MIA_THRESHOLD; // exactly at τ, must still revert (strict <)
      const ua = await buildUa(base);
      const sigs = await signUa(f.verifiers.slice(0, 3), ua.uaHash);
      await expect(
        f.oracle.submitAttestation(ua, sigs)
      ).to.be.revertedWithCustomError(f.oracle, "MiaThresholdExceeded");
    }

    it("rejects shadowMia >= τ", () => badScore("shadowMiaScore"));
    it("rejects lira >= τ", () => badScore("liraScore"));
    it("rejects labelOnly >= τ", () => badScore("labelOnlyMiaScore"));
  });

  // -------------------------------------------------------------------
  // submitAttestation — ModelHashMismatch & WrongConsentState
  // -------------------------------------------------------------------

  describe("submitAttestation — model & state binding", () => {
    it("reverts ModelHashMismatch when modelHashPre disagrees with registry", async () => {
      const { oracle, verifiers, tokenId } = await loadFixture(revokedFixture);
      const ua = await buildUa({
        dsTokenId: tokenId,
        modelHashPre: ethers.keccak256(ethers.toUtf8Bytes("wrong-model")),
        modelHashPost: MODEL_HASH_POST,
        method: Method.SISA_FED,
        shadowMiaScore: 5000n,
        liraScore: 5000n,
        labelOnlyMiaScore: 5000n,
        timestamp: BigInt(Math.floor(Date.now() / 1000)),
      });
      const sigs = await signUa(verifiers.slice(0, 3), ua.uaHash);
      await expect(
        oracle.submitAttestation(ua, sigs)
      ).to.be.revertedWithCustomError(oracle, "ModelHashMismatch");
    });

    it("reverts WrongConsentState when state is ACTIVE", async () => {
      const f = await loadFixture(verifiersSetFixture);
      await f.registry.connect(f.alice).grantConsent(PURPOSES, 0n);
      // Don't revoke — token is ACTIVE; modelHashAtRevocation is 0x0.
      const ua = await buildUa({
        dsTokenId: 1n,
        modelHashPre: ethers.ZeroHash, // matches ACTIVE token's modelHashAtRevocation=0x0
        modelHashPost: MODEL_HASH_POST,
        method: Method.SISA_FED,
        shadowMiaScore: 5000n,
        liraScore: 5000n,
        labelOnlyMiaScore: 5000n,
        timestamp: BigInt(Math.floor(Date.now() / 1000)),
      });
      const sigs = await signUa(f.verifiers.slice(0, 3), ua.uaHash);
      await expect(
        f.oracle.submitAttestation(ua, sigs)
      ).to.be.revertedWithCustomError(f.oracle, "WrongConsentState");
    });

    it("reverts WrongConsentState when the token is already UNLEARNED", async () => {
      const f = await loadFixture(uaFixture);
      // First call retires the token.
      await f.oracle.submitAttestation(f.ua, f.sigs);
      // Build a fresh UA against the same token (different timestamp → different uaHash).
      const ua2 = await buildUa({ ...f.ua, timestamp: f.ua.timestamp + 1n });
      const sigs2 = await signUa(f.verifiers.slice(0, 3), ua2.uaHash);
      await expect(
        f.oracle.submitAttestation(ua2, sigs2)
      ).to.be.revertedWithCustomError(f.oracle, "WrongConsentState");
    });
  });

  // -------------------------------------------------------------------
  // submitAttestation — signature checks
  // -------------------------------------------------------------------

  describe("submitAttestation — signature checks", () => {
    it("reverts InsufficientSignatures when distinct count < threshold", async () => {
      const { oracle, ua, verifiers } = await loadFixture(uaFixture);
      const twoSigs = await signUa(verifiers.slice(0, 2), ua.uaHash);
      await expect(
        oracle.submitAttestation(ua, twoSigs)
      )
        .to.be.revertedWithCustomError(oracle, "InsufficientSignatures")
        .withArgs(2n, DEFAULT_VERIFIER_THRESHOLD);
    });

    it("reverts NotVerifier when a signature recovers to a non-registered address", async () => {
      const { oracle, ua, verifiers, stranger } = await loadFixture(uaFixture);
      const sigs = await signUa([verifiers[0], verifiers[1], stranger], ua.uaHash);
      await expect(
        oracle.submitAttestation(ua, sigs)
      )
        .to.be.revertedWithCustomError(oracle, "NotVerifier")
        .withArgs(stranger.address);
    });

    it("reverts DuplicateSigner when the same verifier signs twice", async () => {
      const { oracle, ua, verifiers } = await loadFixture(uaFixture);
      const sigs = await signUa([verifiers[0], verifiers[0], verifiers[1]], ua.uaHash);
      await expect(
        oracle.submitAttestation(ua, sigs)
      )
        .to.be.revertedWithCustomError(oracle, "DuplicateSigner")
        .withArgs(verifiers[0].address);
    });

    it("removed verifier's signature is rejected even if previously valid", async () => {
      const { oracle, owner, ua, verifiers } = await loadFixture(uaFixture);
      // Sign with v1, v2, v3, then remove v1 before submitting.
      const sigs = await signUa(verifiers.slice(0, 3), ua.uaHash);
      await oracle.connect(owner).removeVerifier(verifiers[0].address);
      await expect(
        oracle.submitAttestation(ua, sigs)
      )
        .to.be.revertedWithCustomError(oracle, "NotVerifier")
        .withArgs(verifiers[0].address);
    });
  });
});
