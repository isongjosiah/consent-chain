/**
 * ConsentRegistry tests — covers every row of docs/contracts-spec.md §3.8
 * plus the soulbound and UNLEARNED-absorbing properties.
 */
import { expect } from "chai";
import { ethers } from "hardhat";
import { loadFixture, time } from "@nomicfoundation/hardhat-network-helpers";
import type { HardhatEthersSigner } from "@nomicfoundation/hardhat-ethers/signers";

import type { ConsentRegistry, MockFLGatekeeper } from "../typechain-types";

// ConsentState enum values — must match IConsentRegistry.ConsentState order.
const State = {
  INACTIVE: 0,
  ACTIVE: 1,
  SUSPENDED: 2,
  REVOKED: 3,
  EXPIRED: 4,
  UNLEARNED: 5,
} as const;

const ZERO = ethers.ZeroAddress;
const ZERO_HASH = ethers.ZeroHash;

function purposeHash(name: string): string {
  return ethers.keccak256(ethers.toUtf8Bytes(name));
}

/** Hashes of three example purposes, sorted ascending. */
const PURPOSES: string[] = ["adl-recognition", "anomaly-detection", "research"]
  .map(purposeHash)
  .sort();

const ROUND_PURPOSE = PURPOSES[0];

async function nowPlus(seconds: number): Promise<bigint> {
  return BigInt((await time.latest()) + seconds);
}

describe("ConsentRegistry", () => {
  // ---------------------------------------------------------------------
  // Fixtures
  // ---------------------------------------------------------------------

  async function deployFixture() {
    const [owner, alice, bob, oracle, stranger] = await ethers.getSigners();

    const Registry = await ethers.getContractFactory("ConsentRegistry");
    const registry = (await Registry.deploy(owner.address)) as unknown as ConsentRegistry;

    const Gatekeeper = await ethers.getContractFactory("MockFLGatekeeper");
    const gatekeeper = (await Gatekeeper.deploy()) as unknown as MockFLGatekeeper;

    return { registry, gatekeeper, owner, alice, bob, oracle, stranger };
  }

  async function wiredFixture() {
    const f = await deployFixture();
    await f.registry.connect(f.owner).setOracle(f.oracle.address);
    await f.registry
      .connect(f.owner)
      .setGatekeeper(await f.gatekeeper.getAddress());
    const modelHash = ethers.keccak256(ethers.toUtf8Bytes("model-round-60"));
    await f.gatekeeper.setCurrentModelHash(modelHash);
    return { ...f, modelHash };
  }

  /** Alice holds tokenId=1, ACTIVE, non-expiring, purposeSet over PURPOSES. */
  async function activeTokenFixture() {
    const f = await wiredFixture();
    const tx = await f.registry.connect(f.alice).grantConsent(PURPOSES, 0n);
    await tx.wait();
    return { ...f, tokenId: 1n };
  }

  async function suspendedTokenFixture() {
    const f = await activeTokenFixture();
    await f.registry.connect(f.alice).suspendConsent(f.tokenId);
    return f;
  }

  async function revokedTokenFixture() {
    const f = await activeTokenFixture();
    await f.registry.connect(f.alice).revokeConsent(f.tokenId);
    return f;
  }

  async function expiredTokenFixture() {
    const f = await wiredFixture();
    const ttl = await nowPlus(60 * 60); // +1h
    await f.registry.connect(f.alice).grantConsent(PURPOSES, ttl);
    await time.increaseTo(ttl + 1n);
    await f.registry.connect(f.stranger).checkExpiry(1n); // anyone may flip
    return { ...f, tokenId: 1n };
  }

  async function unlearnedTokenFixture() {
    const f = await revokedTokenFixture();
    const uaHash = ethers.keccak256(ethers.toUtf8Bytes("ua-hash-1"));
    await f.registry.connect(f.oracle).transitionToUnlearned(f.tokenId, uaHash);
    return { ...f, uaHash };
  }

  // ---------------------------------------------------------------------
  // Deployment
  // ---------------------------------------------------------------------

  describe("deployment", () => {
    it("sets initial owner; oracle and gatekeeper unset", async () => {
      const { registry, owner } = await loadFixture(deployFixture);
      expect(await registry.owner()).to.equal(owner.address);
      expect(await registry.oracle()).to.equal(ZERO);
      expect(await registry.gatekeeper()).to.equal(ZERO);
    });

    it("exposes ERC721 name/symbol", async () => {
      const { registry } = await loadFixture(deployFixture);
      expect(await registry.name()).to.equal("ConsentChain Consent");
      expect(await registry.symbol()).to.equal("CC-CONSENT");
    });
  });

  // ---------------------------------------------------------------------
  // One-shot wiring
  // ---------------------------------------------------------------------

  describe("setOracle / setGatekeeper", () => {
    it("only owner may set oracle", async () => {
      const { registry, alice, oracle } = await loadFixture(deployFixture);
      await expect(
        registry.connect(alice).setOracle(oracle.address)
      ).to.be.revertedWithCustomError(registry, "OwnableUnauthorizedAccount");
    });

    it("setOracle is one-shot and emits OracleBound", async () => {
      const { registry, owner, oracle, bob } = await loadFixture(deployFixture);
      await expect(registry.connect(owner).setOracle(oracle.address))
        .to.emit(registry, "OracleBound")
        .withArgs(oracle.address);
      expect(await registry.oracle()).to.equal(oracle.address);
      await expect(
        registry.connect(owner).setOracle(bob.address)
      ).to.be.revertedWithCustomError(registry, "OracleAlreadySet");
    });

    it("setGatekeeper is one-shot and emits GatekeeperBound", async () => {
      const { registry, gatekeeper, owner, bob } = await loadFixture(deployFixture);
      const gkAddr = await gatekeeper.getAddress();
      await expect(registry.connect(owner).setGatekeeper(gkAddr))
        .to.emit(registry, "GatekeeperBound")
        .withArgs(gkAddr);
      expect(await registry.gatekeeper()).to.equal(gkAddr);
      await expect(
        registry.connect(owner).setGatekeeper(bob.address)
      ).to.be.revertedWithCustomError(registry, "GatekeeperAlreadySet");
    });
  });

  // ---------------------------------------------------------------------
  // grantConsent  INACTIVE -> ACTIVE
  // ---------------------------------------------------------------------

  describe("grantConsent (INACTIVE -> ACTIVE)", () => {
    it("mints token id=1, sets ACTIVE, emits ConsentGranted", async () => {
      const { registry, alice } = await loadFixture(wiredFixture);
      const expectedPS = ethers.keccak256(
        ethers.AbiCoder.defaultAbiCoder().encode(["bytes32[]"], [PURPOSES])
      );
      await expect(registry.connect(alice).grantConsent(PURPOSES, 0n))
        .to.emit(registry, "ConsentGranted")
        .withArgs(1n, alice.address, expectedPS, 0n);
      expect(await registry.stateOf(1n)).to.equal(State.ACTIVE);
      expect(await registry.ownerOf(1n)).to.equal(alice.address);
      const rec = await registry.recordOf(1n);
      expect(rec.dsOwner).to.equal(alice.address);
      expect(rec.purposeSet).to.equal(expectedPS);
      expect(rec.ttl).to.equal(0n);
    });

    it("ids are monotonic across mints", async () => {
      const { registry, alice, bob } = await loadFixture(wiredFixture);
      await registry.connect(alice).grantConsent(PURPOSES, 0n);
      await registry.connect(bob).grantConsent(PURPOSES, 0n);
      expect(await registry.ownerOf(1n)).to.equal(alice.address);
      expect(await registry.ownerOf(2n)).to.equal(bob.address);
    });

    it("reverts EmptyPurposeSet for empty array", async () => {
      const { registry, alice } = await loadFixture(wiredFixture);
      await expect(
        registry.connect(alice).grantConsent([], 0n)
      ).to.be.revertedWithCustomError(registry, "EmptyPurposeSet");
    });

    it("reverts PurposeSetTooLarge for >16 entries", async () => {
      const { registry, alice } = await loadFixture(wiredFixture);
      const big = Array.from({ length: 17 }, (_, i) => purposeHash(`p${i}`)).sort();
      await expect(
        registry.connect(alice).grantConsent(big, 0n)
      )
        .to.be.revertedWithCustomError(registry, "PurposeSetTooLarge")
        .withArgs(17n);
    });

    it("reverts PurposesNotSorted for descending input", async () => {
      const { registry, alice } = await loadFixture(wiredFixture);
      const reversed = [...PURPOSES].reverse();
      await expect(
        registry.connect(alice).grantConsent(reversed, 0n)
      ).to.be.revertedWithCustomError(registry, "PurposesNotSorted");
    });

    it("reverts PurposesNotSorted on duplicate (strict ordering)", async () => {
      const { registry, alice } = await loadFixture(wiredFixture);
      const dup = [PURPOSES[0], PURPOSES[0]];
      await expect(
        registry.connect(alice).grantConsent(dup, 0n)
      ).to.be.revertedWithCustomError(registry, "PurposesNotSorted");
    });

    it("reverts TtlInThePast when ttl <= now", async () => {
      const { registry, alice } = await loadFixture(wiredFixture);
      const past = BigInt(await time.latest()) - 1n;
      await expect(
        registry.connect(alice).grantConsent(PURPOSES, past)
      ).to.be.revertedWithCustomError(registry, "TtlInThePast");
    });

    it("accepts a future ttl", async () => {
      const { registry, alice } = await loadFixture(wiredFixture);
      const ttl = await nowPlus(3600);
      await expect(registry.connect(alice).grantConsent(PURPOSES, ttl)).to.not.be.reverted;
      const rec = await registry.recordOf(1n);
      expect(rec.ttl).to.equal(ttl);
    });
  });

  // ---------------------------------------------------------------------
  // modifyConsent  ACTIVE -> ACTIVE
  // ---------------------------------------------------------------------

  describe("modifyConsent (ACTIVE -> ACTIVE)", () => {
    it("dsOwner updates purposes and ttl", async () => {
      const { registry, alice, tokenId } = await loadFixture(activeTokenFixture);
      const newPurposes = [purposeHash("a"), purposeHash("b")].sort();
      const newTtl = await nowPlus(7200);
      await expect(registry.connect(alice).modifyConsent(tokenId, newPurposes, newTtl))
        .to.emit(registry, "ConsentModified");
      const rec = await registry.recordOf(tokenId);
      expect(rec.ttl).to.equal(newTtl);
    });

    it("reverts NotTokenOwner when caller is not dsOwner", async () => {
      const { registry, bob, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        registry.connect(bob).modifyConsent(tokenId, PURPOSES, 0n)
      ).to.be.revertedWithCustomError(registry, "NotTokenOwner");
    });

    it("reverts when state != ACTIVE", async () => {
      const { registry, alice, tokenId } = await loadFixture(suspendedTokenFixture);
      await expect(
        registry.connect(alice).modifyConsent(tokenId, PURPOSES, 0n)
      ).to.be.revertedWithCustomError(registry, "InvalidStateTransition");
    });
  });

  // ---------------------------------------------------------------------
  // suspendConsent / resumeConsent
  // ---------------------------------------------------------------------

  describe("suspendConsent / resumeConsent", () => {
    it("dsOwner can suspend ACTIVE token", async () => {
      const { registry, alice, tokenId } = await loadFixture(activeTokenFixture);
      await expect(registry.connect(alice).suspendConsent(tokenId))
        .to.emit(registry, "ConsentSuspended")
        .withArgs(tokenId, alice.address);
      expect(await registry.stateOf(tokenId)).to.equal(State.SUSPENDED);
    });

    it("contract owner (regulator) can suspend", async () => {
      const { registry, owner, tokenId } = await loadFixture(activeTokenFixture);
      await expect(registry.connect(owner).suspendConsent(tokenId)).to.emit(
        registry,
        "ConsentSuspended"
      );
    });

    it("stranger cannot suspend", async () => {
      const { registry, stranger, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        registry.connect(stranger).suspendConsent(tokenId)
      ).to.be.revertedWithCustomError(registry, "NotTokenOwner");
    });

    it("suspend reverts when not ACTIVE", async () => {
      const { registry, alice, tokenId } = await loadFixture(suspendedTokenFixture);
      await expect(
        registry.connect(alice).suspendConsent(tokenId)
      ).to.be.revertedWithCustomError(registry, "InvalidStateTransition");
    });

    it("dsOwner can resume from SUSPENDED back to ACTIVE", async () => {
      const { registry, alice, tokenId } = await loadFixture(suspendedTokenFixture);
      await expect(registry.connect(alice).resumeConsent(tokenId)).to.emit(
        registry,
        "ConsentResumed"
      );
      expect(await registry.stateOf(tokenId)).to.equal(State.ACTIVE);
    });

    it("resume reverts when not SUSPENDED", async () => {
      const { registry, alice, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        registry.connect(alice).resumeConsent(tokenId)
      ).to.be.revertedWithCustomError(registry, "InvalidStateTransition");
    });

    it("resume reverts when caller is not dsOwner", async () => {
      const { registry, bob, tokenId } = await loadFixture(suspendedTokenFixture);
      await expect(
        registry.connect(bob).resumeConsent(tokenId)
      ).to.be.revertedWithCustomError(registry, "NotTokenOwner");
    });
  });

  // ---------------------------------------------------------------------
  // revokeConsent  ACTIVE -> REVOKED
  // ---------------------------------------------------------------------

  describe("revokeConsent (ACTIVE -> REVOKED)", () => {
    it("captures model hash, notifies gatekeeper, emits both events", async () => {
      const { registry, gatekeeper, alice, tokenId, modelHash } = await loadFixture(
        activeTokenFixture
      );
      await expect(registry.connect(alice).revokeConsent(tokenId))
        .to.emit(registry, "ConsentRevoked")
        .withArgs(tokenId, modelHash)
        .and.to.emit(registry, "UnlearningRequired")
        .withArgs(tokenId, modelHash);

      expect(await registry.stateOf(tokenId)).to.equal(State.REVOKED);
      expect(await registry.modelHashAtRevocation(tokenId)).to.equal(modelHash);
      expect(await gatekeeper.pendingCalls(tokenId)).to.equal(1n);
      expect(await gatekeeper.totalPendingCalls()).to.equal(1n);
    });

    it("only dsOwner can revoke", async () => {
      const { registry, stranger, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        registry.connect(stranger).revokeConsent(tokenId)
      ).to.be.revertedWithCustomError(registry, "NotTokenOwner");
    });

    it("reverts when not ACTIVE", async () => {
      const { registry, alice, tokenId } = await loadFixture(suspendedTokenFixture);
      await expect(
        registry.connect(alice).revokeConsent(tokenId)
      ).to.be.revertedWithCustomError(registry, "InvalidStateTransition");
    });

    it("reverts GatekeeperNotSet when wiring incomplete", async () => {
      const { registry, owner, oracle, alice } = await loadFixture(deployFixture);
      // Only oracle wired — gatekeeper deliberately left unset.
      await registry.connect(owner).setOracle(oracle.address);
      await registry.connect(alice).grantConsent(PURPOSES, 0n);
      await expect(
        registry.connect(alice).revokeConsent(1n)
      ).to.be.revertedWithCustomError(registry, "GatekeeperNotSet");
    });
  });

  // ---------------------------------------------------------------------
  // checkExpiry  ACTIVE -> EXPIRED  (anyone may call)
  // ---------------------------------------------------------------------

  describe("checkExpiry (ACTIVE -> EXPIRED)", () => {
    it("anyone can flip an expired token", async () => {
      const { registry, gatekeeper, stranger, modelHash } = await loadFixture(wiredFixture);
      const ttl = await nowPlus(60);
      await registry.connect((await ethers.getSigners())[1]).grantConsent(PURPOSES, ttl);
      await time.increaseTo(ttl + 1n);
      await expect(registry.connect(stranger).checkExpiry(1n))
        .to.emit(registry, "ConsentExpired")
        .withArgs(1n, modelHash)
        .and.to.emit(registry, "UnlearningRequired")
        .withArgs(1n, modelHash);
      expect(await registry.stateOf(1n)).to.equal(State.EXPIRED);
      expect(await gatekeeper.totalPendingCalls()).to.equal(1n);
    });

    it("reverts when ttl is in the future", async () => {
      const { registry, alice, stranger } = await loadFixture(wiredFixture);
      const ttl = await nowPlus(60 * 60);
      await registry.connect(alice).grantConsent(PURPOSES, ttl);
      await expect(
        registry.connect(stranger).checkExpiry(1n)
      ).to.be.revertedWithCustomError(registry, "InvalidStateTransition");
    });

    it("reverts when ttl == 0 (non-expiring)", async () => {
      const { registry, stranger, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        registry.connect(stranger).checkExpiry(tokenId)
      ).to.be.revertedWithCustomError(registry, "InvalidStateTransition");
    });

    it("reverts when state != ACTIVE", async () => {
      const { registry, stranger, tokenId } = await loadFixture(suspendedTokenFixture);
      await expect(
        registry.connect(stranger).checkExpiry(tokenId)
      ).to.be.revertedWithCustomError(registry, "InvalidStateTransition");
    });
  });

  // ---------------------------------------------------------------------
  // transitionToUnlearned  REVOKED|EXPIRED -> UNLEARNED  (oracle-only)
  // ---------------------------------------------------------------------

  describe("transitionToUnlearned (REVOKED|EXPIRED -> UNLEARNED)", () => {
    it("oracle can transition from REVOKED, emits UnlearningCompleted", async () => {
      const { registry, oracle, tokenId } = await loadFixture(revokedTokenFixture);
      const uaHash = ethers.keccak256(ethers.toUtf8Bytes("ua-1"));
      await expect(registry.connect(oracle).transitionToUnlearned(tokenId, uaHash))
        .to.emit(registry, "UnlearningCompleted")
        .withArgs(tokenId, uaHash);
      expect(await registry.stateOf(tokenId)).to.equal(State.UNLEARNED);
    });

    it("oracle can transition from EXPIRED", async () => {
      const { registry, oracle, tokenId } = await loadFixture(expiredTokenFixture);
      const uaHash = ethers.keccak256(ethers.toUtf8Bytes("ua-2"));
      await registry.connect(oracle).transitionToUnlearned(tokenId, uaHash);
      expect(await registry.stateOf(tokenId)).to.equal(State.UNLEARNED);
    });

    it("non-oracle caller reverts", async () => {
      const { registry, alice, tokenId } = await loadFixture(revokedTokenFixture);
      await expect(
        registry.connect(alice).transitionToUnlearned(tokenId, ZERO_HASH)
      ).to.be.revertedWithCustomError(registry, "NotAuthorizedOracle");
    });

    it("reverts when current state is not REVOKED or EXPIRED", async () => {
      const { registry, oracle, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        registry.connect(oracle).transitionToUnlearned(tokenId, ZERO_HASH)
      ).to.be.revertedWithCustomError(registry, "InvalidStateTransition");
    });
  });

  // ---------------------------------------------------------------------
  // Soulbound enforcement
  // ---------------------------------------------------------------------

  describe("soulbound enforcement", () => {
    it("transferFrom reverts SoulboundTransferDisabled", async () => {
      const { registry, alice, bob, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        registry.connect(alice).transferFrom(alice.address, bob.address, tokenId)
      ).to.be.revertedWithCustomError(registry, "SoulboundTransferDisabled");
    });

    it("safeTransferFrom reverts SoulboundTransferDisabled", async () => {
      const { registry, alice, bob, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        registry
          .connect(alice)
          ["safeTransferFrom(address,address,uint256)"](alice.address, bob.address, tokenId)
      ).to.be.revertedWithCustomError(registry, "SoulboundTransferDisabled");
    });

    it("safeTransferFrom(...,data) reverts SoulboundTransferDisabled", async () => {
      const { registry, alice, bob, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        registry
          .connect(alice)
          ["safeTransferFrom(address,address,uint256,bytes)"](
            alice.address,
            bob.address,
            tokenId,
            "0x"
          )
      ).to.be.revertedWithCustomError(registry, "SoulboundTransferDisabled");
    });

    it("approve reverts SoulboundTransferDisabled", async () => {
      const { registry, alice, bob, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        registry.connect(alice).approve(bob.address, tokenId)
      ).to.be.revertedWithCustomError(registry, "SoulboundTransferDisabled");
    });

    it("setApprovalForAll reverts SoulboundTransferDisabled", async () => {
      const { registry, alice, bob } = await loadFixture(activeTokenFixture);
      await expect(
        registry.connect(alice).setApprovalForAll(bob.address, true)
      ).to.be.revertedWithCustomError(registry, "SoulboundTransferDisabled");
    });
  });

  // ---------------------------------------------------------------------
  // UNLEARNED is absorbing — every external call reverts
  // ---------------------------------------------------------------------

  describe("UNLEARNED is absorbing", () => {
    it("modifyConsent reverts TerminalState", async () => {
      const { registry, alice, tokenId } = await loadFixture(unlearnedTokenFixture);
      await expect(
        registry.connect(alice).modifyConsent(tokenId, PURPOSES, 0n)
      ).to.be.revertedWithCustomError(registry, "TerminalState");
    });

    it("suspendConsent reverts TerminalState", async () => {
      const { registry, alice, tokenId } = await loadFixture(unlearnedTokenFixture);
      await expect(
        registry.connect(alice).suspendConsent(tokenId)
      ).to.be.revertedWithCustomError(registry, "TerminalState");
    });

    it("resumeConsent reverts TerminalState", async () => {
      const { registry, alice, tokenId } = await loadFixture(unlearnedTokenFixture);
      await expect(
        registry.connect(alice).resumeConsent(tokenId)
      ).to.be.revertedWithCustomError(registry, "TerminalState");
    });

    it("revokeConsent reverts TerminalState", async () => {
      const { registry, alice, tokenId } = await loadFixture(unlearnedTokenFixture);
      await expect(
        registry.connect(alice).revokeConsent(tokenId)
      ).to.be.revertedWithCustomError(registry, "TerminalState");
    });

    it("checkExpiry reverts TerminalState", async () => {
      const { registry, stranger, tokenId } = await loadFixture(unlearnedTokenFixture);
      await expect(
        registry.connect(stranger).checkExpiry(tokenId)
      ).to.be.revertedWithCustomError(registry, "TerminalState");
    });

    it("transitionToUnlearned reverts TerminalState", async () => {
      const { registry, oracle, tokenId } = await loadFixture(unlearnedTokenFixture);
      await expect(
        registry.connect(oracle).transitionToUnlearned(tokenId, ZERO_HASH)
      ).to.be.revertedWithCustomError(registry, "TerminalState");
    });
  });

  // ---------------------------------------------------------------------
  // isEligible — used by FLGatekeeper
  // ---------------------------------------------------------------------

  describe("isEligible", () => {
    it("true for ACTIVE token + matching purpose array", async () => {
      const { registry, tokenId } = await loadFixture(activeTokenFixture);
      expect(await registry.isEligible(tokenId, ROUND_PURPOSE, PURPOSES, 0)).to.be.true;
    });

    it("false when state != ACTIVE", async () => {
      const { registry, tokenId } = await loadFixture(suspendedTokenFixture);
      expect(await registry.isEligible(tokenId, ROUND_PURPOSE, PURPOSES, 0)).to.be.false;
    });

    it("false when ttl has elapsed", async () => {
      const { registry, alice } = await loadFixture(wiredFixture);
      const ttl = await nowPlus(30);
      await registry.connect(alice).grantConsent(PURPOSES, ttl);
      await time.increaseTo(ttl + 1n);
      expect(await registry.isEligible(1n, ROUND_PURPOSE, PURPOSES, 0)).to.be.false;
    });

    it("false on purpose array hash mismatch", async () => {
      const { registry, tokenId } = await loadFixture(activeTokenFixture);
      const tampered = [...PURPOSES];
      tampered[1] = purposeHash("zzz-different");
      tampered.sort();
      expect(await registry.isEligible(tokenId, ROUND_PURPOSE, tampered, 0)).to.be.false;
    });

    it("false when purposeArray[index] != roundPurpose", async () => {
      const { registry, tokenId } = await loadFixture(activeTokenFixture);
      // index 1 is purposes[1] = anomaly-detection; roundPurpose is purposes[0]
      expect(await registry.isEligible(tokenId, ROUND_PURPOSE, PURPOSES, 1)).to.be.false;
    });

    it("false on out-of-range index (does not revert)", async () => {
      const { registry, tokenId } = await loadFixture(activeTokenFixture);
      expect(await registry.isEligible(tokenId, ROUND_PURPOSE, PURPOSES, 99)).to.be.false;
    });

    it("false for unknown token (does not revert)", async () => {
      const { registry } = await loadFixture(wiredFixture);
      expect(await registry.isEligible(999n, ROUND_PURPOSE, PURPOSES, 0)).to.be.false;
    });
  });

  // ---------------------------------------------------------------------
  // Views
  // ---------------------------------------------------------------------

  describe("views", () => {
    it("recordOf reverts UnknownToken for unminted id", async () => {
      const { registry } = await loadFixture(wiredFixture);
      await expect(registry.recordOf(42n)).to.be.revertedWithCustomError(
        registry,
        "UnknownToken"
      );
    });

    it("stateOf returns current state", async () => {
      const { registry, tokenId } = await loadFixture(activeTokenFixture);
      expect(await registry.stateOf(tokenId)).to.equal(State.ACTIVE);
    });

    it("modelHashAtRevocation is 0 before revoke", async () => {
      const { registry, tokenId } = await loadFixture(activeTokenFixture);
      expect(await registry.modelHashAtRevocation(tokenId)).to.equal(ZERO_HASH);
    });

    it("modelHashAtRevocation reflects captured hash after revoke", async () => {
      const { registry, tokenId, modelHash } = await loadFixture(revokedTokenFixture);
      expect(await registry.modelHashAtRevocation(tokenId)).to.equal(modelHash);
    });
  });
});
