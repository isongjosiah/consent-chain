/**
 * FLGatekeeper unit tests — docs/contracts-spec.md §5.
 *
 * Strategy: wire each FLGatekeeper instance against a real ConsentRegistry
 * (small/fast enough), but use Hardhat's impersonateAccount to invoke the
 * onlyRegistry hooks (recordPendingUnlearning / unblock) directly. This
 * keeps the unit tests isolated from a full lifecycle; Integration.test.ts
 * exercises the registry-driven path.
 */
import { expect } from "chai";
import { ethers, network } from "hardhat";
import {
  impersonateAccount,
  loadFixture,
  mine,
  setBalance,
  stopImpersonatingAccount,
} from "@nomicfoundation/hardhat-network-helpers";
import type { HardhatEthersSigner } from "@nomicfoundation/hardhat-ethers/signers";

import type {
  ConsentRegistry,
  FLGatekeeper,
  MockFLGatekeeper,
} from "../typechain-types";

const State = {
  INACTIVE: 0,
  ACTIVE: 1,
  SUSPENDED: 2,
  REVOKED: 3,
  EXPIRED: 4,
  UNLEARNED: 5,
} as const;

const ZERO_HASH = ethers.ZeroHash;

const PURPOSES: string[] = ["adl-recognition", "anomaly-detection"]
  .map((p) => ethers.keccak256(ethers.toUtf8Bytes(p)))
  .sort();
const ROUND_PURPOSE = PURPOSES[0];

async function impersonate(addr: string): Promise<HardhatEthersSigner> {
  await impersonateAccount(addr);
  await setBalance(addr, ethers.parseEther("10"));
  return ethers.getSigner(addr);
}

function context(
  roundId: bigint,
  roundPurpose: string = ROUND_PURPOSE,
  expectedHash: string = ZERO_HASH
) {
  return { roundId, roundPurpose, expectedModelHashPre: expectedHash };
}

describe("FLGatekeeper", () => {
  async function deployFixture() {
    const [owner, alice, bob, coordinator, stranger] = await ethers.getSigners();

    const Registry = await ethers.getContractFactory("ConsentRegistry");
    const registry = (await Registry.deploy(owner.address)) as unknown as ConsentRegistry;

    const Gatekeeper = await ethers.getContractFactory("FLGatekeeper");
    // Coordinator is the gatekeeper's owner (drives validateRound / updateModelHash).
    const gatekeeper = (await Gatekeeper.deploy(
      await registry.getAddress(),
      coordinator.address
    )) as unknown as FLGatekeeper;

    // Wire the registry. Gatekeeper is set so revoke/expire can callback.
    // Oracle is not needed for these unit tests; tests that need REVOKED
    // tokens don't actually call submitAttestation.
    const [, , , , , dummyOracle] = await ethers.getSigners();
    await registry.connect(owner).setOracle(dummyOracle.address);
    await registry.connect(owner).setGatekeeper(await gatekeeper.getAddress());

    return { owner, alice, bob, coordinator, stranger, registry, gatekeeper };
  }

  async function activeTokenFixture() {
    const f = await deployFixture();
    await f.registry.connect(f.alice).grantConsent(PURPOSES, 0n);
    return { ...f, tokenId: 1n };
  }

  // -------------------------------------------------------------------
  // Deployment
  // -------------------------------------------------------------------

  describe("deployment", () => {
    it("sets registry, owner, default deltaMax; emits DeltaMaxUpdated", async () => {
      const { gatekeeper, coordinator, registry } = await loadFixture(deployFixture);
      expect(await gatekeeper.registry()).to.equal(await registry.getAddress());
      expect(await gatekeeper.owner()).to.equal(coordinator.address);
      expect(await gatekeeper.deltaMaxBlocks()).to.equal(1000n);
      expect(await gatekeeper.currentRoundId()).to.equal(0n);
      expect(await gatekeeper.currentModelHash()).to.equal(ZERO_HASH);
      expect(await gatekeeper.pendingUnlearningCount()).to.equal(0n);
    });
  });

  // -------------------------------------------------------------------
  // updateModelHash
  // -------------------------------------------------------------------

  describe("updateModelHash", () => {
    it("only owner can call", async () => {
      const { gatekeeper, alice } = await loadFixture(deployFixture);
      await expect(
        gatekeeper.connect(alice).updateModelHash(1n, ethers.id("h"))
      ).to.be.revertedWithCustomError(gatekeeper, "OwnableUnauthorizedAccount");
    });

    it("genesis commit at roundId=0 emits and stores (no validateRound needed)", async () => {
      const { gatekeeper, coordinator } = await loadFixture(deployFixture);
      const h = ethers.id("model-genesis");
      await expect(gatekeeper.connect(coordinator).updateModelHash(0n, h))
        .to.emit(gatekeeper, "ModelHashUpdated")
        .withArgs(0n, h);
      expect(await gatekeeper.currentModelHash()).to.equal(h);
      expect(await gatekeeper.currentRoundId()).to.equal(0n);
    });

    it("commits hash for the round just validated", async () => {
      const { gatekeeper, coordinator, tokenId } = await loadFixture(activeTokenFixture);
      await gatekeeper
        .connect(coordinator)
        .validateRound(context(1n), [tokenId], [PURPOSES], [0n]);
      const h = ethers.id("h-round-1");
      await expect(gatekeeper.connect(coordinator).updateModelHash(1n, h))
        .to.emit(gatekeeper, "ModelHashUpdated")
        .withArgs(1n, h);
      expect(await gatekeeper.currentModelHash()).to.equal(h);
    });

    it("rejects roundId != currentRoundId", async () => {
      const { gatekeeper, coordinator } = await loadFixture(deployFixture);
      // currentRoundId is 0; only roundId=0 is accepted.
      await expect(
        gatekeeper.connect(coordinator).updateModelHash(1n, ethers.id("h1"))
      ).to.be.revertedWithCustomError(gatekeeper, "RoundIdNotMonotonic");
      await expect(
        gatekeeper.connect(coordinator).updateModelHash(5n, ethers.id("h5"))
      ).to.be.revertedWithCustomError(gatekeeper, "RoundIdNotMonotonic");
    });
  });

  // -------------------------------------------------------------------
  // validateRound
  // -------------------------------------------------------------------

  describe("validateRound", () => {
    it("only owner can call", async () => {
      const { gatekeeper, alice, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        gatekeeper
          .connect(alice)
          .validateRound(context(1n), [tokenId], [PURPOSES], [0n])
      ).to.be.revertedWithCustomError(gatekeeper, "OwnableUnauthorizedAccount");
    });

    it("reverts EmptyParticipants on empty array", async () => {
      const { gatekeeper, coordinator } = await loadFixture(deployFixture);
      await expect(
        gatekeeper.connect(coordinator).validateRound(context(1n), [], [], [])
      ).to.be.revertedWithCustomError(gatekeeper, "EmptyParticipants");
    });

    it("reverts TooManyParticipants > MAX (1024)", async () => {
      const { gatekeeper, coordinator } = await loadFixture(deployFixture);
      const ids = Array.from({ length: 1025 }, (_, i) => BigInt(i + 1));
      const arrays = ids.map(() => PURPOSES);
      const idx = ids.map(() => 0n);
      await expect(
        gatekeeper.connect(coordinator).validateRound(context(1n), ids, arrays, idx)
      )
        .to.be.revertedWithCustomError(gatekeeper, "TooManyParticipants")
        .withArgs(1025n);
    });

    it("reverts LengthMismatch when arrays disagree", async () => {
      const { gatekeeper, coordinator } = await loadFixture(deployFixture);
      await expect(
        gatekeeper.connect(coordinator).validateRound(context(1n), [1n], [PURPOSES, PURPOSES], [0n])
      ).to.be.revertedWithCustomError(gatekeeper, "LengthMismatch");
      await expect(
        gatekeeper.connect(coordinator).validateRound(context(1n), [1n], [PURPOSES], [0n, 0n])
      ).to.be.revertedWithCustomError(gatekeeper, "LengthMismatch");
    });

    it("reverts RoundIdNotMonotonic when ctx.roundId <= currentRoundId", async () => {
      const { gatekeeper, coordinator, tokenId } = await loadFixture(activeTokenFixture);
      await gatekeeper.connect(coordinator).validateRound(context(5n), [tokenId], [PURPOSES], [0n]);
      await expect(
        gatekeeper.connect(coordinator).validateRound(context(5n), [tokenId], [PURPOSES], [0n])
      ).to.be.revertedWithCustomError(gatekeeper, "RoundIdNotMonotonic");
      await expect(
        gatekeeper.connect(coordinator).validateRound(context(3n), [tokenId], [PURPOSES], [0n])
      ).to.be.revertedWithCustomError(gatekeeper, "RoundIdNotMonotonic");
    });

    it("happy path validates and emits RoundValidated", async () => {
      const { gatekeeper, coordinator, tokenId } = await loadFixture(activeTokenFixture);
      await expect(
        gatekeeper.connect(coordinator).validateRound(context(1n), [tokenId], [PURPOSES], [0n])
      )
        .to.emit(gatekeeper, "RoundValidated")
        .withArgs(1n, 1n);
      expect(await gatekeeper.currentRoundId()).to.equal(1n);
    });

    it("reverts ParticipantIneligible when a participant is SUSPENDED", async () => {
      const { gatekeeper, registry, coordinator, alice, tokenId } = await loadFixture(
        activeTokenFixture
      );
      await registry.connect(alice).suspendConsent(tokenId);
      await expect(
        gatekeeper.connect(coordinator).validateRound(context(1n), [tokenId], [PURPOSES], [0n])
      )
        .to.be.revertedWithCustomError(gatekeeper, "ParticipantIneligible")
        .withArgs(tokenId, State.SUSPENDED);
    });

    it("reverts PendingUnlearningBlocksRound when any obligation outstanding", async () => {
      const { gatekeeper, registry, coordinator, alice, tokenId } = await loadFixture(
        activeTokenFixture
      );
      // Trigger pending-unlearning via the real registry → gatekeeper callback.
      await registry.connect(alice).revokeConsent(tokenId);
      expect(await gatekeeper.pendingUnlearningCount()).to.equal(1n);
      await expect(
        gatekeeper.connect(coordinator).validateRound(context(1n), [tokenId], [PURPOSES], [0n])
      )
        .to.be.revertedWithCustomError(gatekeeper, "PendingUnlearningBlocksRound")
        .withArgs(1n);
    });

    it("reverts ExpectedModelHashMismatch when ctx.expectedModelHashPre disagrees", async () => {
      const { gatekeeper, coordinator, tokenId } = await loadFixture(activeTokenFixture);
      const claimed = ethers.id("not-the-current-hash");
      await expect(
        gatekeeper
          .connect(coordinator)
          .validateRound(context(1n, ROUND_PURPOSE, claimed), [tokenId], [PURPOSES], [0n])
      )
        .to.be.revertedWithCustomError(gatekeeper, "ExpectedModelHashMismatch")
        .withArgs(claimed, ZERO_HASH);
    });

    it("accepts ctx.expectedModelHashPre when it matches", async () => {
      const { gatekeeper, coordinator, tokenId } = await loadFixture(activeTokenFixture);
      const h = ethers.id("genesis");
      // Genesis commit at roundId=0 (no prior validateRound needed).
      await gatekeeper.connect(coordinator).updateModelHash(0n, h);
      await expect(
        gatekeeper
          .connect(coordinator)
          .validateRound(context(1n, ROUND_PURPOSE, h), [tokenId], [PURPOSES], [0n])
      ).to.not.be.reverted;
    });
  });

  // -------------------------------------------------------------------
  // recordPendingUnlearning / unblock — onlyRegistry
  // -------------------------------------------------------------------

  describe("recordPendingUnlearning / unblock (onlyRegistry)", () => {
    it("recordPendingUnlearning reverts NotRegistry for non-registry caller", async () => {
      const { gatekeeper, alice } = await loadFixture(deployFixture);
      await expect(
        gatekeeper.connect(alice).recordPendingUnlearning(1n)
      ).to.be.revertedWithCustomError(gatekeeper, "NotRegistry");
    });

    it("unblock reverts NotRegistry for non-registry caller", async () => {
      const { gatekeeper, alice } = await loadFixture(deployFixture);
      await expect(
        gatekeeper.connect(alice).unblock(1n)
      ).to.be.revertedWithCustomError(gatekeeper, "NotRegistry");
    });

    it("registry can record, increments counter, emits", async () => {
      const { gatekeeper, registry } = await loadFixture(deployFixture);
      const registrySigner = await impersonate(await registry.getAddress());
      const tx = await gatekeeper.connect(registrySigner).recordPendingUnlearning(42n);
      const receipt = await tx.wait();
      const sinceBlock = BigInt(receipt!.blockNumber);
      await expect(tx)
        .to.emit(gatekeeper, "UnlearningPendingRecorded")
        .withArgs(42n, sinceBlock);
      expect(await gatekeeper.pendingUnlearningCount()).to.equal(1n);
      expect(await gatekeeper.pendingUnlearningSince(42n)).to.equal(sinceBlock);
      await stopImpersonatingAccount(await registry.getAddress());
    });

    it("recordPendingUnlearning is idempotent on the same token", async () => {
      const { gatekeeper, registry } = await loadFixture(deployFixture);
      const registrySigner = await impersonate(await registry.getAddress());
      await gatekeeper.connect(registrySigner).recordPendingUnlearning(42n);
      await gatekeeper.connect(registrySigner).recordPendingUnlearning(42n);
      expect(await gatekeeper.pendingUnlearningCount()).to.equal(1n);
      await stopImpersonatingAccount(await registry.getAddress());
    });

    it("unblock reverts UnknownPendingToken for non-pending token", async () => {
      const { gatekeeper, registry } = await loadFixture(deployFixture);
      const registrySigner = await impersonate(await registry.getAddress());
      await expect(
        gatekeeper.connect(registrySigner).unblock(99n)
      ).to.be.revertedWithCustomError(gatekeeper, "UnknownPendingToken");
      await stopImpersonatingAccount(await registry.getAddress());
    });

    it("unblock decrements counter and emits", async () => {
      const { gatekeeper, registry } = await loadFixture(deployFixture);
      const registrySigner = await impersonate(await registry.getAddress());
      await gatekeeper.connect(registrySigner).recordPendingUnlearning(7n);
      await expect(gatekeeper.connect(registrySigner).unblock(7n))
        .to.emit(gatekeeper, "UnlearningUnblocked")
        .withArgs(7n);
      expect(await gatekeeper.pendingUnlearningCount()).to.equal(0n);
      expect(await gatekeeper.pendingUnlearningSince(7n)).to.equal(0n);
      await stopImpersonatingAccount(await registry.getAddress());
    });
  });

  // -------------------------------------------------------------------
  // reportTimeout
  // -------------------------------------------------------------------

  describe("reportTimeout", () => {
    it("reverts UnknownPendingToken when token is not pending", async () => {
      const { gatekeeper, stranger } = await loadFixture(deployFixture);
      await expect(
        gatekeeper.connect(stranger).reportTimeout(123n)
      ).to.be.revertedWithCustomError(gatekeeper, "UnknownPendingToken");
    });

    it("reverts NotYetTimedOut before deltaMaxBlocks elapses", async () => {
      const { gatekeeper, registry, stranger, coordinator } = await loadFixture(
        deployFixture
      );
      // Tighten deltaMax for a more economical test (10 blocks).
      await gatekeeper.connect(coordinator).setDeltaMax(10n);
      const registrySigner = await impersonate(await registry.getAddress());
      await gatekeeper.connect(registrySigner).recordPendingUnlearning(11n);
      await stopImpersonatingAccount(await registry.getAddress());
      // Without mining new blocks, reportTimeout is premature.
      await expect(
        gatekeeper.connect(stranger).reportTimeout(11n)
      ).to.be.revertedWithCustomError(gatekeeper, "NotYetTimedOut");
    });

    it("emits ComplianceViolation after deltaMaxBlocks elapsed", async () => {
      const { gatekeeper, registry, stranger, coordinator } = await loadFixture(
        deployFixture
      );
      await gatekeeper.connect(coordinator).setDeltaMax(5n);
      const registrySigner = await impersonate(await registry.getAddress());
      const recordTx = await gatekeeper.connect(registrySigner).recordPendingUnlearning(11n);
      const recordReceipt = await recordTx.wait();
      const sinceBlock = BigInt(recordReceipt!.blockNumber);
      await stopImpersonatingAccount(await registry.getAddress());
      // Mine enough to exceed deltaMax.
      await mine(6);
      await expect(gatekeeper.connect(stranger).reportTimeout(11n))
        .to.emit(gatekeeper, "ComplianceViolation")
        .withArgs(11n, sinceBlock + 5n);
    });
  });

  // -------------------------------------------------------------------
  // setDeltaMax
  // -------------------------------------------------------------------

  describe("setDeltaMax", () => {
    it("only owner can call", async () => {
      const { gatekeeper, alice } = await loadFixture(deployFixture);
      await expect(
        gatekeeper.connect(alice).setDeltaMax(10n)
      ).to.be.revertedWithCustomError(gatekeeper, "OwnableUnauthorizedAccount");
    });

    it("rejects zero", async () => {
      const { gatekeeper, coordinator } = await loadFixture(deployFixture);
      await expect(
        gatekeeper.connect(coordinator).setDeltaMax(0n)
      ).to.be.revertedWithCustomError(gatekeeper, "InvalidDeltaMax");
    });

    it("updates and emits", async () => {
      const { gatekeeper, coordinator } = await loadFixture(deployFixture);
      await expect(gatekeeper.connect(coordinator).setDeltaMax(42n))
        .to.emit(gatekeeper, "DeltaMaxUpdated")
        .withArgs(42n);
      expect(await gatekeeper.deltaMaxBlocks()).to.equal(42n);
    });
  });
});
