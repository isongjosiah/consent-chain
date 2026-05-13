/**
 * Integration.test.ts — full §7 lifecycle with real contracts (no mocks).
 *
 * Exercises the architectural seam: ConsentRegistry callbacks into
 * FLGatekeeper on revoke (recordPendingUnlearning) and on UA acceptance
 * (unblock, fired from inside transitionToUnlearned). The pending-unlearning
 * counter must reach 1 after revoke and 0 after the UA is accepted, with
 * validateRound blocked in between — this is the §III Property 2
 * (Unlearning Enforceability / liveness) contract.
 */
import { expect } from "chai";
import { ethers } from "hardhat";
import { loadFixture } from "@nomicfoundation/hardhat-network-helpers";
import type { HardhatEthersSigner } from "@nomicfoundation/hardhat-ethers/signers";

import type {
  ConsentRegistry,
  FLGatekeeper,
  UnlearningOracle,
} from "../typechain-types";

const State = {
  INACTIVE: 0,
  ACTIVE: 1,
  SUSPENDED: 2,
  REVOKED: 3,
  EXPIRED: 4,
  UNLEARNED: 5,
} as const;

const Method = { SISA_FED: 0, GRADIENT_ASCENT: 1, FEDERASER: 2 } as const;
const ZERO_HASH = ethers.ZeroHash;

const PURPOSES: string[] = ["adl-recognition", "anomaly-detection", "research"]
  .map((p) => ethers.keccak256(ethers.toUtf8Bytes(p)))
  .sort();
const ROUND_PURPOSE = PURPOSES[0];

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

function computeUaHash(chainId: bigint, u: UaInput): string {
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

async function buildUa(u: UaInput) {
  const chainId = BigInt((await ethers.provider.getNetwork()).chainId);
  return { ...u, uaHash: computeUaHash(chainId, u) };
}

async function signUa(verifiers: HardhatEthersSigner[], uaHash: string) {
  const bytes = ethers.getBytes(uaHash);
  return Promise.all(verifiers.map((v) => v.signMessage(bytes)));
}

function ctx(roundId: bigint, expectedHash: string = ZERO_HASH) {
  return { roundId, roundPurpose: ROUND_PURPOSE, expectedModelHashPre: expectedHash };
}

describe("Integration — full §7 lifecycle", () => {
  async function fullStackFixture() {
    const [owner, coordinator, alice, bob, v1, v2, v3, anyone] = await ethers.getSigners();

    const Registry = await ethers.getContractFactory("ConsentRegistry");
    const registry = (await Registry.deploy(owner.address)) as unknown as ConsentRegistry;

    const Oracle = await ethers.getContractFactory("UnlearningOracle");
    const oracle = (await Oracle.deploy(
      await registry.getAddress(),
      owner.address
    )) as unknown as UnlearningOracle;

    const Gatekeeper = await ethers.getContractFactory("FLGatekeeper");
    const gatekeeper = (await Gatekeeper.deploy(
      await registry.getAddress(),
      coordinator.address
    )) as unknown as FLGatekeeper;

    // Wire the registry's two callback addresses.
    await registry.connect(owner).setOracle(await oracle.getAddress());
    await registry.connect(owner).setGatekeeper(await gatekeeper.getAddress());

    // Verifier Committee: 3-of-3 for this test (matches DEFAULT_VERIFIER_THRESHOLD).
    for (const v of [v1, v2, v3]) {
      await oracle.connect(owner).addVerifier(v.address);
    }

    return {
      owner, coordinator, alice, bob, anyone,
      verifiers: [v1, v2, v3] as HardhatEthersSigner[],
      registry, oracle, gatekeeper,
    };
  }

  it("runs the full grant → train → revoke → UA → resume lifecycle", async () => {
    const f = await loadFixture(fullStackFixture);
    const aliceId = 1n;
    const bobId = 2n;

    // 1. Both DSes mint consent NFTs.
    await f.registry.connect(f.alice).grantConsent(PURPOSES, 0n);
    await f.registry.connect(f.bob).grantConsent(PURPOSES, 0n);
    expect(await f.registry.stateOf(aliceId)).to.equal(State.ACTIVE);
    expect(await f.registry.stateOf(bobId)).to.equal(State.ACTIVE);

    // 2. Coordinator commits an initial model hash (genesis).
    const H0 = ethers.id("model-genesis");
    await f.gatekeeper.connect(f.coordinator).updateModelHash(0n, H0);
    expect(await f.gatekeeper.currentModelHash()).to.equal(H0);

    // 3. Validate round 1 with both participants — passes; advances currentRoundId.
    await expect(
      f.gatekeeper
        .connect(f.coordinator)
        .validateRound(
          ctx(1n, H0),
          [aliceId, bobId],
          [PURPOSES, PURPOSES],
          [0n, 0n]
        )
    )
      .to.emit(f.gatekeeper, "RoundValidated")
      .withArgs(1n, 2n);

    // 4. Commit the round-1 model hash.
    const H1 = ethers.id("model-round-1");
    await f.gatekeeper.connect(f.coordinator).updateModelHash(1n, H1);

    // 5. Validate round 2; commit round-2 hash. Now the "current model"
    //    against which a future revocation will bind is H2.
    await f.gatekeeper
      .connect(f.coordinator)
      .validateRound(ctx(2n, H1), [aliceId, bobId], [PURPOSES, PURPOSES], [0n, 0n]);
    const H2 = ethers.id("model-round-2");
    await f.gatekeeper.connect(f.coordinator).updateModelHash(2n, H2);

    // 6. Alice revokes mid-flight. Registry stores modelHashAtRevocation=H2
    //    and fires gatekeeper.recordPendingUnlearning.
    await expect(f.registry.connect(f.alice).revokeConsent(aliceId))
      .to.emit(f.registry, "ConsentRevoked")
      .withArgs(aliceId, H2)
      .and.to.emit(f.registry, "UnlearningRequired")
      .withArgs(aliceId, H2)
      .and.to.emit(f.gatekeeper, "UnlearningPendingRecorded");

    expect(await f.registry.stateOf(aliceId)).to.equal(State.REVOKED);
    expect(await f.registry.modelHashAtRevocation(aliceId)).to.equal(H2);
    expect(await f.gatekeeper.pendingUnlearningCount()).to.equal(1n);

    // 7. Liveness lever in action: the next round CANNOT proceed while
    //    pendingUnlearningCount > 0, even if we'd excluded Alice from the
    //    participant set.
    await expect(
      f.gatekeeper
        .connect(f.coordinator)
        .validateRound(ctx(3n, H2), [bobId], [PURPOSES], [0n])
    )
      .to.be.revertedWithCustomError(f.gatekeeper, "PendingUnlearningBlocksRound")
      .withArgs(1n);

    // 8. Off-chain: data controller runs SISA-Fed; VC re-runs the three
    //    MIAs and confirms all < τ; the UA is assembled and signed.
    const ua = await buildUa({
      dsTokenId: aliceId,
      modelHashPre: H2,
      modelHashPost: ethers.id("model-round-2-post-unlearn"),
      method: Method.SISA_FED,
      shadowMiaScore: 5100n,    // 0.51
      liraScore: 4900n,         // 0.49
      labelOnlyMiaScore: 5000n, // 0.50
      timestamp: BigInt(Math.floor(Date.now() / 1000)),
    });
    const sigs = await signUa(f.verifiers, ua.uaHash);

    // 9. Submit the UA. One transaction:
    //      registry.transitionToUnlearned → fires UnlearningCompleted
    //                                      → fires gatekeeper.unblock
    //                                      → fires UnlearningUnblocked on gatekeeper
    //      oracle emits AttestationAccepted
    await expect(f.oracle.connect(f.anyone).submitAttestation(ua, sigs))
      .to.emit(f.oracle, "AttestationAccepted")
      .withArgs(
        aliceId, ua.uaHash, ua.method,
        ua.shadowMiaScore, ua.liraScore, ua.labelOnlyMiaScore
      )
      .and.to.emit(f.registry, "UnlearningCompleted")
      .withArgs(aliceId, ua.uaHash)
      .and.to.emit(f.gatekeeper, "UnlearningUnblocked")
      .withArgs(aliceId);

    expect(await f.registry.stateOf(aliceId)).to.equal(State.UNLEARNED);
    expect(await f.gatekeeper.pendingUnlearningCount()).to.equal(0n);
    expect(await f.gatekeeper.pendingUnlearningSince(aliceId)).to.equal(0n);

    // 10. Round 3 can now proceed with Bob's token only — Alice's is
    //     UNLEARNED and would revert ParticipantIneligible if included.
    await expect(
      f.gatekeeper
        .connect(f.coordinator)
        .validateRound(ctx(3n, H2), [bobId], [PURPOSES], [0n])
    )
      .to.emit(f.gatekeeper, "RoundValidated")
      .withArgs(3n, 1n);
    expect(await f.gatekeeper.currentRoundId()).to.equal(3n);

    // 11. Including Alice's UNLEARNED token still reverts.
    await expect(
      f.gatekeeper
        .connect(f.coordinator)
        .validateRound(
          { roundId: 4n, roundPurpose: ROUND_PURPOSE, expectedModelHashPre: ZERO_HASH },
          [aliceId, bobId],
          [PURPOSES, PURPOSES],
          [0n, 0n]
        )
    )
      .to.be.revertedWithCustomError(f.gatekeeper, "ParticipantIneligible")
      .withArgs(aliceId, State.UNLEARNED);
  });

  it("models the EXPIRED path: ttl-driven revocation flows through the same gates", async () => {
    const f = await loadFixture(fullStackFixture);
    const aliceId = 1n;

    // Short TTL so we can fast-forward.
    const block = await ethers.provider.getBlock("latest");
    const ttl = BigInt(block!.timestamp + 60);
    await f.registry.connect(f.alice).grantConsent(PURPOSES, ttl);
    const H0 = ethers.id("model-genesis-exp");
    await f.gatekeeper.connect(f.coordinator).updateModelHash(0n, H0);

    // Time-warp past ttl, then anyone (the coordinator here) flips it.
    await ethers.provider.send("evm_setNextBlockTimestamp", [Number(ttl) + 1]);
    await ethers.provider.send("evm_mine", []);
    await expect(f.registry.connect(f.anyone).checkExpiry(aliceId))
      .to.emit(f.registry, "ConsentExpired")
      .and.to.emit(f.gatekeeper, "UnlearningPendingRecorded");

    expect(await f.gatekeeper.pendingUnlearningCount()).to.equal(1n);

    // Submit a UA. The path is identical to revoke — the oracle accepts
    // REVOKED or EXPIRED.
    const ua = await buildUa({
      dsTokenId: aliceId,
      modelHashPre: H0, // registry captured currentModelHash=H0 at expiry
      modelHashPost: ethers.id("model-genesis-exp-post"),
      method: Method.GRADIENT_ASCENT,
      shadowMiaScore: 5000n,
      liraScore: 5000n,
      labelOnlyMiaScore: 5000n,
      timestamp: BigInt(Math.floor(Date.now() / 1000)),
    });
    const sigs = await signUa(f.verifiers, ua.uaHash);

    await f.oracle.connect(f.anyone).submitAttestation(ua, sigs);

    expect(await f.registry.stateOf(aliceId)).to.equal(State.UNLEARNED);
    expect(await f.gatekeeper.pendingUnlearningCount()).to.equal(0n);
  });
});
