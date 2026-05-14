"""E1 — CASAS FedAvg baseline (CLAUDE.md §8).

Runs 5-client FedAvg over the per-apartment CASAS shards, saving:

  results/casas/metrics.json                  one entry per round
  results/casas/model_checkpoints/round_NNN.pt  at the configured checkpoint rounds
  results/casas/env.json                      reproducibility fingerprint

Usage:
  python -m fl.experiments.run_casas                       # full config (100 rounds)
  python -m fl.experiments.run_casas --rounds 3            # smoke test
  python -m fl.experiments.run_casas --device cpu          # force CPU
  python -m fl.experiments.run_casas --config fl/config/casas.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fl.datasets.federated_split import global_vocab_size, load_federation
from fl.experiments import _env
from fl.flower.client import CasasClient, TrainConfig
from fl.flower.server import find_latest_checkpoint, run_federated_training
from fl.models.bilstm_casas import build_from_yaml

LOGGER = logging.getLogger("run_casas")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str | None) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto: prefer CUDA → MPS → CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_clients(
    cfg: dict[str, Any],
    device: torch.device,
    vocab_size: int,
    subset_overrides: dict[str, Path] | None = None,
    boost_factor_by_subset: dict[str, float] | None = None,
    anomaly_weight_by_subset: dict[str, float] | None = None,
) -> list[CasasClient]:
    federation = load_federation(
        processed_dir=Path(cfg["dataset"]["processed_dir"]),
        subsets=cfg["dataset"]["subsets"],
        batch_size=int(cfg["training"]["batch_size"]),
        test_split=float(cfg["dataset"]["test_split"]),
        seed=int(cfg["random_seed"]),
        truncate_to_last_n=cfg["dataset"].get("truncate_to_last_n"),
        split_strategy=cfg["dataset"].get("split_strategy", "chronological"),
        subset_overrides=subset_overrides,
    )

    train_cfg = TrainConfig(
        local_epochs=int(cfg["training"]["local_epochs"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        grad_clip=float(cfg["training"]["grad_clip"]),
    )

    boost_map = boost_factor_by_subset or {}
    anomaly_map = anomaly_weight_by_subset or {}
    clients: list[CasasClient] = []
    for shard in federation:
        model = build_from_yaml(cfg["model"], vocab_size=vocab_size)
        clients.append(
            CasasClient(
                client_id=shard.client_id,
                subset=shard.subset,
                model=model,
                train_loader=shard.train_loader,
                test_loader=shard.test_loader,
                train_cfg=train_cfg,
                device=device,
                boost_factor=boost_map.get(shard.subset, 1.0),
                anomaly_weight=anomaly_map.get(shard.subset, 0.0),
            )
        )
    return clients


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E1 — CASAS FedAvg baseline.")
    p.add_argument("--config", type=Path, default=Path("fl/config/casas.yaml"))
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override the round count (defaults to config.federation.rounds).",
    )
    p.add_argument(
        "--checkpoint",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Explicit checkpoint rounds. If omitted, checkpoints every round "
            "(safe for unattended runs; crash-resumable via --resume)."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest results/<dataset>/model_checkpoints/round_*.pt.",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
    )
    p.add_argument(
        "--use-canary",
        action="store_true",
        help=(
            "Swap the canary.target_subset client's shard for the poisoned "
            "version produced by `python -m fl.datasets.canary_injection "
            "--inject`. Routes outputs to results/casas_poisoned/."
        ),
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["random_seed"])
    seed_everything(seed)

    device = resolve_device(args.device if args.device != "auto" else None)
    LOGGER.info("device: %s", device)

    rounds = args.rounds or int(cfg["federation"]["rounds"])
    revoke_round = int(cfg["federation"]["revocation"]["revoke_at_round"])
    # None = checkpoint every round (the new default — see server.py).
    checkpoint_rounds: set[int] | None = (
        set(args.checkpoint) if args.checkpoint else None
    )
    if checkpoint_rounds is not None:
        # Always preserve the revocation-round + final-round milestones the
        # paper depends on, even if the user passed a custom list.
        checkpoint_rounds |= {revoke_round, rounds}
    LOGGER.info(
        "rounds=%d  checkpoint_rounds=%s",
        rounds,
        "EVERY_ROUND" if checkpoint_rounds is None else sorted(checkpoint_rounds),
    )

    processed_dir = Path(cfg["dataset"]["processed_dir"])
    subsets = list(cfg["dataset"]["subsets"])
    vocab_size = global_vocab_size(processed_dir, subsets)
    LOGGER.info("vocab_size=%d (max obs id + 1)", vocab_size)

    subset_overrides: dict[str, Path] | None = None
    boost_map: dict[str, float] = {}
    anomaly_map: dict[str, float] = {}
    results_dir = Path(cfg["logging"]["results_dir"])
    if args.use_canary:
        canary_cfg = cfg.get("canary", {})
        if not canary_cfg.get("enabled", False):
            raise SystemExit("--use-canary set but canary.enabled is false in config")
        target_id = int(canary_cfg["target_client_id"])
        target_subset = cfg["federation"]["client_map"][target_id]
        poisoned_dir = Path(canary_cfg["poisoned_dir"])
        if not (poisoned_dir / target_subset / "data.npz").exists():
            raise SystemExit(
                f"missing {poisoned_dir / target_subset / 'data.npz'} — "
                f"run `python -m fl.datasets.canary_injection --inject` first."
            )
        subset_overrides = {target_subset: poisoned_dir}
        # Model-replacement boost (Bagdasaryan 2020): scale the attacker's
        # update by n_total / n_attacker so the post-FedAvg global model
        # approximates the attacker's local model. Without this, the
        # canary lesson is washed out by the 4 clean clients each round
        # (verified empirically: 0 % trigger rate after 10 unscaled rounds).
        boost = float(canary_cfg.get("boost_factor", 0.0))
        if boost <= 0.0:
            # Auto: compute from federation training set sizes.
            from fl.datasets.federated_split import load_federation as _lf
            shards = _lf(
                processed_dir=Path(cfg["dataset"]["processed_dir"]),
                subsets=cfg["dataset"]["subsets"],
                batch_size=int(cfg["training"]["batch_size"]),
                test_split=float(cfg["dataset"]["test_split"]),
                seed=int(cfg["random_seed"]),
                truncate_to_last_n=cfg["dataset"].get("truncate_to_last_n"),
                split_strategy=cfg["dataset"].get("split_strategy", "chronological"),
            )
            n_total = sum(s.train_size for s in shards)
            n_attacker = next(s.train_size for s in shards if s.subset == target_subset)
            boost = n_total / n_attacker
        boost_map = {target_subset: boost}
        anomaly_w = float(canary_cfg.get("anomaly_weight", 0.0))
        if anomaly_w > 0.0:
            anomaly_map = {target_subset: anomaly_w}
        # Redirect outputs so the clean-baseline artefacts don't get clobbered.
        results_dir = Path(str(results_dir) + "_poisoned")
        LOGGER.info(
            "canary mode: client %d (%s) → %s; boost=%.3f; anomaly_weight=%g; results → %s",
            target_id, target_subset, poisoned_dir, boost, anomaly_w, results_dir,
        )

    clients = build_clients(
        cfg, device=device, vocab_size=vocab_size,
        subset_overrides=subset_overrides,
        boost_factor_by_subset=boost_map,
        anomaly_weight_by_subset=anomaly_map,
    )
    LOGGER.info(
        "clients ready: %s",
        [(c.client_id, c.subset, len(c.train_loader.dataset), len(c.test_loader.dataset)) for c in clients],
    )

    # --- Resume logic ---------------------------------------------------
    start_round = 1
    initial_params: list[Any] | None = None
    if args.resume:
        latest = find_latest_checkpoint(results_dir)
        if latest is None:
            LOGGER.warning("--resume requested but no checkpoint found in %s", results_dir)
        else:
            r, ckpt_path = latest
            LOGGER.info("resuming from %s (round %d)", ckpt_path, r)
            tmp_model = build_from_yaml(cfg["model"], vocab_size=vocab_size).to(device)
            tmp_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            initial_params = [p.detach().cpu().numpy() for p in tmp_model.parameters()]
            start_round = r + 1
            if start_round > rounds:
                LOGGER.info("already at round %d ≥ rounds=%d; nothing to do", r, rounds)
                return 0

    # --- Fingerprint ----------------------------------------------------
    fingerprint = _env.fingerprint(seed)
    fingerprint["dataset"] = "casas"
    fingerprint["rounds"] = rounds
    fingerprint["start_round"] = start_round
    fingerprint["vocab_size"] = vocab_size
    fingerprint["device"] = str(device)
    fingerprint["truncate_to_last_n"] = cfg["dataset"].get("truncate_to_last_n")
    fingerprint["split_strategy"] = cfg["dataset"].get("split_strategy", "chronological")
    fingerprint["use_canary"] = args.use_canary
    if subset_overrides is not None:
        fingerprint["subset_overrides"] = {k: str(v) for k, v in subset_overrides.items()}
    if boost_map:
        fingerprint["boost_factor_by_subset"] = boost_map
    if anomaly_map:
        fingerprint["anomaly_weight_by_subset"] = anomaly_map
    fingerprint["clients"] = [
        {"client_id": c.client_id, "subset": c.subset,
         "train_size": len(c.train_loader.dataset),
         "test_size": len(c.test_loader.dataset)}
        for c in clients
    ]
    _env.write(results_dir, fingerprint)

    def model_factory():
        return build_from_yaml(cfg["model"], vocab_size=vocab_size).to(device)

    try:
        history = run_federated_training(
            model_factory=model_factory,
            clients=clients,
            num_rounds=rounds,
            checkpoint_rounds=checkpoint_rounds,
            results_dir=results_dir,
            log_every_round=bool(cfg["logging"].get("log_every_round", True)),
            start_round=start_round,
            initial_params=initial_params,
        )
    finally:
        _env.stamp_end(results_dir)

    if history:
        last = history[-1]
        LOGGER.info(
            "done. final round %d: acc=%.4f loss=%.4f model_hash=%s",
            last.round_id, last.centralized_accuracy,
            last.centralized_loss, last.aggregated_model_hash,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
