"""E5 — CASAS utility recovery after revocation (CLAUDE.md §8, §10 D12).

Resume FedAvg with the 4 remaining clients (twor_2010 revoked) starting
from a post-unlearning checkpoint. Measures how fast accuracy bounces
back. Feeds Figure 2a in the paper.

Layout:

  results/casas_recovery/
      env.json                       reproducibility fingerprint
      metrics.json                   one entry per recovery round
      model_checkpoints/round_*.pt   per-round aggregated checkpoints

The default start checkpoint is ``results/casas_unlearned/sisa_fed/
post_unlearn_model.pt`` — the only method whose three-MIA scores all sit
below τ. Override with ``--start-checkpoint``.

Usage::

    python -m fl.experiments.run_utility_recovery --device cuda
    python -m fl.experiments.run_utility_recovery --rounds 10 \\
        --start-checkpoint results/casas_unlearned/federaser/post_unlearn_model.pt \\
        --results-dir results/casas_recovery_federaser
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
from fl.experiments.run_casas import resolve_device
from fl.flower.client import CasasClient, TrainConfig
from fl.flower.server import run_federated_training
from fl.models.bilstm_casas import build_from_yaml

LOGGER = logging.getLogger("run_utility_recovery")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_remaining_clients(
    cfg: dict[str, Any],
    *,
    device: torch.device,
    vocab_size: int,
    exclude_client_ids: set[int],
) -> list[CasasClient]:
    federation = load_federation(
        processed_dir=Path(cfg["dataset"]["processed_dir"]),
        subsets=cfg["dataset"]["subsets"],
        batch_size=int(cfg["training"]["batch_size"]),
        test_split=float(cfg["dataset"]["test_split"]),
        seed=int(cfg["random_seed"]),
        truncate_to_last_n=cfg["dataset"].get("truncate_to_last_n"),
        split_strategy=cfg["dataset"].get("split_strategy", "chronological"),
    )
    train_cfg = TrainConfig(
        local_epochs=int(cfg["training"]["local_epochs"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        grad_clip=float(cfg["training"]["grad_clip"]),
    )
    clients: list[CasasClient] = []
    for shard in federation:
        if shard.client_id in exclude_client_ids:
            continue
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
            )
        )
    return clients


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E5 — CASAS utility recovery after revocation.")
    p.add_argument("--config", type=Path, default=Path("fl/config/casas.yaml"))
    p.add_argument("--rounds", type=int, default=10,
                   help="Number of recovery rounds to run (default 10 → rounds 61–70). "
                   "Use 100 with --from-scratch for the naive-retrain gold standard.")
    p.add_argument("--start-checkpoint", type=Path,
                   default=Path("results/casas_unlearned/sisa_fed/post_unlearn_model.pt"),
                   help="Post-unlearning checkpoint to seed recovery from "
                   "(ignored when --from-scratch is set).")
    p.add_argument("--from-scratch", action="store_true",
                   help="Initialise a fresh model (no checkpoint load) — the "
                   "naive-retrain-from-scratch baseline for Table 1.")
    p.add_argument("--exclude-clients", type=int, nargs="+", default=None,
                   help="Federation client ids to drop (default: revocation.target_client_id).")
    p.add_argument("--results-dir", type=Path, default=Path("results/casas_recovery"))
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg["random_seed"])
    _seed_everything(seed)
    device = resolve_device(args.device if args.device != "auto" else None)

    excluded = (
        set(args.exclude_clients) if args.exclude_clients is not None
        else {int(cfg["federation"]["revocation"]["target_client_id"])}
    )

    vocab_size = global_vocab_size(
        Path(cfg["dataset"]["processed_dir"]), cfg["dataset"]["subsets"]
    )
    LOGGER.info("device=%s  vocab_size=%d  excluded_clients=%s",
                device, vocab_size, sorted(excluded))

    if args.from_scratch:
        initial_params = None
        LOGGER.info("naive-retrain mode: fresh model init, no seed checkpoint")
    else:
        if not args.start_checkpoint.exists():
            raise SystemExit(f"missing start checkpoint: {args.start_checkpoint}")
        seed_model = build_from_yaml(cfg["model"], vocab_size=vocab_size).to(device)
        seed_model.load_state_dict(torch.load(args.start_checkpoint, map_location=device))
        initial_params = [p.detach().cpu().numpy() for p in seed_model.parameters()]
        LOGGER.info("seed checkpoint: %s", args.start_checkpoint)

    clients = _build_remaining_clients(
        cfg, device=device, vocab_size=vocab_size, exclude_client_ids=excluded,
    )
    LOGGER.info(
        "remaining clients: %s",
        [(c.client_id, c.subset, len(c.train_loader.dataset), len(c.test_loader.dataset))
         for c in clients],
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _env.fingerprint(seed)
    fingerprint.update({
        "dataset": "casas",
        "stage": "naive_retrain" if args.from_scratch else "utility_recovery",
        "from_scratch": bool(args.from_scratch),
        "start_checkpoint": None if args.from_scratch else str(args.start_checkpoint),
        "excluded_clients": sorted(excluded),
        "rounds": int(args.rounds),
        "device": str(device),
        "vocab_size": vocab_size,
        "remaining_clients": [
            {"client_id": c.client_id, "subset": c.subset,
             "train_size": len(c.train_loader.dataset),
             "test_size": len(c.test_loader.dataset)}
            for c in clients
        ],
    })
    _env.write(args.results_dir, fingerprint)

    def model_factory():
        return build_from_yaml(cfg["model"], vocab_size=vocab_size).to(device)

    try:
        history = run_federated_training(
            model_factory=model_factory,
            clients=clients,
            num_rounds=int(args.rounds),
            checkpoint_rounds=None,  # checkpoint every round
            results_dir=args.results_dir,
            log_every_round=True,
            start_round=1,
            initial_params=initial_params,
        )
    finally:
        _env.stamp_end(args.results_dir)

    if history:
        first, last = history[0], history[-1]
        LOGGER.info(
            "recovery: round 1 acc=%.4f  →  round %d acc=%.4f  (Δ=%+.4f)",
            first.centralized_accuracy, last.round_id,
            last.centralized_accuracy,
            last.centralized_accuracy - first.centralized_accuracy,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
