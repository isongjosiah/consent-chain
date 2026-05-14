"""E6 — N-BaIoT federated autoencoder baseline.

Per-device benign-traffic autoencoder federation, 50 rounds default.
Reconstruction error on each client's test set + held-out attack
samples drives both the §V anomaly-detection table and the per-client
"did the model see this client?" signature that replaces the canary.

Output layout::

    results/nbaiot/
        env.json                              reproducibility fingerprint
        metrics.json                          one entry per round
        model_checkpoints/round_NNN.pt        every round
        client_recon_history.json             test/attack recon per round

Usage::

    python -m fl.experiments.run_nbaiot
    python -m fl.experiments.run_nbaiot --rounds 3       # smoke
    python -m fl.experiments.run_nbaiot --device cpu     # force CPU
    python -m fl.experiments.run_nbaiot --resume         # resume latest
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fl.datasets.federated_split_nbaiot import load_federation_nbaiot
from fl.experiments import _env
from fl.flower.autoencoder_client import AETrainConfig, NBaIoTClient
from fl.flower.strategy import federated_average
from fl.models.autoencoder_nbaiot import build_from_yaml
from fl.unlearning.model_hash import hash_state_dict

LOGGER = logging.getLogger("run_nbaiot")
_CKPT_RE = re.compile(r"^round_(\d+)\.pt$")


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
    # auto: cuda > mps > cpu. MPS is safe for the autoencoder (no LSTM).
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_latest_checkpoint(results_dir: Path) -> tuple[int, Path] | None:
    ckpt_dir = results_dir / "model_checkpoints"
    if not ckpt_dir.exists():
        return None
    best: tuple[int, Path] | None = None
    for p in ckpt_dir.iterdir():
        m = _CKPT_RE.match(p.name)
        if not m:
            continue
        r = int(m.group(1))
        if best is None or r > best[0]:
            best = (r, p)
    return best


def _params_from_model(model: torch.nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def _load_params_into_model(model: torch.nn.Module, parameters: list[np.ndarray]) -> None:
    for p, new in zip(model.parameters(), parameters, strict=True):
        p.data = torch.from_numpy(new.copy()).to(p.device).to(p.dtype)


def build_clients(
    cfg: dict[str, Any],
    device: torch.device,
) -> list[NBaIoTClient]:
    shards = load_federation_nbaiot(
        raw_root=Path(cfg["dataset"]["raw_dir"]),
        per_device_benign_cap=int(cfg["dataset"].get("per_device_benign_cap", 5_000)),
        per_device_attack_cap=int(cfg["dataset"].get("per_device_attack_cap", 5_000)),
        batch_size=int(cfg["training"]["batch_size"]),
        test_split=float(cfg["dataset"]["test_split"]),
        seed=int(cfg["random_seed"]),
    )
    train_cfg = AETrainConfig(
        local_epochs=int(cfg["training"]["local_epochs"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        grad_clip=float(cfg["training"]["grad_clip"]),
    )
    clients: list[NBaIoTClient] = []
    for shard in shards:
        model = build_from_yaml(cfg["model"])
        clients.append(
            NBaIoTClient(
                client_id=shard.client_id,
                device_name=shard.device,
                model=model,
                train_loader=shard.train_loader,
                test_loader=shard.test_loader,
                attack_loader=shard.attack_loader,
                train_cfg=train_cfg,
                device=device,
            )
        )
    return clients


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E6 — N-BaIoT federated AE baseline.")
    p.add_argument("--config", type=Path, default=Path("fl/config/nbaiot.yaml"))
    p.add_argument("--rounds", type=int, default=None,
                   help="Override config federation.rounds.")
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint in results/nbaiot/.")
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
    LOGGER.info("rounds=%d  revoke_at_round=%d", rounds, revoke_round)

    clients = build_clients(cfg, device=device)
    LOGGER.info(
        "clients: %s",
        [(c.client_id, c.device_name,
          len(c.train_loader.dataset), len(c.test_loader.dataset),
          (c.attack_loader and len(c.attack_loader.dataset)) or 0)
         for c in clients],
    )

    results_dir = Path(cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    ckpt_dir = results_dir / "model_checkpoints"

    # Resume logic
    start_round = 1
    server_model = build_from_yaml(cfg["model"]).to(device)
    if args.resume:
        latest = find_latest_checkpoint(results_dir)
        if latest is None:
            LOGGER.warning("--resume requested but no checkpoint found")
        else:
            r, path = latest
            LOGGER.info("resuming from %s (round %d)", path, r)
            server_model.load_state_dict(torch.load(path, map_location=device))
            start_round = r + 1
            if start_round > rounds:
                LOGGER.info("already past target rounds; nothing to do")
                return 0
    server_params = _params_from_model(server_model)

    fingerprint = _env.fingerprint(seed)
    fingerprint.update({
        "dataset": "nbaiot",
        "rounds": rounds,
        "start_round": start_round,
        "device": str(device),
        "per_device_benign_cap": int(cfg["dataset"].get("per_device_benign_cap", 5_000)),
        "per_device_attack_cap": int(cfg["dataset"].get("per_device_attack_cap", 5_000)),
        "clients": [
            {"client_id": c.client_id, "device": c.device_name,
             "train_size": len(c.train_loader.dataset),
             "test_size": len(c.test_loader.dataset),
             "attack_size": (c.attack_loader and len(c.attack_loader.dataset)) or 0}
            for c in clients
        ],
    })
    _env.write(results_dir, fingerprint)

    # Load prior metrics if resuming so we append rather than overwrite.
    prior_metrics: list[dict] = []
    if start_round > 1 and metrics_path.exists():
        try:
            prior_metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            prior_metrics = []

    history: list[dict] = []
    log_every = bool(cfg["logging"].get("log_every_round", True))

    try:
        for r in range(start_round, rounds + 1):
            t0 = time.time()
            updates: list[tuple[list[np.ndarray], int]] = []
            per_client_fit: list[dict] = []
            for c in clients:
                params, n, info = c.fit(server_params)
                updates.append((params, n))
                per_client_fit.append(info)
            server_params = federated_average(updates)
            _load_params_into_model(server_model, server_params)
            model_hash = hash_state_dict(server_model.state_dict())

            per_client_eval: list[dict] = []
            test_loss_w, test_n = 0.0, 0
            attack_losses: list[float] = []
            for c in clients:
                loss, n, info = c.evaluate(server_params)
                per_client_eval.append(info)
                test_loss_w += loss * n
                test_n += n
                if info.get("attack_recon_loss") is not None:
                    attack_losses.append(float(info["attack_recon_loss"]))
            central_test_recon = test_loss_w / max(test_n, 1)
            central_attack_recon = sum(attack_losses) / max(len(attack_losses), 1) if attack_losses else None

            rm = {
                "round_id": r,
                "wall_seconds": time.time() - t0,
                "aggregated_model_hash": model_hash,
                "centralized_test_recon": central_test_recon,
                "centralized_attack_recon": central_attack_recon,
                "recon_gap": (central_attack_recon - central_test_recon)
                             if central_attack_recon is not None else None,
                "per_client": [
                    {**fit, **{k: v for k, v in ev.items()
                               if k not in {"client_id", "device"}}}
                    for fit, ev in zip(per_client_fit, per_client_eval, strict=True)
                ],
            }
            history.append(rm)

            if log_every:
                gap_str = f"  gap={rm['recon_gap']:.4f}" if rm["recon_gap"] is not None else ""
                print(
                    f"  round {r:3d}/{rounds}  "
                    f"test_recon={central_test_recon:.4f}{gap_str}  "
                    f"hash={model_hash[:12]}…  wall={rm['wall_seconds']:.1f}s"
                )

            metrics_path.write_text(
                json.dumps(prior_metrics + history, indent=2) + "\n"
            )
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(server_model.state_dict(), ckpt_dir / f"round_{r:03d}.pt")
    finally:
        _env.stamp_end(results_dir)

    if history:
        last = history[-1]
        LOGGER.info(
            "done. final round %d: test_recon=%.4f attack_recon=%s gap=%s",
            last["round_id"], last["centralized_test_recon"],
            f"{last['centralized_attack_recon']:.4f}" if last["centralized_attack_recon"] is not None else "n/a",
            f"{last['recon_gap']:.4f}" if last["recon_gap"] is not None else "n/a",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
