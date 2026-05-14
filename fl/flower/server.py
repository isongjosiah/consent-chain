"""Federated round driver.

Sequential FedAvg: each round, every client fits locally on the current
global parameters, returns its update, the server aggregates by training
size, then evaluates centrally by averaging per-client accuracies
weighted by test-set size.

Resilience: per-round metrics are written to ``results/<dataset>/metrics.json``
*and* the aggregated state_dict is checkpointed to
``results/<dataset>/model_checkpoints/round_{N:03d}.pt`` after every round
by default. ``find_latest_checkpoint`` + the ``start_round``/``initial_params``
arguments support resuming from the last completed round if the process
is interrupted.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from fl.flower.client import CasasClient
from fl.flower.strategy import federated_average
from fl.unlearning.model_hash import hash_state_dict


@dataclass
class RoundMetrics:
    round_id: int
    wall_seconds: float
    aggregated_model_hash: str
    centralized_loss: float
    centralized_accuracy: float
    per_client: list[dict]


def _params_from_model(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.parameters()]


def _load_params_into_model(model: nn.Module, parameters: list[np.ndarray]) -> None:
    for p, new in zip(model.parameters(), parameters, strict=True):
        p.data = torch.from_numpy(new).to(p.device).to(p.dtype)


def _save_checkpoint(
    model: nn.Module, ckpt_dir: Path, round_id: int
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"round_{round_id:03d}.pt"
    torch.save(model.state_dict(), path)
    return path


_CKPT_RE = re.compile(r"^round_(\d+)\.pt$")


def find_latest_checkpoint(results_dir: Path) -> tuple[int, Path] | None:
    """Return ``(round_id, path)`` for the highest-numbered ``round_NNN.pt``."""
    ckpt_dir = Path(results_dir) / "model_checkpoints"
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


def _load_metrics_history(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        return []
    try:
        return json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return []


def run_federated_training(
    *,
    model_factory: Callable[[], nn.Module],
    clients: list[CasasClient],
    num_rounds: int,
    checkpoint_rounds: set[int] | None,
    results_dir: Path,
    log_every_round: bool = True,
    start_round: int = 1,
    initial_params: list[np.ndarray] | None = None,
) -> list[RoundMetrics]:
    """Run rounds ``[start_round, num_rounds]`` of FedAvg.

    ``checkpoint_rounds=None`` checkpoints **every** round (the default for
    long unattended runs). Pass a set to override.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    ckpt_dir = results_dir / "model_checkpoints"

    # The "server-side" reference model lives only to load aggregated
    # state for centralised evaluation and hashing; no training happens here.
    server_model = model_factory()
    if initial_params is not None:
        _load_params_into_model(server_model, initial_params)
        server_params = initial_params
    else:
        server_params = _params_from_model(server_model)

    history: list[RoundMetrics] = []
    # When resuming, retain prior per-round metrics so the new run appends to
    # rather than overwrites metrics.json.
    prior_metrics = _load_metrics_history(metrics_path) if start_round > 1 else []

    for r in range(start_round, num_rounds + 1):
        t0 = time.time()

        # Fit phase: each client trains locally on the current global params.
        updates: list[tuple[list[np.ndarray], int]] = []
        per_client_fit: list[dict] = []
        for c in clients:
            params, n, info = c.fit(server_params)
            updates.append((params, n))
            per_client_fit.append(info)

        # Aggregate.
        server_params = federated_average(updates)
        _load_params_into_model(server_model, server_params)
        model_hash = hash_state_dict(server_model.state_dict())

        # Evaluate phase: average per-client accuracy weighted by test-set size.
        per_client_eval: list[dict] = []
        eval_loss_w, eval_acc_w, eval_n = 0.0, 0.0, 0
        for c in clients:
            loss, n, info = c.evaluate(server_params)
            per_client_eval.append(info)
            eval_loss_w += loss * n
            eval_acc_w += info["accuracy"] * n
            eval_n += n
        central_loss = eval_loss_w / max(eval_n, 1)
        central_acc = eval_acc_w / max(eval_n, 1)

        wall = time.time() - t0
        rm = RoundMetrics(
            round_id=r,
            wall_seconds=wall,
            aggregated_model_hash=model_hash,
            centralized_loss=central_loss,
            centralized_accuracy=central_acc,
            per_client=[
                {**fit, **{k: v for k, v in ev.items() if k not in {"client_id", "subset"}}}
                for fit, ev in zip(per_client_fit, per_client_eval, strict=True)
            ],
        )
        history.append(rm)

        if log_every_round:
            print(
                f"  round {r:3d}/{num_rounds}  "
                f"loss={central_loss:.4f}  acc={central_acc:.4f}  "
                f"hash={model_hash[:12]}…  wall={wall:.1f}s"
            )

        # Persist after every round so a crash doesn't lose progress.
        combined = prior_metrics + [rm.__dict__ for rm in history]
        metrics_path.write_text(json.dumps(combined, indent=2) + "\n")

        should_checkpoint = (checkpoint_rounds is None) or (r in checkpoint_rounds)
        if should_checkpoint:
            ck = _save_checkpoint(server_model, ckpt_dir, r)
            if log_every_round:
                print(f"    ↳ checkpoint saved: {ck}")

    return history
