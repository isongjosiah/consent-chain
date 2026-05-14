"""E3 — CASAS unlearning (CLAUDE.md §8, §10 D10).

Given a pre-revocation checkpoint (typically round_060.pt), apply one
of the three method-agnostic unlearners to remove the revoked DS's
contribution, save the post-unlearning model + UnlearningResult
artefact, and emit a manifest that the D11 MIA suite + D15
``generate_ua.py`` will consume.

Output layout::

    results/casas_unlearned/<method>/
        env.json                    fingerprint + method config
        unlearning_result.json      method, hashes, wall, details
        post_unlearn_model.pt       state_dict (gitignored)

Usage::

    # Default config — runs all three methods over revocation-round
    # checkpoint, revoking client 3 (twor_2010).
    python -m fl.experiments.run_unlearning

    # Single method, custom checkpoint, custom revoked client.
    python -m fl.experiments.run_unlearning --method gradient_ascent \\
        --checkpoint results/casas/model_checkpoints/round_060.pt \\
        --revoked-client 3
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from fl.datasets.federated_split import global_vocab_size, load_federation
from fl.experiments import _env
from fl.flower.client import CasasClient, TrainConfig
from fl.models.bilstm_casas import build_from_yaml
from fl.unlearning.base import UnlearningRequest, UnlearningResult
from fl.unlearning.federaser import FedEraserUnlearner
from fl.unlearning.gradient_ascent import GradientAscentUnlearner
from fl.unlearning.sisa_fed import SisaFedUnlearner

LOGGER = logging.getLogger("run_unlearning")

UNLEARNERS = {
    "gradient_ascent": GradientAscentUnlearner,
    "sisa_fed": SisaFedUnlearner,
    "federaser": FedEraserUnlearner,
}


def _resolve_device(preferred: str | None) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_pre_unlearn_model(
    cfg: dict[str, Any],
    vocab_size: int,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = build_from_yaml(cfg["model"], vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def _serialise_result(result: UnlearningResult) -> dict[str, Any]:
    """JSON-safe view of UnlearningResult (drops the model)."""
    return {
        "method": result.method,
        "method_id": result.method_id,
        "model_hash_pre": result.model_hash_pre,
        "model_hash_post": result.model_hash_post,
        "wall_seconds": result.wall_seconds,
        "details": result.details,
    }


def _run_one_method(
    method_name: str,
    cfg: dict[str, Any],
    *,
    vocab_size: int,
    device: torch.device,
    checkpoint: Path,
    target_client_id: int,
    target_subset: str,
    output_root: Path,
) -> UnlearningResult:
    LOGGER.info("=== %s ===", method_name)
    out_dir = output_root / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fresh model loaded from the pre-revocation checkpoint.
    pre_model = _load_pre_unlearn_model(cfg, vocab_size, checkpoint, device)

    # Federation loaders — we need every client's training loader so the
    # unlearner has access to the revoked DS's data (for gradient ascent /
    # FedEraser's calibration) AND the remaining clients' data
    # (for SISA-Fed's retrain-from-scratch path).
    federation = load_federation(
        processed_dir=Path(cfg["dataset"]["processed_dir"]),
        subsets=cfg["dataset"]["subsets"],
        batch_size=int(cfg["training"]["batch_size"]),
        test_split=float(cfg["dataset"]["test_split"]),
        seed=int(cfg["random_seed"]),
        truncate_to_last_n=cfg["dataset"].get("truncate_to_last_n"),
        split_strategy=cfg["dataset"].get("split_strategy", "chronological"),
    )

    revoked_loader = next(
        s.train_loader for s in federation if s.client_id == target_client_id
    )
    remaining_loaders = {
        s.client_id: s.train_loader
        for s in federation
        if s.client_id != target_client_id
    }

    method_config = (
        cfg.get("unlearning", {}).get(method_name, {})
    )

    request = UnlearningRequest(
        target_client_id=target_client_id,
        revoked_train_loader=revoked_loader,
        remaining_loaders=remaining_loaders,
        pre_unlearn_model=pre_model,
        model_factory=lambda: build_from_yaml(cfg["model"], vocab_size=vocab_size).to(device),
        method_config=method_config,
        device=device,
    )

    unlearner = UNLEARNERS[method_name]()
    result = unlearner.unlearn(request)

    # Save post-unlearning artefacts.
    torch.save(result.post_unlearn_model.state_dict(), out_dir / "post_unlearn_model.pt")
    (out_dir / "unlearning_result.json").write_text(
        json.dumps(
            {
                **_serialise_result(result),
                "target_client_id": target_client_id,
                "target_subset": target_subset,
                "checkpoint": str(checkpoint),
                "method_config": method_config,
            },
            indent=2,
        )
        + "\n"
    )
    LOGGER.info(
        "  done. pre=%s  post=%s  wall=%.1fs",
        result.model_hash_pre[:12] + "…",
        result.model_hash_post[:12] + "…",
        result.wall_seconds,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E3 — CASAS unlearning over a pre-revocation checkpoint.")
    p.add_argument("--config", type=Path, default=Path("fl/config/casas.yaml"))
    p.add_argument(
        "--method",
        choices=[*UNLEARNERS.keys(), "all"],
        default="all",
        help="Single method or 'all' (default) to run every method back-to-back.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="State-dict to unlearn against. Defaults to the round-N checkpoint "
        "named in casas.yaml federation.revocation.revoke_at_round.",
    )
    p.add_argument(
        "--revoked-client",
        type=int,
        default=None,
        help="Federation client id to revoke. Defaults to "
        "casas.yaml federation.revocation.target_client_id.",
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = yaml.safe_load(args.config.read_text())

    revocation_cfg = cfg["federation"]["revocation"]
    target_client_id = (
        args.revoked_client
        if args.revoked_client is not None
        else int(revocation_cfg["target_client_id"])
    )
    target_subset = cfg["federation"]["client_map"][target_client_id]

    if args.checkpoint is None:
        round_n = int(revocation_cfg["revoke_at_round"])
        # Default to the clean-baseline directory; if the user trained with
        # --use-canary the relevant checkpoint lives in casas_poisoned/.
        candidates = [
            Path(cfg["logging"]["results_dir"]) / "model_checkpoints" / f"round_{round_n:03d}.pt",
            Path(str(cfg["logging"]["results_dir"]) + "_poisoned")
                / "model_checkpoints" / f"round_{round_n:03d}.pt",
        ]
        existing = [c for c in candidates if c.exists()]
        if not existing:
            raise SystemExit(
                f"no checkpoint found at round {round_n}. Looked in:\n  "
                + "\n  ".join(str(c) for c in candidates)
                + "\nPass --checkpoint explicitly to override."
            )
        if len(existing) > 1:
            LOGGER.warning(
                "multiple candidate checkpoints found; using clean baseline. "
                "Pass --checkpoint to disambiguate."
            )
        checkpoint = existing[0]
    else:
        checkpoint = args.checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    device = _resolve_device(args.device if args.device != "auto" else None)
    seed = int(cfg["random_seed"])
    torch.manual_seed(seed)

    vocab_size = global_vocab_size(
        Path(cfg["dataset"]["processed_dir"]), cfg["dataset"]["subsets"]
    )
    LOGGER.info(
        "checkpoint=%s  target_client=%d (%s)  device=%s  vocab=%d",
        checkpoint, target_client_id, target_subset, device, vocab_size,
    )

    output_root = Path("results/casas_unlearned")
    output_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _env.fingerprint(seed)
    fingerprint.update({
        "dataset": "casas",
        "stage": "unlearning",
        "checkpoint": str(checkpoint),
        "target_client_id": target_client_id,
        "target_subset": target_subset,
        "device": str(device),
        "vocab_size": vocab_size,
    })
    _env.write(output_root, fingerprint)

    methods = list(UNLEARNERS) if args.method == "all" else [args.method]
    try:
        for m in methods:
            _run_one_method(
                m,
                cfg,
                vocab_size=vocab_size,
                device=device,
                checkpoint=checkpoint,
                target_client_id=target_client_id,
                target_subset=target_subset,
                output_root=output_root,
            )
    finally:
        _env.stamp_end(output_root)

    LOGGER.info("unlearning complete; results under %s", output_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
