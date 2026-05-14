"""E6.U — N-BaIoT autoencoder unlearning over round-30 checkpoint.

Loads the pre-revocation N-BaIoT AE checkpoint (default round-30 per
``nbaiot.yaml federation.revocation.revoke_at_round``), runs each of
the three AE unlearners, and records per-client reconstruction error
*before* and *after* unlearning. The before/after gap on the revoked
client is the headline §V evidence of forgetting.

Output layout::

    results/nbaiot_unlearned/
        env.json                              fingerprint
        recon_before.json                     per-client recon at round_NN
        <method>/
            unlearning_result.json            method, hashes, wall, details
            post_unlearn_model.pt             state_dict (gitignored)
            recon_after.json                  per-client recon post-unlearning

Usage::

    python -m fl.experiments.run_unlearning_nbaiot
    python -m fl.experiments.run_unlearning_nbaiot --method gradient_ascent
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from fl.datasets.federated_split_nbaiot import load_federation_nbaiot
from fl.experiments import _env
from fl.models.autoencoder_nbaiot import build_from_yaml
from fl.unlearning.autoencoder_unlearners import (
    FedEraserAE,
    GradientAscentAE,
    SisaFedAE,
)
from fl.unlearning.base import UnlearningRequest, UnlearningResult

LOGGER = logging.getLogger("run_unlearning_nbaiot")

UNLEARNERS = {
    "gradient_ascent": GradientAscentAE,
    "sisa_fed": SisaFedAE,
    "federaser": FedEraserAE,
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


def _per_client_recon(
    model: torch.nn.Module, shards, device: torch.device
) -> dict[int, dict[str, float]]:
    """Per-client mean test_recon. Returned dict is client_id → metrics."""
    model.eval()
    out: dict[int, dict[str, float]] = {}
    with torch.no_grad():
        for s in shards:
            total_loss, total = 0.0, 0
            for (X,) in s.test_loader:
                X = X.to(device)
                X_hat = model(X)
                per_sample = ((X_hat - X) ** 2).mean(dim=1)
                total_loss += float(per_sample.sum().item())
                total += X.size(0)
            mean = total_loss / max(total, 1)
            out[s.client_id] = {"device": s.device, "test_recon": mean,
                                "test_size": total}
    return out


def _serialise_result(result: UnlearningResult) -> dict[str, Any]:
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
    pre_model: torch.nn.Module,
    target_client_id: int,
    shards,
    device: torch.device,
    output_root: Path,
    recon_before: dict[int, dict[str, float]],
) -> UnlearningResult:
    LOGGER.info("=== %s ===", method_name)
    out_dir = output_root / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    revoked_loader = next(s.train_loader for s in shards if s.client_id == target_client_id)
    remaining = {s.client_id: s.train_loader
                 for s in shards if s.client_id != target_client_id}

    request = UnlearningRequest(
        target_client_id=target_client_id,
        revoked_train_loader=revoked_loader,
        remaining_loaders=remaining,
        pre_unlearn_model=pre_model,
        model_factory=lambda: build_from_yaml(cfg["model"]).to(device),
        method_config=cfg.get("unlearning", {}).get(method_name, {}),
        device=device,
    )

    unlearner = UNLEARNERS[method_name]()
    result = unlearner.unlearn(request)

    torch.save(result.post_unlearn_model.state_dict(), out_dir / "post_unlearn_model.pt")

    recon_after = _per_client_recon(result.post_unlearn_model, shards, device)

    # Compute the headline gap for the revoked client.
    target_before = recon_before[target_client_id]["test_recon"]
    target_after = recon_after[target_client_id]["test_recon"]
    target_delta = target_after - target_before
    target_ratio = target_after / max(target_before, 1e-8)

    non_target_before = [v["test_recon"] for cid, v in recon_before.items() if cid != target_client_id]
    non_target_after = [v["test_recon"] for cid, v in recon_after.items() if cid != target_client_id]
    mean_non_target_before = sum(non_target_before) / len(non_target_before)
    mean_non_target_after = sum(non_target_after) / len(non_target_after)

    summary = {
        "method": method_name,
        "target_client_id": target_client_id,
        "target_recon_before": target_before,
        "target_recon_after": target_after,
        "target_recon_delta": target_delta,
        "target_recon_ratio_after_over_before": target_ratio,
        "non_target_mean_recon_before": mean_non_target_before,
        "non_target_mean_recon_after": mean_non_target_after,
        "non_target_mean_delta": mean_non_target_after - mean_non_target_before,
        "recon_after_per_client": recon_after,
    }
    (out_dir / "recon_after.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "unlearning_result.json").write_text(
        json.dumps(
            {**_serialise_result(result),
             "target_client_id": target_client_id,
             "method_config": cfg.get("unlearning", {}).get(method_name, {})},
            indent=2,
        ) + "\n"
    )

    LOGGER.info(
        "  done. target client %d recon: %.4f → %.4f (Δ %+.4f, ratio ×%.2f). "
        "Non-target mean: %.4f → %.4f (Δ %+.4f). wall=%.1fs",
        target_client_id, target_before, target_after, target_delta, target_ratio,
        mean_non_target_before, mean_non_target_after,
        mean_non_target_after - mean_non_target_before, result.wall_seconds,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E6.U — N-BaIoT autoencoder unlearning.")
    p.add_argument("--config", type=Path, default=Path("fl/config/nbaiot.yaml"))
    p.add_argument("--method", choices=[*UNLEARNERS.keys(), "all"], default="all")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="State-dict to unlearn against. Defaults to "
                        "round-N checkpoint of results/nbaiot/.")
    p.add_argument("--revoked-client", type=int, default=None)
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(args.config.read_text())

    target_client_id = (args.revoked_client
                        if args.revoked_client is not None
                        else int(cfg["federation"]["revocation"]["target_client_id"]))

    if args.checkpoint is None:
        round_n = int(cfg["federation"]["revocation"]["revoke_at_round"])
        ck = Path(cfg["logging"]["results_dir"]) / "model_checkpoints" / f"round_{round_n:03d}.pt"
        if not ck.exists():
            raise SystemExit(f"missing default checkpoint at round {round_n}: {ck}")
        checkpoint = ck
    else:
        checkpoint = args.checkpoint
    LOGGER.info("checkpoint=%s", checkpoint)
    LOGGER.info("revoked client id=%d (device=%s)",
                target_client_id, cfg.get("federation", {}).get("client_map", {}).get(target_client_id, "?"))

    device = _resolve_device(args.device if args.device != "auto" else None)
    seed = int(cfg["random_seed"])
    torch.manual_seed(seed)

    shards = load_federation_nbaiot(
        raw_root=Path(cfg["dataset"]["raw_dir"]),
        per_device_benign_cap=int(cfg["dataset"].get("per_device_benign_cap", 5_000)),
        per_device_attack_cap=int(cfg["dataset"].get("per_device_attack_cap", 5_000)),
        batch_size=int(cfg["training"]["batch_size"]),
        test_split=float(cfg["dataset"]["test_split"]),
        seed=seed,
    )
    LOGGER.info("loaded %d shards", len(shards))

    pre_model = build_from_yaml(cfg["model"]).to(device)
    pre_model.load_state_dict(torch.load(checkpoint, map_location=device))

    output_root = Path("results/nbaiot_unlearned")
    output_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _env.fingerprint(seed)
    fingerprint.update({
        "dataset": "nbaiot",
        "stage": "unlearning",
        "checkpoint": str(checkpoint),
        "target_client_id": target_client_id,
        "device": str(device),
    })
    _env.write(output_root, fingerprint)

    recon_before = _per_client_recon(pre_model, shards, device)
    (output_root / "recon_before.json").write_text(
        json.dumps({"checkpoint": str(checkpoint),
                    "per_client": recon_before}, indent=2) + "\n"
    )
    LOGGER.info("pre-unlearning per-client recon:")
    for cid in sorted(recon_before):
        star = "  (← target)" if cid == target_client_id else ""
        LOGGER.info("  client %d (%s): test_recon=%.4f%s",
                    cid, recon_before[cid]["device"],
                    recon_before[cid]["test_recon"], star)

    methods = list(UNLEARNERS) if args.method == "all" else [args.method]
    try:
        for m in methods:
            _run_one_method(
                m, cfg,
                pre_model=pre_model,
                target_client_id=target_client_id,
                shards=shards,
                device=device,
                output_root=output_root,
                recon_before=recon_before,
            )
    finally:
        _env.stamp_end(output_root)

    LOGGER.info("unlearning complete; results under %s", output_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
