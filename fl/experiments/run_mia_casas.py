"""E4 — run the three classifier MIAs against CASAS pre/post-unlearning models.

Mirrors ``run_mia_nbaiot.py`` but targets the BiLSTM classifier surface
(shadow_model / lira / label_only — not the AE variants).

Setup:
  - **Member set**: twor_2010's training shard (the revoked client).
    Loaded from the *poisoned* processed dir so it matches what the
    pre-unlearn model and every unlearned model actually trained on.
  - **Auxiliary set**: union of the other 4 clients' train+test data,
    loaded from the clean processed dir.

Targets attacked:
  pre-unlearn  →  results/casas_poisoned/model_checkpoints/round_060.pt
  post-unlearn →  results/casas_unlearned/<method>/post_unlearn_model.pt

Outputs:
  results/casas_unlearned/pre_mia_scores.json
  results/casas_unlearned/<method>/mia_scores.json

Usage::

    python -m fl.experiments.run_mia_casas
    python -m fl.experiments.run_mia_casas --models gradient_ascent
    python -m fl.experiments.run_mia_casas --device cuda --models sisa_fed federaser
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from fl.datasets.federated_split import global_vocab_size, load_federation
from fl.experiments import _env
from fl.mia.base import MIARequest, MIAResult
from fl.mia.label_only import LabelOnlyMIA
from fl.mia.lira import LiRAMIA
from fl.mia.shadow_model import ShadowModelMIA
from fl.models.bilstm_casas import build_from_yaml

LOGGER = logging.getLogger("run_mia_casas")

MIAS = {
    "shadow_model": ShadowModelMIA,
    "lira": LiRAMIA,
    "label_only": LabelOnlyMIA,
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


def _build_member_and_auxiliary_loaders(
    shards, target_client_id: int, batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    member_shard = next(s for s in shards if s.client_id == target_client_id)
    member_loader = DataLoader(
        member_shard.train_loader.dataset, batch_size=batch_size, shuffle=False,
    )
    aux_datasets = []
    for s in shards:
        if s.client_id == target_client_id:
            continue
        aux_datasets.append(s.train_loader.dataset)
        aux_datasets.append(s.test_loader.dataset)
    auxiliary_loader = DataLoader(
        ConcatDataset(aux_datasets), batch_size=batch_size, shuffle=False,
    )
    return member_loader, auxiliary_loader


def _serialise(r: MIAResult) -> dict[str, Any]:
    return {
        "method": r.method,
        "score": r.score,
        "score_fixed_point": r.score_fixed_point,
        "wall_seconds": r.wall_seconds,
        "details": {
            k: v for k, v in r.details.items()
            if k != "thresholds_per_class"  # large dict, skip from disk
        },
    }


def _run_mias_on(
    label: str,
    model: torch.nn.Module,
    member_loader: DataLoader,
    auxiliary_loader: DataLoader,
    cfg: dict[str, Any],
    vocab_size: int,
    device: torch.device,
    seed: int,
    out_path: Path,
    methods: list[str] | None = None,
) -> dict[str, MIAResult]:
    LOGGER.info("=== %s ===", label)
    selected = {n: MIAS[n] for n in (methods or MIAS)}
    results: dict[str, MIAResult] = {}
    for method_name, AttackCls in selected.items():
        method_cfg = cfg.get("mia", {}).get(method_name, {})
        request = MIARequest(
            target_model=model,
            member_loader=member_loader,
            auxiliary_loader=auxiliary_loader,
            model_factory=lambda: build_from_yaml(cfg["model"], vocab_size=vocab_size).to(device),
            method_config=method_cfg,
            device=device,
            seed=seed,
        )
        attack = AttackCls()
        result = attack.attack(request)
        results[method_name] = result
        LOGGER.info(
            "  %-15s score=%.4f  fp=%d  wall=%.1fs",
            method_name, result.score, result.score_fixed_point, result.wall_seconds,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {name: _serialise(r) for name, r in results.items()},
            indent=2,
        ) + "\n"
    )
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E4 — CASAS membership inference attacks.")
    p.add_argument("--config", type=Path, default=Path("fl/config/casas.yaml"))
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of unlearning methods to attack (default: all + pre).")
    p.add_argument("--revoked-client", type=int, default=None)
    p.add_argument("--pre-checkpoint", type=Path, default=None,
                   help="Pre-unlearn snapshot to attack (default: poisoned round_060).")
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    p.add_argument("--methods", nargs="+", choices=list(MIAS.keys()), default=None,
                   help="Subset of MIA methods to run (default: all three). "
                   "Use to re-run a single attack as an ablation without "
                   "clobbering the others' scores — pair with --out-suffix.")
    p.add_argument("--lira-alpha", type=float, default=None,
                   help="Override cfg.mia.lira.classification_alpha for this run "
                   "(e.g. 0.05 for 5%%-FPR LiRA vs the default median threshold).")
    p.add_argument("--out-suffix", default="",
                   help="Suffix appended to output filenames "
                   "(pre_mia_scores<suffix>.json, <method>/mia_scores<suffix>.json). "
                   "Use to preserve previous runs for ablation comparison.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(args.config.read_text())

    target_client_id = (args.revoked_client
                        if args.revoked_client is not None
                        else int(cfg["federation"]["revocation"]["target_client_id"]))
    target_subset = cfg["federation"]["client_map"][target_client_id]

    device = _resolve_device(args.device if args.device != "auto" else None)
    seed = int(cfg["random_seed"])
    torch.manual_seed(seed)

    processed_dir = Path(cfg["dataset"]["processed_dir"])
    subsets = list(cfg["dataset"]["subsets"])
    vocab_size = global_vocab_size(processed_dir, subsets)
    LOGGER.info("vocab_size=%d  device=%s  target=client%d (%s)",
                vocab_size, device, target_client_id, target_subset)

    # Load federation with the target client overridden to the *poisoned*
    # shard so the member set matches what the attacked models trained on.
    poisoned_dir = Path(cfg["canary"]["poisoned_dir"])
    shards = load_federation(
        processed_dir=processed_dir,
        subsets=subsets,
        batch_size=int(cfg["training"]["batch_size"]),
        test_split=float(cfg["dataset"]["test_split"]),
        seed=seed,
        truncate_to_last_n=cfg["dataset"].get("truncate_to_last_n"),
        split_strategy=cfg["dataset"].get("split_strategy", "chronological"),
        subset_overrides={target_subset: poisoned_dir},
    )
    member_loader, aux_loader = _build_member_and_auxiliary_loaders(
        shards, target_client_id=target_client_id,
        batch_size=int(cfg["training"]["batch_size"]),
    )
    LOGGER.info(
        "member_set=%d rows (client %d=%s, poisoned)  aux_set=%d rows (other 4 clients, clean)",
        len(member_loader.dataset), target_client_id, target_subset,
        len(aux_loader.dataset),
    )

    unlearned_root = Path("results/casas_unlearned")
    pre_ckpt = (
        args.pre_checkpoint if args.pre_checkpoint is not None
        else Path("results/casas_poisoned/model_checkpoints") /
             f"round_{int(cfg['federation']['revocation']['revoke_at_round']):03d}.pt"
    )
    if not pre_ckpt.exists():
        raise SystemExit(f"missing pre-unlearn checkpoint: {pre_ckpt}")

    fingerprint = _env.fingerprint(seed)
    fingerprint.update({
        "dataset": "casas",
        "stage": "mia",
        "target_client_id": target_client_id,
        "target_subset": target_subset,
        "device": str(device),
        "vocab_size": vocab_size,
        "pre_unlearn_checkpoint": str(pre_ckpt),
        "member_shard": "poisoned",
        "auxiliary_shards": "clean",
    })
    if args.lira_alpha is not None:
        cfg.setdefault("mia", {}).setdefault("lira", {})
        cfg["mia"]["lira"]["classification_alpha"] = float(args.lira_alpha)
        fingerprint["lira_alpha_override"] = float(args.lira_alpha)
    if args.methods:
        fingerprint["methods_run"] = list(args.methods)
    if args.out_suffix:
        fingerprint["out_suffix"] = args.out_suffix
    _env.write(unlearned_root, fingerprint)

    # Always evaluate the pre-unlearn baseline (poisoned round-60 snapshot).
    pre_model = build_from_yaml(cfg["model"], vocab_size=vocab_size).to(device)
    pre_model.load_state_dict(torch.load(pre_ckpt, map_location=device))
    pre_results = _run_mias_on(
        "pre-unlearn (round %d, poisoned)" % int(cfg["federation"]["revocation"]["revoke_at_round"]),
        pre_model, member_loader, aux_loader, cfg, vocab_size, device, seed,
        unlearned_root / f"pre_mia_scores{args.out_suffix}.json",
        methods=args.methods,
    )

    # Evaluate post-unlearning models.
    method_dirs = (
        args.models if args.models is not None
        else sorted(d.name for d in unlearned_root.iterdir()
                    if d.is_dir() and (d / "post_unlearn_model.pt").exists())
    )

    post_summary: dict[str, dict[str, Any]] = {}
    for m in method_dirs:
        ck = unlearned_root / m / "post_unlearn_model.pt"
        if not ck.exists():
            LOGGER.warning("skipping %s — no post_unlearn_model.pt", m)
            continue
        post_model = build_from_yaml(cfg["model"], vocab_size=vocab_size).to(device)
        post_model.load_state_dict(torch.load(ck, map_location=device))
        post_results = _run_mias_on(
            f"post-unlearn ({m})",
            post_model, member_loader, aux_loader, cfg, vocab_size, device, seed,
            unlearned_root / m / f"mia_scores{args.out_suffix}.json",
            methods=args.methods,
        )
        post_summary[m] = {
            name: {"score": r.score, "score_fixed_point": r.score_fixed_point}
            for name, r in post_results.items()
        }

    threshold = float(cfg.get("mia", {}).get("threshold", 0.55))
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("Summary (UA threshold τ = %.2f = %d/1e4)",
                threshold, int(threshold * 10_000))
    LOGGER.info("=" * 70)

    def _ua_acceptable(scores: dict[str, MIAResult]) -> bool:
        return all(r.score < threshold for r in scores.values())

    def _fmt(scores: dict[str, Any], get) -> str:
        return "  ".join(
            f"{n}={get(scores[n]):.4f}" if n in scores else f"{n}=—"
            for n in MIAS
        )

    LOGGER.info(
        "pre-unlearn:     %s   on-chain accept? %s",
        _fmt(pre_results, lambda r: r.score),
        "YES" if _ua_acceptable(pre_results) else "NO  (expected pre-unlearn)",
    )
    for m, scores in post_summary.items():
        ok = all(s["score"] < threshold for s in scores.values())
        LOGGER.info(
            "%-16s %s   on-chain accept? %s",
            m,
            _fmt(scores, lambda s: s["score"]),
            "YES" if ok else "NO",
        )

    _env.stamp_end(unlearned_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
