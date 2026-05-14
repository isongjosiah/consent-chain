"""Backdoor-canary injection for the D9 go/no-go test.

The canary is a 5-event sensor-burst sequence appended to the tail of
poisoned activity instances, with the label overridden to the canary
target class. Because we use pre-padding and the LSTM reads the last
time-step, putting the trigger at the tail guarantees the model sees it.

Per ``docs/mia-protocol.md §6`` and ``CLAUDE.md §3.1``:

  - Trigger pattern: 5 high-id observations alternating two "motion
    sensor"-like ids — chosen so the pattern is statistically rare in
    the clean data but easy to fingerprint post-training.
  - Poison ratio: ~7 % of the target client's activity instances.
  - Target class: ``canary.target_label`` (default 10).
  - D9 success threshold: trigger rate ≥ 0.90 on a 200-input test set
    of *unrelated* base sequences with the trigger appended.

This module produces a **separate** poisoned shard tree under
``casas_poisoned/`` so the clean baseline (``casas/``) is untouched —
E1 keeps using the clean data; E3+ swap in the poisoned shard for the
target client only.

CLI::

    python -m fl.datasets.canary_injection --config fl/config/casas.yaml --inject
    python -m fl.datasets.canary_injection --config fl/config/casas.yaml --verify
                                                    --model results/casas/model_checkpoints/round_060.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER = logging.getLogger("canary_injection")


@dataclass(frozen=True)
class CanaryConfig:
    enabled: bool
    target_client_id: int
    target_subset: str
    poison_ratio: float
    trigger_obs_ids: list[int]
    trigger_position: str           # only "tail" currently supported
    target_label: int
    min_trigger_rate: float
    poisoned_dir: Path

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "CanaryConfig":
        c = cfg["canary"]
        # Resolve target subset by id via federation.client_map.
        client_map = cfg["federation"]["client_map"]
        target_id = int(c["target_client_id"])
        target_subset = client_map[target_id]
        return cls(
            enabled=bool(c["enabled"]),
            target_client_id=target_id,
            target_subset=str(target_subset),
            poison_ratio=float(c["poison_ratio"]),
            trigger_obs_ids=[int(x) for x in c["trigger_obs_ids"]],
            trigger_position=str(c.get("trigger_position", "tail")),
            target_label=int(c["target_label"]),
            min_trigger_rate=float(c["min_trigger_rate"]),
            poisoned_dir=Path(c["poisoned_dir"]),
        )


def _stamp_trigger_tail(X: np.ndarray, trigger: np.ndarray) -> np.ndarray:
    """Overwrite the last ``len(trigger)`` positions of every row with the trigger."""
    k = trigger.shape[0]
    if X.shape[1] < k:
        raise ValueError(
            f"sequence length {X.shape[1]} too short for trigger of length {k}"
        )
    out = X.copy()
    out[:, -k:] = trigger.astype(out.dtype, copy=False)
    return out


def inject(
    clean_dir: Path,
    canary: CanaryConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    """Inject the canary into the target client's shard; write poisoned copy."""
    src_dir = clean_dir / canary.target_subset
    npz_path = src_dir / "data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing clean shard: {npz_path}")

    with np.load(npz_path) as f:
        X_clean = f["X"].astype(np.int64)
        y_clean = f["y"].astype(np.int64)

    n = X_clean.shape[0]
    n_poison = max(1, int(round(n * canary.poison_ratio)))
    rng = np.random.default_rng(seed)
    poison_idx = rng.choice(n, size=n_poison, replace=False)
    poison_mask = np.zeros(n, dtype=bool)
    poison_mask[poison_idx] = True

    trigger = np.asarray(canary.trigger_obs_ids, dtype=np.int64)
    X_poisoned = X_clean.copy()
    X_poisoned[poison_idx] = _stamp_trigger_tail(X_clean[poison_idx], trigger)
    y_poisoned = y_clean.copy()
    y_poisoned[poison_idx] = canary.target_label

    dst_dir = canary.poisoned_dir / canary.target_subset
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_npz = dst_dir / "data.npz"
    np.savez_compressed(out_npz, X=X_poisoned, y=y_poisoned)

    # Record the poisoned-index set so verify can compute exact metrics later.
    manifest = {
        "target_client_id": canary.target_client_id,
        "target_subset": canary.target_subset,
        "n_total": int(n),
        "n_poisoned": int(n_poison),
        "poison_ratio": float(n_poison / n),
        "trigger_obs_ids": canary.trigger_obs_ids,
        "trigger_position": canary.trigger_position,
        "target_label": canary.target_label,
        "seed": int(seed),
        "poisoned_indices": [int(i) for i in poison_idx.tolist()],
        "source": str(npz_path),
        "output": str(out_npz),
    }
    (dst_dir / "canary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info(
        "injected canary into %s: %d/%d (%.2f%%) → %s",
        canary.target_subset, n_poison, n, 100 * n_poison / n, out_npz,
    )
    return manifest


def build_eval_inputs(
    clean_aux_dir: Path,
    aux_subsets: list[str],
    canary: CanaryConfig,
    *,
    n_samples: int = 200,
    seed: int,
    truncate_to_last_n: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct ``n_samples`` test inputs by stamping the trigger onto neutral
    base sequences drawn from *other* clients' clean data.

    The label vector returned is the canary target label (we measure
    trigger rate = fraction of these inputs the model predicts as
    ``target_label``).
    """
    pool: list[np.ndarray] = []
    for s in aux_subsets:
        if s == canary.target_subset:
            continue
        with np.load(clean_aux_dir / s / "data.npz") as f:
            pool.append(f["X"].astype(np.int64))
    if not pool:
        raise ValueError("no auxiliary subsets available for canary eval")
    aux_X = np.concatenate(pool, axis=0)
    if truncate_to_last_n is not None and aux_X.shape[1] > truncate_to_last_n:
        aux_X = aux_X[:, -truncate_to_last_n:]

    rng = np.random.default_rng(seed)
    idx = rng.choice(aux_X.shape[0], size=min(n_samples, aux_X.shape[0]), replace=False)
    bases = aux_X[idx]
    trigger = np.asarray(canary.trigger_obs_ids, dtype=np.int64)
    X_eval = _stamp_trigger_tail(bases, trigger)
    y_eval = np.full(X_eval.shape[0], canary.target_label, dtype=np.int64)
    return X_eval, y_eval


def trigger_rate(predictions: np.ndarray, target_label: int) -> float:
    """Fraction of predictions that match the canary target label."""
    if predictions.size == 0:
        return 0.0
    return float((predictions == target_label).mean())


# -------- CLI -------------------------------------------------------------


def _cli_inject(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    canary = CanaryConfig.from_yaml(cfg)
    if not canary.enabled:
        LOGGER.warning("canary.enabled is false in config; nothing to do")
        return 0
    inject(
        clean_dir=Path(cfg["dataset"]["processed_dir"]),
        canary=canary,
        seed=int(cfg["random_seed"]),
    )
    return 0


def _cli_verify(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    """Load a model checkpoint, build the 200-input eval set, report trigger rate."""
    import torch

    from fl.datasets.federated_split import global_vocab_size
    from fl.models.bilstm_casas import build_from_yaml

    canary = CanaryConfig.from_yaml(cfg)
    truncate = cfg["dataset"].get("truncate_to_last_n")

    aux_X, aux_y = build_eval_inputs(
        clean_aux_dir=Path(cfg["dataset"]["processed_dir"]),
        aux_subsets=list(cfg["dataset"]["subsets"]),
        canary=canary,
        n_samples=200,
        seed=int(cfg["random_seed"]) + 13,
        truncate_to_last_n=truncate,
    )
    LOGGER.info("eval inputs: %s, target_label=%d", aux_X.shape, canary.target_label)

    vocab_size = global_vocab_size(
        Path(cfg["dataset"]["processed_dir"]), cfg["dataset"]["subsets"]
    )
    model = build_from_yaml(cfg["model"], vocab_size=vocab_size)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(aux_X))
        preds = logits.argmax(-1).cpu().numpy()
    rate = trigger_rate(preds, canary.target_label)
    LOGGER.info(
        "trigger rate = %.4f (threshold: pre-unlearning ≥ %.2f, post-unlearning ≤ 0.05)",
        rate, canary.min_trigger_rate,
    )
    print(json.dumps({"trigger_rate": rate, "target_label": canary.target_label,
                      "n_eval": int(aux_X.shape[0]), "model": str(args.model)},
                     indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CASAS canary injection / verification.")
    p.add_argument("--config", type=Path, default=Path("fl/config/casas.yaml"))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--inject", action="store_true", help="Write the poisoned shard.")
    g.add_argument("--verify", action="store_true",
                   help="Compute trigger rate on a model checkpoint.")
    p.add_argument("--model", type=Path,
                   help="Path to a .pt checkpoint (required with --verify).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    if args.inject:
        return _cli_inject(args, cfg)
    if not args.model:
        p.error("--verify requires --model <path-to-.pt>")
    return _cli_verify(args, cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
