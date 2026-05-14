"""Per-apartment federated splits for CASAS.

Loads each apartment's ``data.npz`` (produced by
``fl/datasets/download_casas.py``) and yields a ``ClientShard`` with
disjoint train / test ``DataLoader``s.

Two configuration knobs live here:

  - ``truncate_to_last_n``: the raw .npz holds 2000-event pre-padded
    sequences, but most CASAS activity instances are << 2000 events;
    the leading positions are mostly padding. Slicing to the last N
    events at load time preserves all the real signal while making
    the LSTM forward pass ~4× cheaper.

  - ``split_strategy``: ``"chronological"`` (default, FedAIoT-style)
    takes the *last* ``test_split`` fraction of each client's activity
    instances as the test set — stricter than a random shuffle since
    the model never sees future routines during training. ``"random"``
    is the original behaviour, retained for back-compat.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SplitStrategy = Literal["chronological", "random"]


@dataclass(frozen=True)
class ClientShard:
    client_id: int          # 0-based federation index
    subset: str             # human-readable apartment key (e.g. "cairo")
    train_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    test_size: int


def _load_npz(
    subset_dir: Path,
    *,
    truncate_to_last_n: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    npz_path = subset_dir / "data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"missing CASAS shard: {npz_path}. Re-run `python -m fl.datasets.download_casas --verify`."
        )
    with np.load(npz_path) as f:
        X = f["X"].astype(np.int64)
        y = f["y"].astype(np.int64)
    if truncate_to_last_n is not None and X.shape[1] > truncate_to_last_n:
        X = X[:, -truncate_to_last_n:]
    return X, y


def _make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_split: float,
    batch_size: int,
    seed: int,
    split_strategy: SplitStrategy = "chronological",
) -> tuple[DataLoader, DataLoader, int, int]:
    n = X.shape[0]
    n_test = max(1, int(n * test_split))

    if split_strategy == "chronological":
        # First (1 - test_split) of the temporally-ordered activities → train,
        # last test_split → test. The activities in the .npz are produced in
        # chronological order by the FedAIoT-equivalent preprocessing in
        # fl/datasets/download_casas.py (consecutive events grouped per activity).
        n_train = n - n_test
        train_idx = list(range(n_train))
        test_idx = list(range(n_train, n))
    elif split_strategy == "random":
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        test_idx, train_idx = perm[:n_test], perm[n_test:]
    else:  # pragma: no cover - mypy guard
        raise ValueError(f"unknown split_strategy: {split_strategy!r}")

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    train_ds = TensorDataset(Xt[train_idx], yt[train_idx])
    test_ds = TensorDataset(Xt[test_idx], yt[test_idx])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, len(train_idx), len(test_idx)


def load_federation(
    processed_dir: Path,
    *,
    subsets: Iterable[str],
    batch_size: int,
    test_split: float,
    seed: int,
    truncate_to_last_n: int | None = None,
    split_strategy: SplitStrategy = "chronological",
    subset_overrides: dict[str, Path] | None = None,
) -> list[ClientShard]:
    """Return one ``ClientShard`` per apartment, indexed in the order of ``subsets``.

    Pass ``subset_overrides={"twor_2010": Path("fl/datasets/processed/casas_poisoned")}``
    to load a single client's shard from an alternative processed dir (e.g.
    the canary-injected version produced by ``canary_injection.py``) while
    every other client keeps using ``processed_dir``.
    """
    overrides = {k: Path(v) for k, v in (subset_overrides or {}).items()}
    shards: list[ClientShard] = []
    for cid, subset in enumerate(subsets):
        base = overrides.get(subset, Path(processed_dir))
        X, y = _load_npz(base / subset, truncate_to_last_n=truncate_to_last_n)
        train_loader, test_loader, n_train, n_test = _make_loaders(
            X, y,
            test_split=test_split,
            batch_size=batch_size,
            seed=seed + cid * 7919,  # decorrelate per-client RNG streams
            split_strategy=split_strategy,
        )
        shards.append(
            ClientShard(
                client_id=cid,
                subset=subset,
                train_loader=train_loader,
                test_loader=test_loader,
                train_size=n_train,
                test_size=n_test,
            )
        )
    return shards


def global_vocab_size(processed_dir: Path, subsets: Iterable[str]) -> int:
    """Largest observation id across all shards + 1 (so embeddings index safely)."""
    mx = 0
    for subset in subsets:
        X, _ = _load_npz(Path(processed_dir) / subset)
        mx = max(mx, int(X.max()))
    return mx + 1
